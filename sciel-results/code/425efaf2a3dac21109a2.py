"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _energy_cartesian(points):
    """Compute logarithmic energy from Cartesian points using math.fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return float(math.fsum(terms))


def _energy_and_grad_spherical(params):
    """
    Compute logarithmic energy and gradients with respect to spherical coordinates.
    params: (N, 2) array of [theta, phi]
    theta in [0, pi], phi in [0, 2*pi]
    """
    n = len(params)
    theta = params[:, 0]
    phi = params[:, 1]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Avoid numerical issues at poles
    sin_theta_safe = np.maximum(sin_theta, 1e-15)

    pts = np.column_stack([
        sin_theta_safe * cos_phi,
        sin_theta_safe * sin_phi,
        cos_theta
    ])

    # Compute pairwise dot products
    c = pts @ pts.T

    # Mask diagonal
    idx = np.triu_indices(n, k=1)
    c_ij = c[idx]

    # Energy E = -sum log(sqrt(2 - 2c)) = -0.5 sum log(2 - 2c)
    val = 2.0 - 2.0 * c_ij
    val = np.maximum(val, 1e-15)

    energy = -0.5 * np.sum(np.log(val))

    # Gradients
    cos_dphi = np.cos(phi[:, np.newaxis] - phi[np.newaxis, :])
    sin_dphi = np.sin(phi[:, np.newaxis] - phi[np.newaxis, :])

    # dc/dtheta_i
    dc_dti = (cos_theta[:, np.newaxis] * sin_theta[np.newaxis, :] * cos_dphi
              - sin_theta[:, np.newaxis] * cos_theta[np.newaxis, :])

    # dc/dphi_i
    dc_dphi_i = -sin_theta[:, np.newaxis] * sin_theta[np.newaxis, :] * sin_dphi

    denom = 1.0 - c
    mask = ~np.eye(n, dtype=bool)

    with np.errstate(divide='ignore', invalid='ignore'):
        grad_theta_mat = np.where(mask, dc_dti / np.where(denom > 1e-15, denom, 1.0), 0.0)
        grad_phi_mat = np.where(mask, dc_dphi_i / np.where(denom > 1e-15, denom, 1.0), 0.0)

    grad_theta = np.sum(grad_theta_mat, axis=1)
    grad_phi = np.sum(grad_phi_mat, axis=1)

    grads = np.column_stack([grad_theta, grad_phi])

    return energy, grads


def _energy_and_grad_tangent(points):
    """Compute energy and tangent-space gradient for Cartesian points on sphere."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists = np.maximum(dists, 1e-15)

    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    inv_d2[~mask] = 0.0

    # grad_i = -sum_{j!=i} (xi-xj) / |xi-xj|^2
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)

    # Project to tangent space
    grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points

    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = float(np.sum(np.log(1.0 / d)))

    return energy, grad_tangent


def _spherical_to_cartesian(params):
    """Convert spherical coordinates to Cartesian on unit sphere."""
    theta = params[:, 0]
    phi = params[:, 1]
    st = np.sin(theta)
    ct = np.cos(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    pts = np.column_stack([st * cp, st * sp, ct])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)


def _cartesian_to_spherical(pts):
    """Convert Cartesian points to spherical coordinates."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.column_stack([theta, phi])


def _make_fibonacci_sphere_params():
    """Generate Fibonacci spiral points in spherical coordinates."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    theta = np.arccos(z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    return np.column_stack([theta, phi])


def _make_ring_params(n_rings=5, rng=None):
    """Generate ring-based initialization: points distributed in latitude rings."""
    if rng is None:
        rng = np.random.default_rng(42)
    params = np.zeros((N, 2))
    points_per_ring = N // n_rings
    remainder = N % n_rings
    idx = 0
    for r in range(n_rings):
        count = points_per_ring + (1 if r < remainder else 0)
        # Spread rings in z from -1 to 1
        z = -1.0 + 2.0 * (r + 0.5) / n_rings
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        # Add some jitter to theta
        theta += rng.standard_normal(count) * 0.05
        theta = np.clip(theta, 1e-6, np.pi - 1e-6)
        # Evenly spaced phi with offset
        phi = 2.0 * np.pi * np.arange(count) / count + r * 0.3
        params[idx:idx+count, 0] = theta
        params[idx:idx+count, 1] = phi % (2 * np.pi)
        idx += count
    return params


def _random_sphere_params(rng):
    """Generate random spherical coordinates."""
    theta = rng.uniform(0, np.pi, N)
    phi = rng.uniform(0, 2 * np.pi, N)
    return np.column_stack([theta, phi])


def _adam_spherical(params0, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, max_iters=3000,
                     adaptive_lr=False, lr_decay=0.9995, min_lr=1e-6):
    """Adam optimizer in spherical coordinates with optional adaptive LR."""
    n = len(params0)
    params = params0.copy()
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    t = 0
    current_lr = lr

    best_energy = _energy_and_grad_spherical(params)[0]
    best_params = params.copy()

    for it in range(max_iters):
        if _remaining_time() < 15.0:
            break

        energy, grad = _energy_and_grad_spherical(params)

        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()

        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Per-element adaptive step size
        if adaptive_lr:
            # Scale by inverse gradient magnitude per coordinate
            grad_mag = np.sqrt(np.sum(grad ** 2, axis=1, keepdims=True)) + 1e-10
            per_point_lr = current_lr / np.sqrt(grad_mag[:, 0] + 1e-10)
            per_point_lr = np.clip(per_point_lr, min_lr, current_lr * 10)
            update = -per_point_lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            update = -current_lr * m_hat / (np.sqrt(v_hat) + eps)

        # Update parameters
        params = params + update

        # Clip theta to valid range
        params[:, 0] = np.clip(params[:, 0], 1e-6, np.pi - 1e-6)
        # Wrap phi
        params[:, 1] = params[:, 1] % (2 * np.pi)

        if adaptive_lr:
            current_lr = max(current_lr * lr_decay, min_lr)

    return best_params, best_energy


def _adam_tangent(points0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iters=3000,
                  adaptive_lr=False):
    """Adam optimizer on the tangent space of the sphere."""
    n = len(points0)
    points = _project_to_sphere(points0.copy())
    m = np.zeros_like(points)
    v = np.zeros_like(points)
    t = 0
    current_lr = lr

    best_energy = _energy_cartesian(points)
    best_points = points.copy()

    for it in range(max_iters):
        if _remaining_time() < 15.0:
            break

        energy, grad = _energy_and_grad_tangent(points)

        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()

        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        if adaptive_lr:
            grad_mag = np.sqrt(np.sum(grad ** 2, axis=1, keepdims=True)) + 1e-10
            per_point_lr = current_lr / np.sqrt(grad_mag + 1e-10)
            per_point_lr = np.clip(per_point_lr, 1e-6, current_lr * 10)
            update = -per_point_lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            update = -current_lr * m_hat / (np.sqrt(v_hat) + eps)

        # Project update to tangent space
        update_tangent = update - np.sum(update * points, axis=1, keepdims=True) * points

        # Move and project back to sphere
        points = points + update_tangent
        points = _project_to_sphere(points)

        if adaptive_lr:
            current_lr *= 0.9995

    return best_points, best_energy


def _lbfgs_spherical(params0, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """L-BFGS-B optimization in spherical coordinates."""
    if _remaining_time() < 15.0:
        return None, float('inf')

    try:
        res = minimize(
            _energy_and_grad_spherical,
            params0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        return res.x, res.fun
    except Exception:
        return None, float('inf')


def _lbfgs_tangent(points0, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """L-BFGS optimization in Cartesian with projection to sphere."""
    if _remaining_time() < 15.0:
        return None, float('inf')

    n = len(points0)
    points = _project_to_sphere(points0.copy())
    x0 = points.flatten()

    def objective(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, _ = _energy_and_grad_tangent(pts)
        return e

    def gradient(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        e, g = _energy_and_grad_tangent(pts)
        return g.flatten()

    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        pts = _project_to_sphere(res.x.reshape(n, 3))
        return pts, float(res.fun)
    except Exception:
        return None, float('inf')


def _perturb_params(params, scale, rng):
    """Perturb spherical coordinates by given scale."""
    perturbed = params.copy()
    noise = rng.standard_normal((N, 2)) * scale
    perturbed[:, 0] += noise[:, 0]
    perturbed[:, 1] += noise[:, 1]
    perturbed[:, 0] = np.clip(perturbed[:, 0], 1e-6, np.pi - 1e-6)
    perturbed[:, 1] = perturbed[:, 1] % (2 * np.pi)
    return perturbed


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, 1e-15)


def _high_precision_polish(params, maxiter=5000):
    """Final high-precision polish with very tight tolerances."""
    if _remaining_time() < 10.0:
        return params, _energy_and_grad_spherical(params)[0]

    # First a tight L-BFGS
    params_l, energy_l = _lbfgs_spherical(params, maxiter=3000, ftol=1e-16, gtol=1e-13)
    if params_l is not None:
        params = params_l

    # Then a very fine Adam pass
    if _remaining_time() > 10.0:
        params_a, energy_a = _adam_spherical(params, lr=0.0002, max_iters=3000,
                                             adaptive_lr=True, lr_decay=0.999, min_lr=1e-7)
        if energy_a < _energy_and_grad_spherical(params)[0]:
            params = params_a

    # Final L-BFGS with tightest tolerance
    if _remaining_time() > 8.0:
        params_f, energy_f = _lbfgs_spherical(params, maxiter=3000, ftol=1e-16, gtol=1e-14)
        if params_f is not None:
            params = params_f

    return params, _energy_and_grad_spherical(params)[0]


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    global _START_TIME
    _START_TIME = time.time()

    rng = np.random.default_rng(42)

    best_params = None
    best_energy = float('inf')
    best_pts = None

    def _update_best(params, energy):
        nonlocal best_params, best_energy, best_pts
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()
            best_pts = _spherical_to_cartesian(params)

    # Phase 1: Multiple diverse initializations
    # 1a: Fibonacci spiral
    fib_params = _make_fibonacci_sphere_params()
    params_opt, energy_opt = _lbfgs_spherical(fib_params, maxiter=1500, ftol=1e-13, gtol=1e-10)
    if params_opt is not None and energy_opt < best_energy:
        _update_best(params_opt, energy_opt)

    # 1b: Ring-based initialization
    if _remaining_time() > 60.0:
        ring_params = _make_ring_params(n_rings=5, rng=rng)
        params_r, energy_r = _lbfgs_spherical(ring_params, maxiter=1500, ftol=1e-13, gtol=1e-10)
        if params_r is not None and energy_r < best_energy:
            _update_best(params_r, energy_r)

    # 1c: Another ring config
    if _remaining_time() > 55.0:
        ring_params2 = _make_ring_params(n_rings=10, rng=rng)
        params_r2, energy_r2 = _lbfgs_spherical(ring_params2, maxiter=1000, ftol=1e-13, gtol=1e-10)
        if params_r2 is not None and energy_r2 < best_energy:
            _update_best(params_r2, energy_r2)

    # Phase 2: Adam from best so far
    if best_params is not None and _remaining_time() > 45.0:
        params_adam, energy_adam = _adam_spherical(best_params, lr=0.003, max_iters=2500,
                                                   adaptive_lr=True)
        if energy_adam < best_energy:
            _update_best(params_adam, energy_adam)

    # Phase 3: Multi-scale perturbations with adaptive Adam + L-BFGS polish
    if best_params is not None and _remaining_time() > 35.0:
        scales = [0.03, 0.08, 0.15, 0.25, 0.4]
        for scale in scales:
            if _remaining_time() < 20.0:
                break
            for attempt in range(3):
                if _remaining_time() < 15.0:
                    break
                perturbed = _perturb_params(best_params, scale, rng)
                # Quick Adam pass with adaptive LR
                params_a, energy_a = _adam_spherical(perturbed, lr=0.002, max_iters=1000,
                                                     adaptive_lr=True)
                if energy_a < best_energy:
                    _update_best(params_a, energy_a)
                # Follow with L-BFGS polish
                if _remaining_time() > 15.0:
                    params_l, energy_l = _lbfgs_spherical(params_a, maxiter=500,
                                                           ftol=1e-14, gtol=1e-11)
                    if params_l is not None and energy_l < best_energy:
                        _update_best(params_l, energy_l)

    # Phase 4: Random restarts with tangent-space Adam
    if _remaining_time() > 25.0:
        for r in range(6):
            if _remaining_time() < 15.0:
                break
            rand_params = _random_sphere_params(rng)
            rand_pts = _spherical_to_cartesian(rand_params)

            # Use tangent-space Adam for diversity
            pts_t, energy_t = _adam_tangent(rand_pts, lr=0.008, max_iters=1500,
                                            adaptive_lr=True)
            # Convert back to spherical for comparison
            params_t = _cartesian_to_spherical(pts_t)
            energy_sph = _energy_and_grad_spherical(params_t)[0]
            if energy_sph < best_energy:
                _update_best(params_t, energy_sph)

            if _remaining_time() > 15.0:
                params_l, energy_l = _lbfgs_spherical(params_t, maxiter=800,
                                                       ftol=1e-14, gtol=1e-10)
                if params_l is not None and energy_l < best_energy:
                    _update_best(params_l, energy_l)

    # Phase 5: High-precision final polish
    if best_params is not None and _remaining_time() > 12.0:
        params_final, energy_final = _high_precision_polish(best_params)
        if energy_final < best_energy:
            _update_best(params_final, energy_final)

    # Fallback if nothing worked
    if best_pts is None:
        fib_params = _make_fibonacci_sphere_params()
        best_pts = _spherical_to_cartesian(fib_params)

    # Final validation and correction
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)

    # Check for close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    if min_dist < 1e-9:
        for _ in range(200):
            min_idx = np.argmin(dist[iu])
            i, j = iu[0][min_idx], iu[1][min_idx]
            rand_dir = rng.standard_normal(3)
            rand_dir /= np.linalg.norm(rand_dir)
            best_pts[j] = best_pts[j] + 1e-6 * rand_dir
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            if np.min(dist[iu]) >= 1e-9:
                break

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)

    return best_pts

# EVOLVE-BLOCK-END

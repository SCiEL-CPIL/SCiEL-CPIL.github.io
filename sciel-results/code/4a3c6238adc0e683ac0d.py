"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
from scipy.optimize import minimize


def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and tangent-space gradient on the sphere.
    pts: (N, 3) array of points assumed to be on the unit sphere.
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-12)

    # E = sum_{i<j} log(1/d_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    energy = -0.25 * np.sum(log_dist_sq)

    # grad_raw_i = 0.5 * sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)

    term1 = inv_dist_sq @ pts
    term2 = pts * np.sum(inv_dist_sq, axis=1)[:, None]
    grad_raw = 0.5 * (term1 - term2)

    # Project to tangent space
    dot = np.sum(grad_raw * pts, axis=1)
    grad_tan = grad_raw - dot[:, None] * pts

    return energy, grad_tan


def _log_energy(pts):
    """Compute log(1/d) energy sum over all pairs using chordal distances (fsum for exactness)."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d = np.clip(d, 1e-15, None)
    terms = [-math.log(di) for di in d]
    return math.fsum(terms)


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _initial_random(rng):
    """Random points on sphere."""
    pts = rng.standard_normal((N, 3))
    return _project_to_sphere(pts)


def _conjugate_gradient(pts, max_iter=2000, tol=1e-10, deadline=None):
    """
    Conjugate Gradient method on the sphere with strong Wolfe line search.
    Uses the vectorized tangent gradient.
    """
    x = pts.copy()
    energy, grad = _energy_and_tangent_grad(x)
    p = -grad
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    grad_prev = grad.copy()

    c1 = 1e-4
    c2 = 0.9
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        dir_deriv = np.sum(grad * p)
        if dir_deriv > 0:
            p = -grad
            dot_p_x = np.sum(p * x, axis=1)
            p = p - dot_p_x[:, None] * x
            dir_deriv = np.sum(grad * p)

        alpha = 1.0
        found_step = False
        while alpha > 1e-12:
            x_new = x + alpha * p
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_tangent_grad(x_new)

            if e_new <= energy + c1 * alpha * dir_deriv:
                new_dir_deriv = np.sum(g_new * p)
                if new_dir_deriv >= c2 * dir_deriv:
                    found_step = True
                    break
                # Curvature failed: try expanding
                alpha_try = alpha
                while alpha_try < 10.0:
                    alpha_try *= 1.5
                    x_try = x + alpha_try * p
                    x_try = _project_to_sphere(x_try)
                    e_try, g_try = _energy_and_tangent_grad(x_try)
                    if e_try <= energy + c1 * alpha_try * dir_deriv:
                        if np.sum(g_try * p) >= c2 * dir_deriv:
                            x_new, e_new, g_new = x_try, e_try, g_try
                            alpha = alpha_try
                        found_step = True
                        break
                    else:
                        found_step = True
                        break
                if not found_step:
                    found_step = True
                break
            else:
                alpha *= 0.5

        if not found_step:
            break

        x = x_new
        energy = e_new
        grad = g_new

        if np.linalg.norm(grad) < tol:
            break

        grad_dot = np.sum(grad * (grad - grad_prev))
        grad_prev_dot = np.sum(grad_prev * grad_prev)
        beta = 0.0 if grad_prev_dot < 1e-16 else max(0.0, grad_dot / grad_prev_dot)
        p_new = -grad + beta * p
        dot_p_new_x = np.sum(p_new * x, axis=1)
        p_new = p_new - dot_p_new_x[:, None] * x
        if np.sum(grad * p_new) > 0:
            p_new = -grad
            dot_p_new_x = np.sum(p_new * x, axis=1)
            p_new = p_new - dot_p_new_x[:, None] * x
        p = p_new
        grad_prev = grad.copy()

    return x, _log_energy(x)


def _optimize_lbfgs(pts, max_iter=5000, ftol=1e-12, gtol=1e-10, deadline=None):
    """Optimize using L-BFGS-B with tangent-projected gradient."""
    x0 = pts.flatten()

    def objective(x):
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, pg = _energy_and_tangent_grad(p)
        return e, pg.flatten()

    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    try:
        result = minimize(
            objective, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxfun': max_iter * 3}
        )
        result_pts = _project_to_sphere(result.x.reshape(N, 3))
        result_energy = _log_energy(result_pts)
        if result_energy < best_energy:
            best_pts = result_pts
            best_energy = result_energy
    except Exception:
        pass

    return best_pts, best_energy


def _basin_hopping(pts, n_iter=30, stepsize=0.1, deadline=None):
    """Basin hopping with CG polish at each step (vectorized gradient)."""
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    current_pts = pts.copy()
    current_energy = best_energy
    rng = np.random.default_rng(12345)

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        perturbed = current_pts.copy()
        for i in range(N):
            v = rng.standard_normal(3)
            v = v - np.dot(v, perturbed[i]) * perturbed[i]
            nv = np.linalg.norm(v)
            if nv > 1e-12:
                v = v / nv
                perturbed[i] = perturbed[i] + stepsize * v
                perturbed[i] = perturbed[i] / np.linalg.norm(perturbed[i])

        polished, polished_energy = _conjugate_gradient(perturbed, max_iter=400, tol=1e-9, deadline=deadline)

        delta_e = polished_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / 0.5):
            current_pts = polished
            current_energy = polished_energy
        if current_energy < best_energy:
            best_pts = current_pts.copy()
            best_energy = current_energy

    return best_pts, best_energy


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    pts = _initial_fibonacci()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    # Phase 1: CG from Fibonacci
    if time.time() < deadline - 30:
        c, e = _conjugate_gradient(pts, max_iter=3000, tol=1e-11, deadline=deadline)
        if e < best_energy:
            best_pts, best_energy = c, e

    # Phase 2: Multiple diverse restarts
    rng = np.random.default_rng(42)
    restart_configs = []
    for mag in [0.05, 0.1, 0.2]:
        cfg = _project_to_sphere(_initial_fibonacci() + rng.standard_normal((N, 3)) * mag)
        restart_configs.append(cfg)
    for _ in range(4):
        restart_configs.append(_initial_random(rng))

    for cfg in restart_configs:
        if time.time() >= deadline - 30:
            break
        remaining = deadline - time.time()
        if remaining < 8:
            break
        # CG then L-BFGS polish
        c, e = _conjugate_gradient(cfg, max_iter=2000, tol=1e-10, deadline=deadline)
        if e < best_energy:
            best_pts, best_energy = c, e
        if time.time() < deadline - 20:
            c2, e2 = _optimize_lbfgs(best_pts, max_iter=2000, ftol=1e-11, gtol=1e-9, deadline=deadline)
            if e2 < best_energy:
                best_pts, best_energy = c2, e2

    # Phase 3: Basin hopping from best
    if time.time() < deadline - 25:
        remaining = deadline - time.time()
        n_bh = min(30, int(remaining / 1.5))
        if n_bh > 5:
            c, e = _basin_hopping(best_pts, n_iter=n_bh, stepsize=0.1, deadline=deadline)
            if e < best_energy:
                best_pts, best_energy = c, e

    # Phase 4: Final high-precision polish (tighter tolerance, more iterations)
    if time.time() < deadline - 10:
        c, e = _conjugate_gradient(best_pts, max_iter=5000, tol=1e-13, deadline=deadline)
        if e < best_energy:
            best_pts, best_energy = c, e
    if time.time() < deadline - 8:
        c, e = _optimize_lbfgs(best_pts, max_iter=10000, ftol=1e-15, gtol=1e-14, deadline=deadline)
        if e < best_energy:
            best_pts, best_energy = c, e

    best_pts = _project_to_sphere(best_pts)

    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] += 1e-8 * best_pts[i]
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])

    return best_pts
# EVOLVE-BLOCK-END

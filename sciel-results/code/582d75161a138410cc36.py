"""Logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
_START = time.time()
_DEADLINE = _START + 140.0


def _remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_fast(pts):
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    n = pts.shape[0]
    pts = _normalize(pts)
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))

    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)

    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)

    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts

    return energy, grad


def _precise_energy(pts):
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _exp_map(pts, tangent):
    """Exponential map on the unit sphere."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    mask_small = norms < 1e-8
    theta = norms_safe
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    v_normalized = tangent / norms_safe
    new_pts = pts * cos_theta + v_normalized * sin_theta
    if np.any(mask_small):
        new_pts[mask_small] = pts[mask_small] + tangent[mask_small]
    return _normalize(new_pts)


def _log_map(pts, ref):
    """Logarithmic map: tangent vector from ref to pts on the sphere."""
    d = np.sum(pts * ref, axis=1, keepdims=True)
    d = np.clip(d, -1.0 + 1e-15, 1.0 - 1e-15)
    theta = np.arccos(d)
    theta_safe = np.where(np.abs(theta) < 1e-15, 1e-15, theta)
    sin_theta = np.sin(theta_safe)
    tangent = (pts - d * ref) / sin_theta
    tangent = tangent * theta_safe
    return tangent


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _langevin_exp(pts, deadline, rng, max_iter=2000, dt=0.003, gamma=0.5, T_init=0.4, T_min=1e-7):
    """Langevin dynamics with exponential map on the sphere."""
    current = _normalize(pts.copy())
    velocity = np.zeros_like(current)
    best_e = _energy_fast(current)
    best_pts = current.copy()
    n = N

    for it in range(max_iter):
        if time.time() > deadline:
            break

        frac = it / max_iter
        T = T_init * (T_min / T_init) ** (frac ** 0.35)

        e, g = _energy_grad(current)
        F = -g

        # Half-step velocity
        velocity += 0.5 * dt * F

        # Project velocity to tangent space
        dot_v = np.sum(velocity * current, axis=1, keepdims=True)
        velocity = velocity - dot_v * current

        # Full position step via exp map
        new_pts = _exp_map(current, dt * velocity)

        # New force
        new_e, new_g = _energy_grad(new_pts)
        new_F = -new_g

        # Half-step velocity with new force
        velocity += 0.5 * dt * new_F

        # Project to tangent space of new_pts
        dot_v = np.sum(velocity * new_pts, axis=1, keepdims=True)
        velocity = velocity - dot_v * new_pts

        # Damping
        velocity *= (1.0 - gamma * dt)

        # Noise (Langevin kick)
        if T > T_min:
            noise_scale = np.sqrt(2.0 * gamma * T * dt)
            noise = noise_scale * rng.standard_normal((n, 3))
            # Project noise to tangent space
            dot_noise = np.sum(noise * new_pts, axis=1, keepdims=True)
            noise = noise - dot_noise * new_pts
            velocity += noise

        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()

        current = new_pts

    return best_pts, best_e


def _riemannian_lbfgs(pts, deadline, max_iter=5000, ftol=1e-12, gtol=1e-8):
    """
    Riemannian L-BFGS on the sphere using exponential map retraction.
    Uses Wolfe conditions for more robust line search.
    """
    n = pts.shape[0]
    current = _normalize(pts.copy())
    best_pts = current.copy()
    best_e = _precise_energy(current)

    # L-BFGS memory
    m = 12
    S = []  # tangent displacements
    Y = []  # gradient differences

    g_prev = None

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, g = _energy_grad(current)
        g_flat = g.reshape(-1)
        g_norm = np.linalg.norm(g_flat)

        if g_norm < gtol:
            break

        # Compute L-BFGS direction
        if g_prev is None or len(S) == 0:
            d = -g_flat
        else:
            # Two-loop recursion
            q = g_flat.copy()
            alphas = []
            for i in range(len(S) - 1, -1, -1):
                ys = np.dot(Y[i], S[i])
                if abs(ys) < 1e-15:
                    alphas.append(0.0)
                else:
                    alphas.append(np.dot(S[i], q) / ys)
                q = q - alphas[-1] * Y[i]
            alphas.reverse()

            # Scale
            if len(S) > 0:
                gamma_ = np.dot(S[-1], Y[-1]) / max(np.dot(Y[-1], Y[-1]), 1e-15)
            else:
                gamma_ = 1.0
            r = gamma_ * q

            for i in range(len(S)):
                ys = np.dot(Y[i], S[i])
                if abs(ys) < 1e-15:
                    beta = 0.0
                else:
                    beta = np.dot(Y[i], r) / ys
                r = r + (alphas[i] - beta) * S[i]
            d = -r

        # Project direction to tangent space
        d_3d = d.reshape(n, 3)
        dot_d = np.sum(d_3d * current, axis=1, keepdims=True)
        d_3d = d_3d - dot_d * current

        d_norm = np.linalg.norm(d_3d)
        if d_norm < 1e-15:
            break

        # Wolfe line search with adaptive initial step
        c1 = 1e-4
        c2 = 0.9
        slope = np.dot(g_flat, d_3d.reshape(-1))
        
        if slope >= 0:
            # Not a descent direction, use steepest descent
            d_3d = -g.reshape(n, 3)
            dot_d = np.sum(d_3d * current, axis=1, keepdims=True)
            d_3d = d_3d - dot_d * current
            d_norm = np.linalg.norm(d_3d)
            if d_norm < 1e-15:
                break
            slope = np.dot(g_flat, d_3d.reshape(-1))
            alpha_step = 1.0 / max(d_norm, 1e-10)
        else:
            # Heuristic initial step size based on gradient and direction norms
            alpha_step = min(1.0, 0.5 * g_norm / max(d_norm, 1e-10))
            alpha_step = max(alpha_step, 1e-4)

        accepted = False
        alpha_min = 1e-12
        for ls in range(40):
            new_pts = _exp_map(current, alpha_step * d_3d)
            new_e = _energy_fast(new_pts)
            new_e2, new_g2 = _energy_grad(new_pts)
            new_g_flat = new_g2.reshape(-1)
            new_slope = np.dot(new_g_flat, d_3d.reshape(-1))
            
            # Wolfe conditions
            if new_e <= e + c1 * alpha_step * slope and new_slope <= c2 * slope:
                accepted = True
                break
            
            if alpha_step < alpha_min:
                break
            alpha_step *= 0.5

        if not accepted:
            break

        # Update L-BFGS memory
        disp = _log_map(new_pts, current).reshape(-1)
        grad_diff = new_g_flat - g_flat

        S.append(disp)
        Y.append(grad_diff)

        if len(S) > m:
            S.pop(0)
            Y.pop(0)

        g_prev = g_flat
        current = new_pts

        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()

    return best_pts, best_e


def _newton_polish(pts, deadline, max_iter=200):
    """High-precision gradient polish with adaptive step size and tight tolerance."""
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)
    
    alpha = 0.05  # initial step size
    alpha_min = 1e-15
    
    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, g = _energy_grad(best_pts)
        g_flat = g.reshape(-1)
        g_norm = np.linalg.norm(g_flat)

        if g_norm < 1e-16:
            break

        # Steepest descent direction in tangent space
        step_3d = -g
        dot_s = np.sum(step_3d * best_pts, axis=1, keepdims=True)
        step_3d = step_3d - dot_s * best_pts

        step_norm = np.linalg.norm(step_3d)
        if step_norm < 1e-16:
            break

        # Adaptive step: scale by gradient magnitude for better convergence
        alpha_eff = alpha * max(1.0, g_norm / 0.1)
        alpha_eff = min(alpha_eff, 1.0)
        
        step_3d = step_3d / step_norm * alpha_eff

        new_pts = _exp_map(best_pts, step_3d)
        new_e = _precise_energy(new_pts)

        if new_e < best_energy:
            best_pts = new_pts
            best_energy = new_e
            alpha = min(alpha * 1.3, 0.3)  # increase step if improving
        else:
            alpha *= 0.4
            if alpha < alpha_min:
                break

    return best_pts, best_energy


def _optimize_from_start(pts, deadline):
    """Multi-phase optimization: Riemannian L-BFGS then high-precision polish."""
    n = pts.shape[0]
    best_pts = _normalize(pts.copy())
    best_energy = _precise_energy(best_pts)

    # Phase 1: Riemannian L-BFGS (coarse)
    if time.time() < deadline - 20:
        try:
            r1_pts, r1_e = _riemannian_lbfgs(best_pts, deadline - 15, max_iter=1500, gtol=1e-4)
            if r1_e < best_energy:
                best_pts = r1_pts
                best_energy = r1_e
        except Exception:
            pass

    # Phase 2: Riemannian L-BFGS (fine)
    if time.time() < deadline - 10:
        try:
            r2_pts, r2_e = _riemannian_lbfgs(best_pts, deadline - 5, max_iter=2000, gtol=1e-7)
            if r2_e < best_energy:
                best_pts = r2_pts
                best_energy = r2_e
        except Exception:
            pass

    # Phase 3: High-precision gradient polish
    if time.time() < deadline - 0.5:
        try:
            np_pts, np_e = _newton_polish(best_pts, deadline, max_iter=300)
            if np_e < best_energy:
                best_pts = np_pts
                best_energy = np_e
        except Exception:
            pass

    return best_pts, best_energy


def _spectral_init(n, rng, l_max=8):
    """Structured smooth initial configuration."""
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_phi = 2.0 * rng.uniform(0, 1, n) - 1.0
    sin_phi = np.sqrt(1.0 - cos_phi ** 2)

    pts = np.column_stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])

    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            amp = rng.normal(0, 0.01 / (l + 1))
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arctan2(pts[:, 1], pts[:, 0])
            c = pts[:, 2]
            mod = amp * np.cos(m * t + phase) * (c ** abs(m))
            pts[:, 2] += mod * 0.1
            pts[:, 0] += amp * 0.05 * np.sin(m * t + phase) * sin_phi

    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    rng = np.random.default_rng(12345)

    initializations = []

    # 1. Fibonacci spiral variants
    for offset in [0.0, 0.15, 0.3, 0.5, 0.7]:
        initializations.append(_fibonacci_spiral(N, offset))

    # 2. Perturbed Fibonacci
    base = _fibonacci_spiral(N)
    for scale in [0.005, 0.01, 0.02, 0.05]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))

    # 3. Spectral/smooth initializations
    for seed in range(4):
        r = np.random.default_rng(seed * 100 + 7)
        initializations.append(_spectral_init(N, r, l_max=6))

    # 4. Random initializations
    for i in range(8):
        initializations.append(_random_sphere(N, rng))

    # Score and sort
    scored_inits = []
    for init in initializations:
        e_quick = _energy_fast(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    # Phase A: Langevin exploration on top initializations
    for e_quick, init_pts in scored_inits[:5]:
        if time.time() > deadline - 30:
            break
        try:
            lang_pts, lang_e = _langevin_exp(init_pts, deadline - 25, rng,
                                             max_iter=1200, dt=0.002,
                                             gamma=0.5, T_init=0.3, T_min=1e-7)
            polished_pts, polished_e = _optimize_from_start(lang_pts, deadline)
            if polished_e < best_energy:
                best_energy = polished_e
                best_pts = polished_pts
        except Exception:
            pass

    # Phase B: Direct optimization on remaining initializations
    for e_quick, init_pts in scored_inits:
        if time.time() > deadline - 10:
            break
        if best_pts is not None and e_quick > best_energy + 5:
            continue  # Skip if clearly worse than current best

        try:
            optimized_pts, e_precise = _optimize_from_start(init_pts, deadline)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized_pts
        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    # Final high-precision polish with remaining time
    remaining = deadline - time.time()
    if remaining > 2:
        try:
            final_pts, final_e = _newton_polish(best_pts, deadline, max_iter=500)
            if final_e < best_energy:
                best_pts = final_pts
                best_energy = final_e
        except Exception:
            pass

    # Final normalization and validation
    best_pts = _normalize(best_pts)
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _normalize(_fibonacci_spiral(N))

    return best_pts

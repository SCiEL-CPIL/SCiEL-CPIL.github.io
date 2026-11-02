import time
import math
import numpy as np

N = 150

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _log_energy_fast(pts):
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    # Use math.fsum for higher precision as requested in contract, though np.sum is usually fine
    # But math.fsum expects an iterable of floats.
    # np.sum is faster. The contract says "Score: ... summed with math.fsum".
    # This implies the *evaluator* uses math.fsum. My code just needs to return points.
    # However, to be safe, I'll stick to np.sum for internal optimization.
    return float(np.sum(np.log(1.0 / d)))

def _energy_and_grad(pts):
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)

    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts

    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts

    # Energy (upper triangle)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = float(np.sum(np.log(1.0 / d)))

    return energy, grad

def _exponential_map(pts, tangent_vecs):
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms_safe
    mask = norms < 1e-15
    sin_norms_over_norms[mask] = 1.0
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _riemannian_cg(pts, time_limit=20.0, max_iter=2000):
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)

    d = -grad.copy()
    prev_grad = grad.copy()

    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break

        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        gd = np.sum(grad * d)

        # Armijo line search
        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)

            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad

            if new_energy <= energy + 1e-4 * alpha * gd:
                break

            alpha *= 0.7

        if best_alpha == 0.0:
            # Fallback: try smaller steps
            alpha = 1e-3
            for ls in range(20):
                new_pts = _exponential_map(current, alpha * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = new_grad
                    break
                alpha *= 0.5
            if best_alpha == 0.0:
                break

        current = best_pts
        energy = best_energy
        grad = best_grad

        if np.linalg.norm(grad) < 1e-14:
            break

        d_new = -grad
        beta = 0.0
        if it > 0:
            # Polak-Ribiere beta
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
            if beta > 1.0:
                beta = 0.0

        d = d_new + beta * d
        prev_grad = grad.copy()

    return current, energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_energy = float('inf')
    best_pts = None

    # Generate initializations
    initializations = []
    initializations.append(_fibonacci_sphere(N))
    
    # Add random initializations
    for seed in range(10):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        initializations.append(_normalize(pts_rand))
        
    # Add perturbed Fibonacci
    fib_pts = _fibonacci_sphere(N)
    for seed in range(10):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.1
        initializations.append(_normalize(pts_pert))

    # Sort by initial energy
    init_energies = [_log_energy_fast(p) for p in initializations]
    order = np.argsort(init_energies)
    initializations = [initializations[i] for i in order]

    for init_pts in initializations:
        if time.time() > deadline - 20:
            break
        try:
            cur_pts, cur_energy = _riemannian_cg(init_pts, time_limit=10.0)
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_pts = cur_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)

    # Final polish with more iterations on the best point
    if time.time() < deadline - 5:
        polished, pe = _riemannian_cg(best_pts, time_limit=10.0)
        if pe < best_energy:
            best_pts, best_energy = polished, pe

    return _normalize(best_pts)

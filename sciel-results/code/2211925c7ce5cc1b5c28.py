import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _log_energy_fast(pts):
    """Vectorized log energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))

def _energy_and_grad(pts):
    """Vectorized energy + tangent-projected gradient."""
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

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Exponential map on the unit sphere."""
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

def _halton_sphere(n, base=2, base2=3):
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    pts = []
    for i in range(1, n + 1):
        u = halton(i, base)
        v = halton(i, base2)
        z = 2.0 * u - 1.0
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * np.pi * v
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    pts = np.array(pts)
    return _normalize(pts)

def _farthest_point_init(n):
    """Farthest Point Sampling initialization."""
    pts = np.zeros((n, 3))
    # Start with a random point
    rng = np.random.default_rng(42)
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    min_dists = np.full(n, np.inf)
    
    for i in range(1, n):
        # Compute distance to the closest existing point
        diff = pts[i-1][np.newaxis, :] - pts[:i, :]
        d2 = np.sum(diff * diff, axis=1)
        min_dists[i-1] = np.minimum(min_dists[i-1], d2)
        
        # Find the point farthest from all existing points
        # Note: min_dists for points < i are updated, but we only care about the max of the current min_dists for the new point
        # Actually, we need to find the point among the candidates (which are all points on the sphere? No, we are placing n points).
        # FPS places points one by one. The next point is the one on the sphere that maximizes the distance to the set of already placed points.
        # This requires sampling many candidate points on the sphere and picking the one with the max min-distance.
        
        # To be efficient, we can sample a large number of random points and pick the best.
        candidates = rng.standard_normal((10000, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Compute min distance from each candidate to the placed points
        # placed_points = pts[:i]
        # dists = ||c - p||^2 for all c, p
        # This is a matrix multiplication if we expand, but memory might be an issue for 10000 x 150 x 3.
        # 10000 * 150 * 3 * 8 bytes = 36 MB. This is fine.
        
        placed = pts[:i]
        # diff shape: (10000, i, 3)
        diff = candidates[:, np.newaxis, :] - placed[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        min_d2 = np.min(d2, axis=1)
        
        best_idx = np.argmax(min_d2)
        pts[i] = candidates[best_idx]
        
    return _normalize(pts)

def _riemannian_cg(pts, time_limit=15.0, max_iter=1000):
    """Riemannian conjugate gradient with improved line search."""
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

def _newton_polish(pts, time_limit=8.0, max_iter=100):
    """Exact Newton-type local solve."""
    start = time.time()
    x = pts.flatten().copy()

    for _ in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break

        p = _normalize(x.reshape(N, 3))
        _, g = _energy_and_grad(p)
        gf = g.flatten()
        gn = np.linalg.norm(gf)
        if gn < 1e-15:
            break

        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, 1.0)
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)

        row_sum = np.sum(inv_d2, axis=1)
        A = np.diag(row_sum) - inv_d2
        H = np.kron(A, np.eye(3)) - np.kron(np.eye(N), np.eye(3))
        H = 0.5 * (H + H.T)

        try:
            p_step, res = np.linalg.lstsq(H, -gf, rcond=None)
            if not np.all(np.isfinite(p_step)):
                raise np.linalg.LinAlgError("non-finite step")
        except Exception:
            p_step = -gf

        alpha = 1.0
        accepted = False
        for _ in range(40):
            x_try = x + alpha * p_step
            if not np.all(np.isfinite(x_try)):
                alpha *= 0.5
                continue
            p_try = _normalize(x_try.reshape(N, 3))
            e_try = _log_energy_fast(p_try)
            if e_try < _log_energy_fast(_normalize(x.reshape(N, 3))) - 1e-16 * (1.0 + abs(e_try)) or alpha < 1e-12:
                x = x_try
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            break

    return _normalize(x.reshape(N, 3)), _log_energy_fast(_normalize(x.reshape(N, 3)))

def _refine_pipeline(pts, deadline):
    """Run the full refinement pipeline on one start, respecting deadline."""
    current = pts.copy()
    energy = _log_energy_fast(current)

    if time.time() < deadline - 20:
        current, energy = _riemannian_cg(current, time_limit=10.0)

    if time.time() < deadline - 10:
        current, energy = _newton_polish(current, time_limit=8.0)

    return current, energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_energy = float('inf')
    best_pts = None

    initializations = []
    initializations.append(_fibonacci_sphere(N))
    initializations.append(_halton_sphere(N))
    initializations.append(_farthest_point_init(N))
    
    # Random
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        initializations.append(_normalize(pts_rand))
        
    # Perturbed Fibonacci
    fib_pts = _fibonacci_sphere(N)
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.05
        initializations.append(_normalize(pts_pert))

    # Sort by initial energy
    init_energies = [_log_energy_fast(p) for p in initializations]
    order = np.argsort(init_energies)
    initializations = [initializations[i] for i in order]

    for init_pts in initializations:
        if time.time() > deadline - 30:
            break
        try:
            cur_pts, cur_energy = _refine_pipeline(init_pts, deadline)
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_pts = cur_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)

    # Final polish
    if time.time() < deadline - 5:
        polished, pe = _newton_polish(best_pts, time_limit=5.0)
        if pe < best_energy:
            best_pts, best_energy = polished, pe

    return _normalize(best_pts)

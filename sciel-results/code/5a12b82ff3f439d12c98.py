import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_S = 142.0  # safety margin before 150s limit


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)

    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts

    # Project gradient to tangent space: g - (g . p) p
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts

    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))

    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _log_energy_sphere(x_flat):
    """Objective and gradient for L-BFGS-B on flattened sphere coordinates.
    Uses local chart: each point represented by 2 coordinates (stereographic-like).
    We use the tangent-space parameterization: fix a reference frame per point.
    """
    # For L-BFGS-B, we need to work in R^(3N) with projection
    # Instead, use a simpler approach: minimize in R^(3N) with penalty
    pts = x_flat.reshape(N, 3)
    pts = _normalize(pts)
    energy, grad = _energy_and_grad(pts)
    # Gradient in ambient space (not projected) for L-BFGS-B
    # We need the ambient gradient, so recompute without projection
    n = N
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)

    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    ambient_grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts

    return energy, ambient_grad.flatten()


def _lbfgs_sphere(pts, time_limit=15.0):
    """L-BFGS-B optimization on the sphere using ambient coordinates with
    normalization after each step. This is a practical approach that works
    well for moderate N."""
    start = time.time()
    x0 = pts.flatten().copy()

    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return float('inf'), np.zeros_like(x)
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        energy, grad = _energy_and_grad(pts)
        # For L-BFGS-B in ambient space, use the unprojected gradient
        # but since we normalize, the effective gradient is the projected one
        # mapped back. Use the tangent gradient as the gradient signal.
        return energy, grad.flatten()

    result = minimize(
        obj_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 2000,
            'ftol': 1e-14,
            'gtol': 1e-10,
            'maxfun': 20000,
        }
    )

    if result.success or result.fun < float('inf'):
        best_pts = _normalize(result.x.reshape(N, 3))
        best_energy = _log_energy_fast(best_pts)
    else:
        best_pts = pts.copy()
        best_energy = _log_energy_fast(pts)

    return best_pts, best_energy


def _farthest_point_sampling(n, seed=42):
    """Generate well-distributed points using farthest point sampling."""
    rng = np.random.default_rng(seed)
    # Start with a random point
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])

    for _ in range(n - 1):
        # Generate candidate points
        candidates = rng.standard_normal((500, 3))
        candidates = _normalize(candidates)

        # Find the candidate that maximizes minimum distance to existing points
        best_candidate = None
        best_min_dist = -1
        for c in candidates:
            dists = np.linalg.norm(pts_arr - c, axis=1) if len(pts) > 1 else np.array([0])
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = c

        pts.append(best_candidate)

    # Vectorized version for speed
    # Actually, let's use a batch approach
    pts = np.array(pts)
    return _normalize(pts)


def _fps_fast(n, seed=42, batch_size=1000):
    """Fast farthest point sampling using vectorized operations."""
    rng = np.random.default_rng(seed)
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    pts_arr = np.array(pts)

    for _ in range(n - 1):
        candidates = rng.standard_normal((batch_size, 3))
        candidates = _normalize(candidates)

        # Compute min distance from each candidate to existing points
        # Use chunked computation to avoid memory issues
        min_dists = np.full(batch_size, np.inf)
        chunk = 50
        for i in range(0, len(pts_arr), chunk):
            chunk_pts = pts_arr[i:i+chunk]
            d = np.linalg.norm(candidates[:, np.newaxis, :] - chunk_pts[np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, np.min(d, axis=1))

        best_idx = np.argmax(min_dists)
        pts_arr = np.vstack([pts_arr, candidates[best_idx:best_idx+1]])

    return _normalize(pts_arr)


def _basin_hopping(pts, time_limit=20.0, max_restarts=10):
    """Basin hopping with L-BFGS local optimization and adaptive perturbation."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)

    # Initial optimization
    time_left_init = time_limit - (time.time() - start) - 2.0
    if time_left_init > 5.0:
        current, energy = _lbfgs_sphere(current, time_limit=min(10.0, time_left_init))

    best_pts = current.copy()
    best_energy = energy

    # Basin Hopping Loop with reheat
    temp = 0.5
    decay = 0.85
    reheat_every = 5
    reheat_temp = 0.3

    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2.0:
            break

        # Reheat
        if rh > 0 and rh % reheat_every == 0:
            temp = reheat_temp

        # Adaptive perturbation
        perturb_scale = max(0.03, min(0.25, 0.08 * temp + 0.03))

        # Generate random tangent vectors
        rng = np.random.default_rng(rh * 7919 + 1)
        perturbations = rng.standard_normal((N, 3))
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        pert_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
        pert_norms = np.maximum(pert_norms, 1e-15)
        perturbations = perturbations / pert_norms * perturb_scale

        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)

        # Optimize the perturbed point
        time_left = time_limit - (time.time() - start) - 1.0
        if time_left < 4.0:
            break

        new_opt_pts, new_energy = _lbfgs_sphere(new_pts, time_limit=min(8.0, time_left))
        new_energy = _log_energy_fast(new_opt_pts)

        # Metropolis Criterion
        delta_e = new_energy - energy
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(temp, 1e-10)):
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()

        temp *= decay

    return best_pts, best_energy


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
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
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        pts.append([x, y, z])

    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _spherical_design_init(n, seed=42):
    """Generate points using structured approach with random rotations."""
    rng = np.random.default_rng(seed)
    pts = _fibonacci_sphere(n)
    for _ in range(3):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        pts = pts @ R.T
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_S

    # Initializations - diverse set
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_spherical_design_init(N, seed=42))
    inits.append(_spherical_design_init(N, seed=123))

    # FPS-based inits
    inits.append(_fps_fast(N, seed=7))
    inits.append(_fps_fast(N, seed=13))

    # Random inits
    for s in range(3):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)

    # Perturbed structured inits
    for s in range(3):
        rng = np.random.default_rng(s + 100)
        base = _fibonacci_sphere(N) if s % 2 == 0 else _halton_sphere(N)
        pts = base + rng.standard_normal((N, 3)) * 0.12
        pts = _normalize(pts)
        inits.append(pts)

    best_pts = None
    best_energy = float('inf')

    # Process each initialization with adaptive time budget
    total_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        remaining = deadline - time.time() - 10.0  # reserve 10s for final polish
        if remaining < 8.0:
            break

        # Adaptive time per restart: more time for early inits, less for later
        # Front-load time to structured inits
        if idx < 5:
            time_per_restart = min(18.0, remaining / (total_inits - idx) * 1.2)
        else:
            time_per_restart = min(12.0, remaining / (total_inits - idx) * 1.0)

        time_per_restart = max(6.0, time_per_restart)

        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            opt_energy = _log_energy_fast(opt_pts)

            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()

        except Exception:
            pass

    # If no valid points found, fallback to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)

    # Final polish: multiple L-BFGS passes
    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        try:
            for polish_round in range(3):
                if time.time() > deadline - 3.0:
                    break
                polish_time = min(10.0, deadline - time.time() - 3.0)
                if polish_time < 2.0:
                    break
                final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=polish_time)
                final_energy = _log_energy_fast(final_pts)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pts = final_pts
        except Exception:
            pass

    return _normalize(best_pts)

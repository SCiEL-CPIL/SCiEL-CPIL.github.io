"""Rewritten program: minimise logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
# Deadline guard: evaluator kills after 150s; keep a safety margin.
DEADLINE = 140.0  # seconds


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_fast(pts):
    """Vectorized log energy (less precise but much faster)."""
    # pts: (N, 3)
    # Compute pairwise squared distances
    sq = np.sum(pts * pts, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)  # avoid log(0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    return np.sum(np.log(1.0 / d))


def _log_energy_grad(pts):
    """Compute log energy and gradient using vectorized ops."""
    # pts: (N, 3)
    sq = np.sum(pts * pts, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = np.sum(np.log(1.0 / d))

    # Gradient: d/dx_i log(1/|x_i - x_j|) = -(x_i - x_j) / |x_i - x_j|^2
    # Sum over j != i
    # For each pair (i,j), contribution to grad[i] is -(x_i - x_j)/d^2, to grad[j] is (x_i - x_j)/d^2
    grad = np.zeros_like(pts)
    # Vectorized: for all pairs
    diff = pts[:, None, :] - pts[None, :, :]  # (N, N, 3)
    inv_d2 = 1.0 / dist_sq  # (N, N)
    # grad[i] = sum_j (x_j - x_i) / |x_i - x_j|^2 = -sum_j (x_i - x_j)/d^2
    # Actually: d/dx_i log(1/d) = -d/dx_i log(d) = -d/dx_i (1/2 log(d^2)) = -(x_i - x_j)/d^2
    # So grad[i] = sum_{j!=i} -(x_i - x_j)/d_ij^2
    # = sum_{j!=i} (x_j - x_i)/d_ij^2
    # diff[i,j] = x_i - x_j, so we want -diff * inv_d2 summed over j
    grad = -np.sum(diff * inv_d2[:, :, None], axis=1)
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_start():
    """Golden-angle Fibonacci spiral on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(seed):
    """Random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _optimize_from_start(pts0, max_time):
    """Run L-BFGS-B optimization from a starting configuration."""
    start_time = time.time()
    
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        e, g = _log_energy_grad(pts)
        return e, g.flatten()
    
    # Use L-BFGS-B with numerical gradient check off
    res = minimize(
        objective_and_grad,
        pts0.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 10000}
    )
    
    elapsed = time.time() - start_time
    if res.success or res.status == 0:
        best_pts = _project_to_sphere(res.x.reshape(N, 3))
    else:
        best_pts = _project_to_sphere(res.x.reshape(N, 3))
    
    return best_pts, res.fun, elapsed


def construct():
    global DEADLINE
    start_time = time.time()
    
    # Start with Fibonacci spiral
    pts_fib = _fibonacci_start()
    current_energy = _log_energy_fast(pts_fib)
    best_pts = pts_fib.copy()
    best_energy = current_energy
    
    # Optimize from Fibonacci start
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        pts_opt, e_opt, elapsed = _optimize_from_start(pts_fib, remaining)
        if time.time() - start_time < DEADLINE - 5:
            # Verify with precise energy
            e_precise = _log_energy(pts_opt)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = pts_opt
    
    # Try multiple random restarts
    rng = np.random.RandomState(42)
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        pts_rand = _random_start(i + 1)
        # Quick local optimization
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 10:
            break
        
        def obj_grad(x):
            pts = x.reshape(N, 3)
            e, g = _log_energy_grad(pts)
            return e, g.flatten()
        
        try:
            res = minimize(
                obj_grad,
                pts_rand.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-9, 'maxfun': 5000}
            )
            pts_new = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy_fast(pts_new)
            if e_new < best_energy:
                # Precise check
                e_precise = _log_energy(pts_new)
                if e_precise < best_energy:
                    best_energy = e_precise
                    best_pts = pts_new
        except Exception:
            continue
    
    # Final verification
    if time.time() - start_time < DEADLINE - 2:
        final_e = _log_energy(best_pts)
        if final_e < best_energy:
            best_energy = final_e
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or np.any(~np.isfinite(best_pts)):
        return _fibonacci_start()
    
    return _project_to_sphere(best_pts)

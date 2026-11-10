"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Uses dot-product trick for speed: |x_i - x_j|^2 = 2 - 2*x_i·x_j for unit vectors.
    """
    pts = pts.reshape(-1, 3)
    n = pts.shape[0]
    
    # Compute pairwise dot products
    dot = pts @ pts.T  # (n, n)
    
    # Squared distances: |x_i - x_j|^2 = 2 - 2*dot_ij for unit vectors
    diff_sq = 2.0 - 2.0 * dot
    # Clamp to avoid negative values from floating point
    diff_sq = np.maximum(diff_sq, 1e-15)
    
    # Mask diagonal
    np.fill_diagonal(diff_sq, np.inf)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(diff_sq[iu])
    energy = -np.sum(np.log(d))
    
    # Gradient: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist_sq = np.zeros_like(diff_sq)
    np.divide(1.0, diff_sq, out=inv_dist_sq, where=diff_sq < np.inf)
    np.fill_diagonal(inv_dist_sq, 0)
    
    # grad_i = sum_j (x_j - x_i) * inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad = inv_dist_sq @ pts - pts * row_sums[:, np.newaxis]
    
    return energy, grad


def _project_and_normalize(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n=N, golden=None):
    """Generate Fibonacci spiral points on the unit sphere."""
    if golden is None:
        golden = np.pi * (1.0 + np.sqrt(5.0))
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = golden * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n=N, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturbed_fibonacci(n=N, seed=42, scale=0.1):
    """Generate perturbed Fibonacci spiral points."""
    rng = np.random.RandomState(seed)
    pts = _fibonacci_spiral(n).copy()
    for i in range(n):
        noise = rng.randn(3) * scale
        pts[i] += noise
        pts[i] /= np.linalg.norm(pts[i])
    return pts


def _ring_initialization(n=N, n_rings=10):
    """Generate points on concentric rings on the sphere."""
    pts = np.zeros((n, 3))
    pts_per_ring = n // n_rings
    remainder = n % n_rings
    idx = 0
    for ring in range(n_rings):
        count = pts_per_ring + (1 if ring < remainder else 0)
        theta = (ring + 0.5) * np.pi / n_rings
        r = np.sin(theta)
        z = np.cos(theta)
        for k in range(count):
            phi = 2 * np.pi * k / count
            pts[idx] = [r * np.cos(phi), r * np.sin(phi), z]
            idx += 1
    return pts


def _local_optimize(pts, deadline, maxiter=5000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B local optimization from given starting points."""
    if time.time() > deadline:
        return pts, _energy_and_grad(pts)[0]
    
    x0 = pts.flatten()
    
    def objective(x):
        p = _project_and_normalize(x.reshape(-1, 3))
        return _energy_and_grad(p)[0]
    
    def grad(x):
        p = _project_and_normalize(x.reshape(-1, 3))
        e, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    try:
        res = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 2}
        )
        cand = _project_and_normalize(res.x.reshape(-1, 3))
        cand_e = objective(cand.flatten())
        return cand, cand_e
    except Exception:
        return pts, _energy_and_grad(pts)[0]


def _simulated_annealing(pts, deadline, max_iter=3000, T0=0.05, T1=1e-8):
    """Simulated annealing with geometric moves: swaps, rotations, and small perturbations."""
    best_pts = pts.copy()
    best_e = _energy_and_grad(best_pts)[0]
    cur_e = best_e
    cur_pts = pts.copy()
    n = N
    
    rng = np.random.RandomState(12345)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        T = T0 * (T1 / T0) ** (it / max_iter)
        
        # Choose move type
        move_type = rng.random()
        
        new_pts = cur_pts.copy()
        
        if move_type < 0.3:
            # Swap two points
            i, j = rng.choice(n, size=2, replace=False)
            new_pts[i], new_pts[j] = cur_pts[j].copy(), cur_pts[i].copy()
        elif move_type < 0.6:
            # Rotate a subset of points around a random axis
            n_rotate = max(2, n // 15)
            idx = rng.choice(n, size=n_rotate, replace=False)
            axis = rng.randn(3)
            axis /= np.linalg.norm(axis)
            angle = rng.randn() * 0.1
            # Rodrigues rotation
            for i in idx:
                v = cur_pts[i]
                rotated = (v * np.cos(angle) + 
                          np.cross(axis, v) * np.sin(angle) + 
                          axis * np.dot(axis, v) * (1 - np.cos(angle)))
                new_pts[i] = rotated / np.linalg.norm(rotated)
        else:
            # Perturb a few points
            n_perturb = max(1, n // 25)
            idx = rng.choice(n, size=n_perturb, replace=False)
            noise_scale = 0.03 * (T / T0)
            for i in idx:
                noise = rng.randn(3) * noise_scale
                new_pts[i] = cur_pts[i] + noise
                new_pts[i] /= np.linalg.norm(new_pts[i])
        
        new_e = _energy_and_grad(new_pts)[0]
        delta_e = new_e - cur_e
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            cur_pts = new_pts
            cur_e = new_e
            if cur_e < best_e:
                best_pts = cur_pts.copy()
                best_e = cur_e
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = np.inf
    
    # Phase 1: Try multiple diverse starting points with local optimization
    starts = [
        _fibonacci_spiral(N),
        _fibonacci_spiral(N, golden=np.pi * (1.0 + np.sqrt(5.0)) * 1.001),
        _random_points(N, seed=42),
        _random_points(N, seed=123),
        _random_points(N, seed=456),
        _perturbed_fibonacci(N, seed=7, scale=0.03),
        _perturbed_fibonacci(N, seed=13, scale=0.08),
        _perturbed_fibonacci(N, seed=99, scale=0.15),
        _ring_initialization(N, n_rings=8),
        _ring_initialization(N, n_rings=12),
    ]
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 40:
            break
        cand, cand_e = _local_optimize(start, deadline, maxiter=3000, ftol=1e-14, gtol=1e-10)
        if cand_e < best_energy:
            best_pts = cand
            best_energy = cand_e
    
    # Phase 2: Simulated annealing to escape local minima
    if time.time() < deadline - 30:
        try:
            sa_pts, sa_e = _simulated_annealing(
                best_pts, deadline,
                max_iter=2000, T0=0.03, T1=1e-8
            )
            if sa_e < best_energy:
                best_pts = sa_pts
                best_energy = sa_e
        except Exception:
            pass
    
    # Phase 3: High-precision polish with tighter tolerances
    if time.time() < deadline - 10:
        try:
            cand, cand_e = _local_optimize(best_pts, deadline, maxiter=10000, ftol=1e-16, gtol=1e-12)
            if cand_e < best_energy:
                best_pts = cand
                best_energy = cand_e
        except Exception:
            pass
    
    # Phase 4: Final ultra-tight polish if time permits
    if time.time() < deadline - 3:
        try:
            cand, cand_e = _local_optimize(best_pts, deadline, maxiter=5000, ftol=1e-16, gtol=1e-14)
            if cand_e < best_energy:
                best_pts = cand
                best_energy = cand_e
        except Exception:
            pass
    
    # Final check and return
    result = _project_and_normalize(best_pts)
    
    # Validate: check for near-duplicate points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                result[i] += np.random.normal(0, 1e-8, 3)
                result[i] /= np.linalg.norm(result[i])
    
    # Ensure non-finite values are handled
    if not np.all(np.isfinite(result)):
        result = _fibonacci_spiral(N)
    
    return result
# EVOLVE-BLOCK-END

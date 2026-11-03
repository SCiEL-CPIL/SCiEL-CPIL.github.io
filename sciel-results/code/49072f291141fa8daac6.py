"""Optimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150
DEADLINE = 145.0  # seconds, with 5s safety margin before 150s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    where d_ij is chordal distance.
    
    Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(d_ij)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    # = sum_{j!=i} (x_j - x_i) / dist_sq[i,j]
    
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    weighted_sum = inv_dist_sq @ pts  # (N, 3)
    
    # grad[i] = sum_j (x_j - x_i) / dist_sq[i,j] = weighted_sum[i] - x_i * row_sums[i]
    grad = weighted_sum - pts * row_sums[:, np.newaxis]  # (N, 3)
    
    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    g_dot_n = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_n * pts


def _fibonacci_sphere(n, offset=0.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_based(n):
    """Generate points using icosahedron subdivision for better initial distribution."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    if n == 12:
        return ico
    
    fib = _fibonacci_sphere(n)
    result = fib.copy()
    for i in range(12):
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _so3_rotate(pts, rng):
    """Apply random SO(3) rotation to points."""
    u = rng.standard_normal(3)
    v = rng.standard_normal(3)
    w = rng.standard_normal(3)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    w = w / np.linalg.norm(w)
    R = np.column_stack([u, v, w])
    return pts @ R.T


def _lbfgs_optimize(pts0, max_time, ftol=1e-12, gtol=1e-10):
    """L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    n = len(pts)
    
    def objective_flat(x):
        if time.time() - start_time > max_time:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e
    
    def grad_flat(x):
        if time.time() - start_time > max_time:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'maxfun': 10000,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 50
        }
    )
    
    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _adam_optimize(pts0, max_time):
    """Adam optimizer on the sphere with adaptive learning rate."""
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    
    lr_initial = 0.02
    lr_min = 0.0005
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr_decay = 0.997
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    lr = lr_initial
    
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)
    
    no_improve_count = 0
    patience = 300
    
    while time.time() - start_time < max_time:
        t += 1
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        m_tangent = m_hat - np.sum(m_hat * pts, axis=1, keepdims=True) * pts
        
        pts = pts - lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)
        
        lr = max(lr_min, lr * lr_decay)
        
        if t % 50 == 0:
            g_tan = _tangent_project(pts, grad)
            if np.linalg.norm(g_tan) < 1e-10:
                break
    
    return best_pts, best_energy


def _simulated_annealing(pts, max_time, rng):
    """Simulated annealing with batch moves to escape local minima."""
    start_time = time.time()
    current = pts.copy()
    current_e, _ = _energy_and_grad(current)
    best = current.copy()
    best_e = current_e
    
    T = 0.05
    T_min = 1e-8
    cooling = 0.995
    
    while time.time() - start_time < max_time and T > T_min:
        # Move a small batch of points
        batch_size = rng.integers(1, 6)
        indices = rng.choice(N, size=batch_size, replace=False)
        
        new_pts = current.copy()
        for idx in indices:
            # Random direction in tangent space
            rand_dir = rng.standard_normal(3)
            rand_dir -= np.dot(rand_dir, current[idx]) * current[idx]
            norm = np.linalg.norm(rand_dir)
            if norm > 1e-15:
                rand_dir /= norm
            
            # Move along random direction
            step_size = T * 0.3
            new_pt = current[idx].copy()
            new_pt += step_size * rand_dir
            new_pt = new_pt / np.linalg.norm(new_pt)
            new_pts[idx] = new_pt
        
        new_e = _energy_only(new_pts)
        delta_e = new_e - current_e
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-12)):
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
        
        T *= cooling
    
    return best, best_e


def _polish(pts0, max_time):
    """Final high-precision polish with very tight tolerances."""
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    n = len(pts)
    
    def objective_flat(x):
        if time.time() - start_time > max_time:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e
    
    def grad_flat(x):
        if time.time() - start_time > max_time:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 10000,
            'maxfun': 20000,
            'ftol': 1e-15,
            'gtol': 1e-13,
            'maxls': 100
        }
    )
    
    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron-based
    starts.append(_icosahedron_based(N))
    
    # 3. Perturbed Fibonacci with different offsets
    for offset in [0.0, 0.3, 0.7]:
        fib = _fibonacci_sphere(N, offset)
        noise = 0.03 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 4. Random points
    for _ in range(3):
        starts.append(_random_sphere(N, rng))
    
    # 5. Rotated Fibonacci
    for _ in range(2):
        fib = _fibonacci_sphere(N)
        starts.append(_so3_rotate(fib, rng))
    
    # 6. Clustered starts
    for _ in range(2):
        centers = _random_sphere(10, rng)
        pts = []
        for c in centers:
            local = c + 0.1 * rng.standard_normal((15, 3))
            pts.append(_project_to_sphere(local))
        starts.append(np.vstack(pts))
    
    # Optimize each start
    best_pts = None
    best_energy = float('inf')
    
    n_starts = len(starts)
    # Allocate time: ~60% for initial optimizations, ~25% for SA, ~15% for final polish
    elapsed = time.time() - start_time
    opt_budget = (DEADLINE - elapsed) * 0.6
    sa_budget = (DEADLINE - elapsed) * 0.25
    polish_budget = (DEADLINE - elapsed) * 0.15
    
    time_per_opt = opt_budget / n_starts
    time_per_opt = max(2.0, min(12.0, time_per_opt))
    
    for i, s in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        remaining = DEADLINE - elapsed
        if remaining < 20:
            break
        
        opt_time = min(time_per_opt, remaining - 15)
        if opt_time < 2:
            break
        
        try:
            # Try L-BFGS-B first
            pts_opt, e_opt = _lbfgs_optimize(s, opt_time)
            
            # If L-BFGS-B didn't converge well, try Adam
            if e_opt > -2300:  # Threshold indicating poor convergence
                adam_time = min(opt_time * 0.5, remaining - 18)
                if adam_time > 1:
                    pts_adam, e_adam = _adam_optimize(s, adam_time)
                    if e_adam < e_opt:
                        pts_opt, e_opt = pts_adam, e_adam
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # Simulated annealing on best result
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 25:
            remaining = DEADLINE - elapsed - 20
            if remaining > 5:
                try:
                    pts_sa, e_sa = _simulated_annealing(best_pts, remaining, rng)
                    if e_sa < best_energy:
                        best_energy = e_sa
                        best_pts = pts_sa.copy()
                except Exception:
                    pass
    
    # Final high-precision polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            remaining = DEADLINE - elapsed - 5
            if remaining > 3:
                try:
                    pts_final, e_final = _polish(best_pts, remaining)
                    if e_final < best_energy:
                        best_energy = e_final
                        best_pts = pts_final.copy()
                except Exception:
                    pass
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    assert np.all(np.linalg.norm(best_pts, axis=1) > 1e-15)
    
    # Check no two points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them slightly
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts

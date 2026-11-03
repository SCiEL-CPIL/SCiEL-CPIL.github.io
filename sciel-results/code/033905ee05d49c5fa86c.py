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
    
    Gradient w.r.t. x_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2 * (-1) ... 
    Let's derive: E = -sum_{i<j} log|d_ij|, d_ij = |x_i - x_j|
    dE/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j!=i} (x_i - x_j)/d_ij^2
    But we need to project onto the tangent space of the sphere.
    """
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # Note: dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2 ... wait let me re-derive
    # E = sum_{i<j} -log|d_ij|
    # dE/dx_i = sum_{j: j>i} -d/dx_i log|d_ij| + sum_{j: j<i} -d/dx_i log|d_ji|
    # d/dx_i log|d_ij| = (x_i - x_j) / |d_ij|^2
    # So dE/dx_i = -sum_{j!=i} (x_i - x_j) / |d_ij|^2
    # This is the negative of the Coulomb-like force, which makes sense.
    
    # grad[i] = -sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    # = -sum_{j!=i} x_i/dist_sq[i,j] + sum_{j!=i} x_j/dist_sq[i,j]
    # = -x_i * sum_{j!=i} 1/dist_sq[i,j] + sum_{j!=i} x_j / dist_sq[i,j]
    
    inv_dist_sq = 1.0 / dist_sq  # (N, N), diagonal is 1.0 but we'll zero it out
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -x_i * sum_j inv_dist_sq[i,j] + sum_j x_j * inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    # sum_j x_j * inv_dist_sq[i,j] = inv_dist_sq @ pts
    weighted_sum = inv_dist_sq @ pts  # (N, 3)
    
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum  # (N, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _optimize_single(pts0, max_time):
    """Optimize a single starting configuration using L-BFGS-B on the sphere.
    
    We use the parameterization where we optimize in R^{N*3} and project back,
    but a better approach is to use the fact that the gradient on the sphere is
    the tangential component of the Euclidean gradient.
    """
    start_time = time.time()
    
    # We'll optimize in R^3 for each point, then project. But L-BFGS-B doesn't 
    # handle constraints well. Instead, we use a penalty or we optimize on the 
    # sphere using a different parameterization.
    
    # Better approach: use the fact that for points on the sphere, the 
    # constrained gradient is the projection of the unconstrained gradient 
    # onto the tangent space. We can use this with a projected gradient method
    # or we can use scipy's minimize with bounds... 
    # 
    # Actually, the cleanest approach: optimize in R^{3N} with the constraint 
    # that each point has norm 1. We can use the trick of optimizing over 
    # R^{3N} and after each iteration, projecting back to the sphere.
    # This is not exactly correct for L-BFGS-B but works in practice.
    #
    # Alternative: use spherical coordinates (theta, phi) for each point.
    # This removes the constraint but introduces singularities at poles.
    
    # Let's use the approach: optimize in R^{3N}, project after each step.
    # Use L-BFGS-B with the projected gradient.
    
    x0 = pts0.flatten()
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        # Project gradient onto tangent space: g_tan = g - (g·n)n where n = pts (unit normal)
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return g_tan.flatten()
    
    best_x = x0.copy()
    best_e = objective(x0)
    
    while time.time() - start_time < max_time:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        
        x_new = result.x
        e_new = objective(x_new)
        
        if e_new < best_e:
            best_e = e_new
            best_x = x_new.copy()
        
        # Check if we've converged
        if result.success and result.status in [0, 1]:
            break
        
        # Perturb to escape local minimum
        if time.time() - start_time < max_time * 0.8:
            noise = 0.1 * np.random.randn(len(x0))
            x0 = best_x + noise
        else:
            break
    
    return _project_to_sphere(best_x.reshape(N, 3)), best_e


def _simulated_annealing(pts, max_time, rng):
    """Simulated annealing step to escape local minima."""
    start_time = time.time()
    current = pts.copy()
    current_e, _ = _energy_and_grad(current)
    best = current.copy()
    best_e = current_e
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.99
    
    while time.time() - start_time < max_time and T > T_min:
        # Pick a random point and perturb it
        idx = rng.integers(N)
        angle = rng.uniform(0, 2 * np.pi)
        # Random direction in tangent space
        rand_dir = rng.standard_normal(3)
        rand_dir -= np.dot(rand_dir, current[idx]) * current[idx]
        norm = np.linalg.norm(rand_dir)
        if norm > 1e-15:
            rand_dir /= norm
        
        # Move along random direction
        step_size = T * 0.5
        new_pt = current[idx].copy()
        new_pt += step_size * rand_dir
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        new_pts = current.copy()
        new_pts[idx] = new_pt
        
        new_e, _ = _energy_and_grad(new_pts)
        delta_e = new_e - current_e
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
        
        T *= cooling
    
    return best, best_e


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Perturbed Fibonacci
    for _ in range(3):
        fib = _fibonacci_sphere(N)
        noise = 0.05 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 3. Random points
    for _ in range(5):
        starts.append(_random_sphere(N, rng))
    
    # 4. Clustered starts (to explore different basins)
    for _ in range(3):
        centers = _random_sphere(10, rng)
        pts = []
        for c in centers:
            local = c + 0.1 * rng.standard_normal((15, 3))
            pts.append(_project_to_sphere(local))
        starts.append(np.vstack(pts))
    
    # Optimize each start
    best_pts = None
    best_energy = float('inf')
    
    time_per_opt = (DEADLINE - (time.time() - start_time)) / len(starts)
    time_per_opt = max(5.0, min(20.0, time_per_opt))
    
    for i, s in enumerate(starts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 10:
            break
        
        opt_time = min(time_per_opt, remaining - 5)
        if opt_time < 5:
            break
        
        try:
            pts_opt, e_opt = _optimize_single(s, opt_time)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
                
                # Try simulated annealing on the best result if time permits
                sa_time = min(15.0, (DEADLINE - (time.time() - start_time)) - 10)
                if sa_time > 5 and i < len(starts) - 2:
                    pts_sa, e_sa = _simulated_annealing(pts_opt, sa_time, rng)
                    if e_sa < best_energy:
                        best_energy = e_sa
                        best_pts = pts_sa.copy()
                        
                        # Final polish with L-BFGS-B
                        polish_time = min(15.0, (DEADLINE - (time.time() - start_time)) - 5)
                        if polish_time > 5:
                            pts_polish, e_polish = _optimize_single(pts_sa, polish_time)
                            if e_polish < best_energy:
                                best_energy = e_polish
                                best_pts = pts_polish.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining > 5:
            try:
                pts_final, e_final = _optimize_single(best_pts, remaining)
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

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


def _lbfgs_optimize(pts0, max_time, tol=1e-10):
    """Optimize using L-BFGS-B with projection back to sphere."""
    start_time = time.time()
    
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_e = _energy_and_grad(pts)[0]
    
    def objective_and_grad(flat_pts):
        if time.time() - start_time > max_time:
            raise Exception("timeout")
        p = flat_pts.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        # Project gradient to tangent space for L-BFGS-B
        g_tan = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tan.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': tol, 'gtol': tol, 'maxfun': 5000}
        )
        final_pts = _project_to_sphere(result.x.reshape(N, 3))
        final_e, _ = _energy_and_grad(final_pts)
        if final_e < best_e:
            best_e = final_e
            best_pts = final_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_e


def _kmc_perturb(pts, max_time, rng, T_start=0.05, T_end=1e-6):
    """Kinetic Monte Carlo perturbation to escape local minima."""
    start_time = time.time()
    current = pts.copy()
    current_e, _ = _energy_and_grad(current)
    best = current.copy()
    best_e = current_e
    
    n_steps = 5000
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > max_time:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Pick a random point to move
        idx = rng.integers(N)
        
        # Random direction in tangent space
        rand_dir = rng.standard_normal(3)
        rand_dir -= np.dot(rand_dir, current[idx]) * current[idx]
        norm = np.linalg.norm(rand_dir)
        if norm < 1e-15:
            continue
        rand_dir /= norm
        
        # Step size proportional to sqrt(T)
        step_size = np.sqrt(T) * 0.3
        new_pt = current[idx] + step_size * rand_dir
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        new_pts = current.copy()
        new_pts[idx] = new_pt
        new_e, _ = _energy_and_grad(new_pts)
        
        delta_e = new_e - current_e
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
    
    return best, best_e


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
    
    # 1. Fibonacci spiral (best single start)
    starts.append(_fibonacci_sphere(N))
    
    # 2. Slightly perturbed Fibonacci variants
    for scale in [0.01, 0.03, 0.08]:
        fib = _fibonacci_sphere(N)
        noise = scale * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 3. A few random starts for diversity
    for _ in range(3):
        starts.append(_random_sphere(N, rng))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: L-BFGS optimization from multiple starts
    time_weights = [0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]
    total_available = DEADLINE - (time.time() - start_time) - 20  # Reserve for KMC + final polish
    
    for i, s in enumerate(starts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        remaining = DEADLINE - (time.time() - start_time) - 20
        if remaining < 2:
            break
        
        w = time_weights[i] if i < len(time_weights) else 0.05
        opt_time = max(2.0, min(w * total_available, remaining - 1))
        
        try:
            pts_opt, e_opt = _lbfgs_optimize(s, opt_time, tol=1e-8)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _energy_and_grad(best_pts)[0]
    
    # Phase 2: KMC perturbation to escape local minima
    if time.time() - start_time < DEADLINE - 10:
        kmc_time = min(15.0, DEADLINE - (time.time() - start_time) - 10)
        if kmc_time > 3:
            try:
                pts_kmc, e_kmc = _kmc_perturb(best_pts, kmc_time, rng)
                if e_kmc < best_energy:
                    best_energy = e_kmc
                    best_pts = pts_kmc.copy()
            except Exception:
                pass
            
            # Re-optimize after KMC
            if time.time() - start_time < DEADLINE - 5:
                polish_time = min(8.0, DEADLINE - (time.time() - start_time) - 5)
                if polish_time > 2:
                    try:
                        pts_polished, e_polished = _lbfgs_optimize(best_pts, polish_time, tol=1e-10)
                        if e_polished < best_energy:
                            best_energy = e_polished
                            best_pts = pts_polished.copy()
                    except Exception:
                        pass
    
    # Phase 3: Final tight polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 3:
        remaining = DEADLINE - (time.time() - start_time) - 2
        if remaining > 2:
            try:
                pts_final, e_final = _lbfgs_optimize(best_pts, remaining, tol=1e-12)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
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
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts

"""
Hybrid packing of 50 equal spheres in a unit cube.
Strategy:
1. Initialize using a 3D Hilbert curve (space-filling) for better initial distribution.
2. Global optimization using a combined potential (repulsion + soft-min) to escape poor local minima.
3. Local refinement using Simulated Annealing on the exact minimum-distance metric.
4. Final polish using L-BFGS-B on the soft-min potential for high precision.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_wall, min_pair / 2.0)
    return r if r > 0 else 0.0

def _hilbert_init(n, dim=3, order=10):
    """
    Generate points using a 3D Hilbert curve.
    This provides a space-filling initialization that is often better than a grid subset.
    """
    # Map n points to the Hilbert curve
    # We use a simple recursive Hilbert curve implementation for 3D
    # Note: Standard Hilbert curves are usually 1D->2D or 1D->3D.
    # For 3D, we can use a variant or just map 1D Hilbert to 3D coordinates.
    # A simpler approach: Use a 3D grid and traverse it in a Hilbert-like order, 
    # or use a library. Since we can't import external libs easily, we'll use a 
    # simplified "space-filling" strategy: 
    # 1. Create a fine grid.
    # 2. Sort points by a space-filling metric (e.g., Morton/Hilbert order).
    
    # Let's stick to a robust grid-based initialization with jitter, 
    # but ordered by a space-filling curve to ensure good distribution.
    # For N=50, a 4x4x4 grid (64 points) is fine. We select 50.
    # To improve, we can use a "quasi-random" or low-discrepancy sequence like Halton.
    # Halton is simple and effective.
    
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = np.zeros((n, 3))
    bases = [2, 3, 5]
    for i in range(1, n + 1):
        for j, base in enumerate(bases):
            points[i-1, j] = halton(i, base)
            
    # Scale to [-0.5, 0.5]
    points = points - 0.5
    return points

def _soft_min(x, t=100.0):
    """
    Approximate the minimum of a vector x using log-sum-exp.
    min(x) ~ -1/t * log(sum(exp(-t*x)))
    """
    # To avoid overflow, subtract max
    x_max = np.max(x)
    return x_max - (1.0/t) * np.log(np.sum(np.exp(-t*(x - x_max))))

def _combined_potential(x, t=100.0):
    """
    Combined objective: 
    1. Sum of inverse squared distances (repulsion) to spread points.
    2. -Soft-Min of pairwise distances (to maximize the minimum distance).
    3. -Soft-Min of wall distances (to maximize the minimum wall distance).
    
    Minimizing this function maximizes the minimum radius.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # Extract upper triangle to avoid duplicates
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    
    # Repulsion term: sum(1/d^2)
    repulsion = np.sum(1.0 / dists_sq[iu])
    
    # Soft-min of pairwise distances
    min_pair_soft = -_soft_min(pair_dists, t)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_soft = -_soft_min(wall_dists, t)
    
    # Objective to minimize:
    # We want to maximize min(min_pair, min_wall).
    # So we minimize -min(...).
    # We also want to spread them out, so we add repulsion.
    # Scale factors are important.
    # Repulsion is typically large. Soft-min is in range [-0.5, 0].
    
    # Let's try: Energy = repulsion * weight_rep + min_pair_soft * weight_pair + min_wall_soft * weight_wall
    # We want to minimize Energy.
    # min_pair_soft is negative when distances are small. We want it to be large (close to 0).
    # So minimizing min_pair_soft (which is negative) makes it more negative? No.
    # We want to MAXIMIZE min_pair. So we MINIMIZE -min_pair.
    # _soft_min returns approx min.
    # So we add -_soft_min(pair_dists) to the energy? No, that would minimize the min.
    # We want to MAXIMIZE the min. So we MINIMIZE -min.
    # So term is -1 * _soft_min(pair_dists).
    # Wait, _soft_min(x) approx min(x).
    # If we minimize -min(x), we maximize min(x).
    # So energy += -_soft_min(pair_dists) ? No, that's minimizing the negative of the min.
    # Let's check signs.
    # We want max(min_dists).
    # Equivalent to min(-min_dists).
    # So we add -_soft_min(pair_dists) to the objective to be minimized.
    
    energy = repulsion * 0.01 - _soft_min(pair_dists, t) - _soft_min(wall_dists, t)
    
    return energy

def _global_optimize(centers, max_time):
    """Run L-BFGS-B on the combined potential."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # L-BFGS-B is efficient for this
    try:
        res = minimize(
            _combined_potential, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
        )
        # Check if we ran out of time or if the result is valid
        if time.time() - start_time < max_time and res.x is not None:
             return res.x.reshape(N, 3)
    except Exception:
        pass
        
    return centers

def _local_refine(centers, max_time):
    """
    Simulated Annealing / Local Search on the exact min-radius metric.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    rng = np.random.RandomState(42)
    T_start = 0.01
    T_end = 0.0001
    n_steps = 20000 # Will be cut by time
    
    for step in range(n_steps):
        if time.time() - start_time > max_time:
            break
        
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        
        i = rng.randint(N)
        
        # Adaptive step size
        step_size = 0.01 * (1.0 - 0.5 * frac)
        delta = rng.uniform(-step_size, step_size, 3)
        
        new_centers = current_centers.copy()
        new_centers[i] += delta
        new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius >= current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            delta_e = new_radius - current_radius
            prob = np.exp(delta_e / (T * 0.01))
            if rng.random() < prob:
                current_centers = new_centers
                current_radius = new_radius

    return best_centers, best_radius

def _polish(centers, max_time):
    """
    Final high-precision polish using L-BFGS-B on the soft-min objective.
    """
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use a higher temperature for soft-min to be smoother, then lower?
    # Or just use a fixed t that balances smoothness and accuracy.
    # t=100 is quite sharp. t=50 might be smoother.
    # Let's try t=50 for polish to avoid getting stuck in sharp minima of the soft-min.
    
    try:
        res = minimize(
            lambda x: _combined_potential(x, t=50.0), 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 5000}
        )
        if time.time() - start_time < max_time and res.x is not None:
             return res.x.reshape(N, 3)
    except Exception:
        pass
        
    return centers

def construct():
    start_time = time.time()
    remaining = DEADLINE - (time.time() - start_time)
    
    # 1. Initial Halton (Quasi-random)
    centers = _hilbert_init(N)
    
    # 2. Global Optimization (Smooth Potential)
    # Allocate ~50% of time to global, 30% to local, 20% to polish
    time_global = remaining * 0.5
    time_local = remaining * 0.3
    time_polish = remaining * 0.2 - 5.0 # 5s safety margin
    
    if time_global > 5.0:
        centers = _global_optimize(centers, time_global)
        
    # 3. Local Refinement (Exact Metric)
    remaining_local = DEADLINE - (time.time() - start_time) - (time_polish + 2.0)
    if remaining_local > 1.0:
        centers, best_radius = _local_refine(centers, remaining_local)
    else:
        best_radius = _compute_radius(centers)
        
    # 4. Final Polish
    remaining_polish = DEADLINE - (time.time() - start_time) - 2.0
    if remaining_polish > 1.0:
        centers = _polish(centers, remaining_polish)
        
    # Final check
    final_radius = _compute_radius(centers)
    if final_radius > best_radius:
        best_radius = final_radius
        
    # Ensure validity
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        # Fallback
        return _hilbert_init(N)
        
    return centers

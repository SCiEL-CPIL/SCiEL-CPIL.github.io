"""Rewritten program: SA + L-BFGS polishing for packing 50 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE_MARGIN = 10.0  # seconds to leave before the 150s wall-clock limit

def _radius(centers):
    """Compute the largest common radius for the given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside the square (with a small tolerance for validity)
    if np.any(np.abs(centers) > HALF + 1e-12):
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = np.min(HALF - np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall_dist)

def _score(centers):
    """The objective to maximize: the common radius."""
    return _radius(centers)

def construct():
    start_time = time.time()
    
    # --- Phase 1: Generate a good initial configuration via simulated annealing ---
    # Start with a grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_radius = _score(best_centers)
    
    # Simulated Annealing
    # We want to maximize radius, so we minimize -radius.
    # But radius is non-smooth. We can use a smooth surrogate or just use the raw radius with SA.
    # Let's use a smooth surrogate: f = - (min_pair_dist/2 + min_wall_dist) / 2 is not quite right.
    # The radius is min(min_pair/2, min_wall).
    # A common smooth approximation for min(a,b) is -log(exp(-a/beta) + exp(-b/beta))*beta for small beta.
    # Or we can just optimize the raw radius with SA, accepting moves that improve it, and occasionally accepting worse ones.
    
    # Let's try to optimize a smooth potential: sum of inverse distances? No, that's for repulsion.
    # For packing, we want to maximize the minimum distance.
    # Let's stick to SA on the raw radius.
    
    sa_time_budget = 40.0  # seconds
    T = 0.01
    T_min = 1e-6
    alpha = 0.999
    
    current_centers = best_centers.copy()
    current_radius = best_radius
    
    while T > T_min:
        if time.time() - start_time > DEADLINE_MARGIN + sa_time_budget:
            break
            
        # Pick a random circle
        i = np.random.randint(N)
        # Random perturbation
        delta = np.random.normal(0, T, size=2)
        new_centers = current_centers.copy()
        new_centers[i] += delta
        # Clamp to square
        new_centers[i] = np.clip(new_centers[i], -HALF, HALF)
        
        new_radius = _score(new_centers)
        
        # Accept if better, or with probability exp((new_radius - current_radius) / T)
        if new_radius >= current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            prob = np.exp((new_radius - current_radius) / T)
            if np.random.random() < prob:
                current_centers = new_centers
                current_radius = new_radius
        
        T *= alpha
    
    # --- Phase 2: Local optimization using L-BFGS ---
    # We want to maximize the radius. L-BFGS minimizes.
    # Objective: minimize -radius. But radius is non-smooth.
    # Let's use a smooth surrogate.
    # f(x) = -min(min_pair/2, min_wall)
    # Approximate min(a,b) by -beta * log(exp(-a/beta) + exp(-b/beta)) for small beta.
    # Actually, min(a,b) = -max(-a,-b).
    # max(-a,-b) = beta * log(exp(-a/beta) + exp(-b/beta)).
    # So min(a,b) = -beta * log(exp(-a/beta) + exp(-b/beta)).
    
    def smooth_radius(centers, beta):
        # Compute all pair distances and wall distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        # We only need the minimum pair distance and minimum wall distance.
        # But for the soft-min, we need all of them? No, just the minimums.
        # Wait, the soft-min of a set of numbers is -beta * log(sum(exp(-x_i/beta))).
        # Here the set is {d_ij/2 for i<j} U {wall_dist_i for i}.
        # This is expensive to compute for all pairs.
        # Alternative: just use the raw radius and hope L-BFGS can handle the non-smoothness?
        # Or use a simpler surrogate: minimize the sum of (1/d_ij)^2 for pairs that are close?
        
        # Let's try a different approach: minimize the maximum "violation".
        # We want all d_ij >= 2r and wall_dist_i >= r.
        # Let t be the radius. We want to maximize t such that d_ij >= 2t and wall_dist_i >= t.
        # This is a feasibility problem for a given t.
        # We can binary search on t and use a local optimizer to find a configuration with radius t.
        
        # For now, let's just use L-BFGS to minimize -radius with a finite difference gradient.
        # This might be slow but let's try.
        
        pass
    
    # Let's use a binary search on the radius t, and for each t, try to find a configuration with radius >= t.
    # For a given t, the constraints are d_ij >= 2t and wall_dist_i >= t.
    # We can minimize the violation: sum(max(0, 2t - d_ij)^2 + sum(max(0, t - wall_dist_i)^2).
    
    def violation(centers, t):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_viol = np.maximum(0, 2*t - dists)
        wall_dists = HALF - np.abs(centers)
        wall_viol = np.maximum(0, t - wall_dists)
        return np.sum(pair_viol**2) + np.sum(wall_viol**2)
    
    # Binary search on t
    t_low = best_radius
    t_high = 0.08  # Upper bound
    
    # For each t, we try to minimize violation using L-BFGS.
    # If violation is close to 0, then t is feasible.
    
    # Let's do a few iterations of this.
    # But this might be slow.
    
    # Alternative: just polish the current best configuration with L-BFGS on -radius.
    # Let's try that.
    
    def neg_radius(centers):
        return -_radius(centers)
    
    # L-BFGS with finite differences
    # This might be slow. Let's limit the time.
    
    polish_time_budget = 20.0
    polish_deadline = start_time + sa_time_budget + polish_time_budget
    
    # We'll do a few L-BFGS runs from different starting points (the best from SA, and maybe some perturbations).
    
    for _ in range(3):  # 3 restarts
        if time.time() > polish_deadline:
            break
        
        # Start from best_centers
        x0 = best_centers.copy()
        
        # Add a small random perturbation to avoid getting stuck
        x0 += np.random.normal(0, 1e-4, size=(N, 2))
        x0 = np.clip(x0, -HALF, HALF)
        
        try:
            res = minimize(
                neg_radius,
                x0,
                method='L-BFGS-B',
                bounds=[(-HALF, HALF)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            new_centers = res.x.reshape(N, 2)
            new_radius = _radius(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    # --- Phase 3: Final ultra-tight polish ---
    # Use a higher precision or more iterations on the best configuration.
    
    final_time_budget = 10.0
    final_deadline = start_time + sa_time_budget + polish_time_budget + final_time_budget
    
    if time.time() < final_deadline:
        x0 = best_centers.copy()
        try:
            res = minimize(
                neg_radius,
                x0,
                method='L-BFGS-B',
                bounds=[(-HALF, HALF)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_centers = res.x.reshape(N, 2)
            new_radius = _radius(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Ensure the result is valid
    if not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    return best_centers

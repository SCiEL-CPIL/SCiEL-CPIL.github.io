import numpy as np
from scipy.optimize import basinhopping, Bounds
from scipy.spatial.distance import pdist
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    def compute_energy(pts, R):
        """
        Compute the sum of squared violations for a given target radius R.
        """
        # Pairwise
        dists = pdist(pts)
        min_dist_required = 2.0 * R
        pair_violations = np.maximum(min_dist_required - dists, 0.0)
        pair_energy = np.sum(pair_violations ** 2)
        
        # Walls
        wall_violations = np.maximum(np.abs(pts) - (0.5 - R), 0.0)
        wall_energy = np.sum(wall_violations ** 2)
        
        return pair_energy + wall_energy

    def compute_radius(pts):
        """Compute the valid radius for a configuration."""
        dists = pdist(pts)
        min_pair_dist = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist, min_wall_dist)

    # Target radius: known best is 0.1113823475125
    # We aim for slightly higher to see if we can beat it.
    R_target = 0.1114
    
    lower_bound = -0.5
    upper_bound = 0.5
    bounds = Bounds(np.full(N*3, lower_bound), np.full(N*3, upper_bound))
    
    def objective_flat(x):
        pts = x.reshape(N, 3)
        return compute_energy(pts, R_target)

    # Initial guess: Random points within the valid box for R_target
    min_coord = -0.5 + R_target
    max_coord = 0.5 - R_target
    rng = np.random.default_rng()
    
    best_pts = None
    best_radius = 0.0
    
    total_time_limit = 110.0
    start_time = time.time()
    
    try:
        x0 = rng.uniform(min_coord, max_coord, size=(N, 3)).flatten()
        
        res = basinhopping(
            objective_flat, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            minimizer_kwargs={'maxiter': 50},
            niter=5,
            stepsize=0.1,
            disp=False
        )
        pts_candidate = res.x.reshape(N, 3)
        pts_candidate = np.clip(pts_candidate, -0.5, 0.5)
        
        r_achieved = compute_radius(pts_candidate)
        
        if r_achieved > best_radius:
            best_radius = r_achieved
            best_pts = pts_candidate.copy()
            
    except Exception as e:
        pass

    # Polish with SA for remaining time
    remaining_time = total_time_limit - (time.time() - start_time)
    
    if best_pts is not None and remaining_time > 5.0:
        pts = best_pts.copy()
        current_energy = compute_energy(pts, R_target)
        T = 0.0005
        step_size = 0.005
        rng_polish = np.random.default_rng()
        
        while time.time() - start_time < total_time_limit:
            idx = rng_polish.integers(N)
            delta = rng_polish.normal(0, step_size, size=3)
            
            new_pts = pts.copy()
            new_pts[idx] = np.clip(new_pts[idx] + delta, -0.5, 0.5)
            
            new_energy = compute_energy(new_pts, R_target)
            
            if new_energy < current_energy or (T > 0 and rng_polish.random() < np.exp((current_energy - new_energy) / T)):
                pts = new_pts
                current_energy = new_energy
                
                if current_energy < 1e-6:
                     r_current = compute_radius(pts)
                     if r_current > best_radius:
                         best_radius = r_current
                         best_pts = pts.copy()
                
                T *= 0.999
                step_size *= 0.9999
    else:
        # Fallback if everything failed
        ticks = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_radius = compute_radius(best_pts)

    return best_pts
# EVOLVE-BLOCK-END

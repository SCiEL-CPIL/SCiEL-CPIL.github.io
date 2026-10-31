import numpy as np
import time

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def get_score(pts):
        """Calculate the score (max_d / min_d)^2 for a set of points."""
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        # Handle edge case of coincident points
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        max_d = np.max(d)
        return (max_d / min_d)**2

    def generate_hexagonal_seeds():
        """Generate multiple initial configurations based on hexagonal lattice."""
        seeds = []
        
        # Seed 1: Standard hexagonal lattice subset
        # Generate a large grid and pick the 16 closest to center
        R = 10
        pts_list = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                pts_list.append((x, y))
        pts_grid = np.array(pts_list)
        
        # Center the grid
        center = np.mean(pts_grid, axis=0)
        pts_grid -= center
        
        # Sort by distance to origin and take first 16
        dists_to_origin = np.sum(pts_grid**2, axis=1)
        sorted_indices = np.argsort(dists_to_origin)
        seed1 = pts_grid[sorted_indices[:N]].copy()
        seeds.append(seed1)
        
        # Seed 2: Perturbed hexagonal lattice
        # Add small random noise to break symmetry
        np.random.seed(42)
        seed2 = seed1 + np.random.normal(0, 0.1, (N, 2))
        seeds.append(seed2)
        
        # Seed 3: Square lattice subset
        pts_list_sq = []
        for x in range(-R, R + 1):
            for y in range(-R, R + 1):
                pts_list_sq.append((x, y))
        pts_sq = np.array(pts_list_sq)
        center_sq = np.mean(pts_sq, axis=0)
        pts_sq -= center_sq
        dists_sq = np.sum(pts_sq**2, axis=1)
        sorted_indices_sq = np.argsort(dists_sq)
        seed3 = pts_sq[sorted_indices_sq[:N]].copy()
        seeds.append(seed3)
        
        # Seed 4: Random points in a circle (normalized)
        np.random.seed(123)
        # Generate random angles and radii
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) # Uniform in area
        # Scale to roughly match hexagonal diameter
        seed4 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]) * 5.0
        seeds.append(seed4)

        # Seed 5: Another perturbation
        np.random.seed(999)
        seed5 = seed1 + np.random.normal(0, 0.5, (N, 2))
        seeds.append(seed5)

        return seeds

    def optimize_seed(initial_pts):
        """Optimize a single seed using Powell's method."""
        if time.time() > deadline - 10:
            return initial_pts, get_score(initial_pts)
        
        # Flatten parameters
        x0 = initial_pts.flatten()
        
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            pts = x.reshape(N, 2)
            return get_score(pts)
        
        # Powell is derivative-free and good for non-smooth functions
        # Set time limit per run to ensure we can try multiple seeds
        # maxiter is a fallback; time check in objective is primary guard
        try:
            # Use maxiter to limit CPU time roughly
            # 1000 iterations of Powell on 16 points (30 dims) might take a few seconds
            # We will run multiple seeds, so keep it moderate
            res = minimize(objective, x0, method='Powell', 
                           options={'maxiter': 500, 'maxfev': 2000, 'disp': False})
            if res.fun < objective(x0):
                best_pts = res.x.reshape(N, 2)
                best_score = res.fun
            else:
                best_pts = initial_pts
                best_score = objective(x0)
        except Exception:
            best_pts = initial_pts
            best_score = objective(x0)
            
        return best_pts, best_score

    # Generate seeds
    seeds = generate_hexagonal_seeds()
    
    best_pts_global = None
    best_score_global = float('inf')
    
    # Sort seeds by initial score to try best ones first?
    # Or just try them all.
    # Let's try all, but stop if time is low.
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
            
        # Quick check: if this seed is already worse than best, maybe skip?
        # No, optimization might improve it significantly.
        
        current_best_pts, current_best_score = optimize_seed(seed)
        
        if current_best_score < best_score_global:
            best_score_global = current_best_score
            best_pts_global = current_best_pts

    if best_pts_global is None:
        # Fallback to first seed if no optimization ran
        best_pts_global = seeds[0]
        best_score_global = get_score(best_pts_global)
        
    # Final polish: Run a high-precision Powell or Nelder-Mead on the best result
    # If we have time left, do one more pass with tighter tolerances
    if time.time() < deadline - 5 and best_pts_global is not None:
        x0 = best_pts_global.flatten()
        
        def objective_polish(x):
            if time.time() > deadline - 2:
                return 1e10
            pts = x.reshape(N, 2)
            return get_score(pts)
            
        try:
            res = minimize(objective_polish, x0, method='Powell', 
                           options={'maxiter': 1000, 'maxfev': 5000, 'disp': False})
            if res.fun < best_score_global:
                best_pts_global = res.x.reshape(N, 2)
                best_score_global = res.fun
        except Exception:
            pass

    return best_pts_global

# EVOLVE-BLOCK-END

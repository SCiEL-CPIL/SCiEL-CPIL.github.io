"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Compute min distance between pairs
    from scipy.spatial.distance import pdist
    dists = pdist(centers)
    if len(dists) == 0:
        return 0.0
    min_pair_dist = np.min(dists)
    # Compute min distance to walls
    wall_dists = []
    for i in range(3):
        wall_dists.append(np.min(0.5 - np.abs(centers[:, i])))
    min_wall = min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial guess: grid-based
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # Use L-BFGS-B to maximize the radius
    # We maximize r such that:
    #   |c_i - c_j| >= 2r for all i < j
    #   0.5 - |c_ik| >= r for all i, k
    # This is a non-smooth problem. We use a smooth approximation.
    
    def objective(params):
        """Objective to minimize (negative radius proxy)."""
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        centers = params.reshape(N, 3)
        # Smooth min using soft-min
        # For pairs
        dists = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        # Wall distances
        wall_d = []
        for k in range(3):
            wall_d.append(0.5 - np.abs(centers[:, k]))
        wall_d = np.array(wall_d).T  # (N, 3)
        all_constraints = np.concatenate([dists.flatten(), wall_d.flatten()])
        all_constraints = all_constraints[all_constraints > 0]
        if len(all_constraints) == 0:
            return 1e10
        # Soft min: -log(sum(exp(-x/t))) * t
        t = 0.01
        soft_min = -t * np.log(np.sum(np.exp(-all_constraints / t)))
        return -soft_min
    
    # We'll do iterative refinement with decreasing t
    current = centers.copy()
    for t_val in [0.05, 0.02, 0.01, 0.005, 0.002]:
        if time.time() > deadline:
            break
        
        def obj(params):
            elapsed = time.time() - start_time
            if elapsed > deadline:
                return 1e10
            c = params.reshape(N, 3)
            dists = np.sqrt(np.sum((c[:, None, :] - c[None, :, :])**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            wall_d = []
            for k in range(3):
                wall_d.append(0.5 - np.abs(c[:, k]))
            wall_d = np.array(wall_d).T
            all_c = np.concatenate([dists.flatten(), wall_d.flatten()])
            all_c = all_c[all_c > 0]
            if len(all_c) == 0:
                return 1e10
            soft_min = -t_val * np.log(np.sum(np.exp(-all_c / t_val)))
            return -soft_min
        
        res = minimize(obj, current.ravel(), method='L-BFGS-B', 
                      options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 20})
        if res.success or res.fun < 1e9:
            current = res.x.reshape(N, 3)
            s = _score(current)
            if s > best_score:
                best_score = s
                best_centers = current.copy()
    
    # Final polish: try to improve with direct optimization of the min
    # Use the best_centers as starting point
    if time.time() < deadline - 5:
        # Try a few random restarts with small perturbations
        for restart in range(5):
            if time.time() > deadline:
                break
            perturbed = best_centers + np.random.normal(0, 0.001, (N, 3))
            perturbed = np.clip(perturbed, -0.499, 0.499)
            
            def obj2(params):
                elapsed = time.time() - start_time
                if elapsed > deadline:
                    return 1e10
                c = params.reshape(N, 3)
                dists = np.sqrt(np.sum((c[:, None, :] - c[None, :, :])**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                wall_d = []
                for k in range(3):
                    wall_d.append(0.5 - np.abs(c[:, k]))
                wall_d = np.array(wall_d).T
                all_c = np.concatenate([dists.flatten(), wall_d.flatten()])
                all_c = all_c[all_c > 0]
                if len(all_c) == 0:
                    return 1e10
                t_val = 0.001
                soft_min = -t_val * np.log(np.sum(np.exp(-all_c / t_val)))
                return -soft_min
            
            res = minimize(obj2, perturbed.ravel(), method='L-BFGS-B',
                          options={'maxiter': 100, 'gtol': 1e-9, 'maxls': 10})
            if res.fun < 1e9:
                c = res.x.reshape(N, 3)
                s = _score(c)
                if s > best_score:
                    best_score = s
                    best_centers = c.copy()
    
    return best_centers
# EVOLVE-BLOCK-END

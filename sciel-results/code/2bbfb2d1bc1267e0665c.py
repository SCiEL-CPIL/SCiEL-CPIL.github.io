"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time


def _score(centers):
    """Compute the radius for a given set of centers.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
    """
    centers = np.asarray(centers, dtype=np.float64)
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to boundaries
    min_bound = np.min(0.5 - np.abs(centers))
    return min(min_pair, min_bound)


def _negative_score(centers_flat):
    """Objective for minimization: negative of the radius."""
    centers = centers_flat.reshape(N, 3)
    return -_score(centers)


def construct():
    """
    Use differential evolution with multi-start refinement to maximize the minimum
    distance between sphere centers and from boundaries.
    
    We reformulate the problem as minimizing -r where r is the common radius.
    Differential evolution is a global optimizer that works well for this type
    of non-smooth objective.
    """
    # Time budget
    t_start = time.time()
    time_budget = 110  # seconds, leaving 10s margin
    
    # Bounds for each coordinate: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use differential evolution with a reasonable population size
    # and max iterations, but we'll monitor time
    best_result = None
    best_score_val = -np.inf
    
    # Multiple restarts with different seeds for diversity
    for seed in range(5):
        if time.time() - t_start > time_budget:
            break
        
        rng = np.random.default_rng(seed * 1000 + 42)
        
        # Initial population: mix of grid-based and random
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        # Create initial population
        pop_size = 30
        x0_list = []
        for p in range(pop_size):
            if p == 0:
                # Grid-based start
                centers = grid[:N].copy()
                centers += rng.normal(0, 0.01, size=centers.shape)
                centers = np.clip(centers, -0.49, 0.49)
                x0 = centers.flatten()
            elif p == 1:
                # Another grid with different perturbation
                centers = grid[:N].copy()
                centers += rng.normal(0, 0.02, size=centers.shape)
                centers = np.clip(centers, -0.49, 0.49)
                x0 = centers.flatten()
            else:
                # Random starts
                x0 = rng.uniform(-0.45, 0.45, size=N * 3)
            x0_list.append(x0)
        
        # Run differential evolution
        result = differential_evolution(
            _negative_score,
            bounds,
            x0=x0_list,
            maxiter=200,
            tol=1e-12,
            popsize=pop_size,
            mutation=(0.5, 1.5),
            recombination=0.8,
            seed=seed * 1000 + 42,
            polish=False,
            init='best2'
        )
        
        trial_score = -result.fun
        if trial_score > best_score_val:
            best_score_val = trial_score
            best_result = result.x.reshape(N, 3)
        
        # Check time
        if time.time() - t_start > time_budget - 20:
            break
    
    # Local refinement using Nelder-Mead on the best solution
    if best_result is not None and time.time() - t_start < time_budget - 10:
        from scipy.optimize import minimize
        
        # Try a few Nelder-Mead refinements
        for nm_seed in range(3):
            if time.time() - t_start > time_budget - 5:
                break
            
            rng = np.random.default_rng(nm_seed * 7 + 123)
            # Perturb the best result slightly
            x0 = best_result.flatten() + rng.normal(0, 0.005, size=N * 3)
            x0 = np.clip(x0, -0.49, 0.49)
            
            res = minimize(
                _negative_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            trial_score = -res.fun
            if trial_score > best_score_val:
                best_score_val = trial_score
                best_result = res.x.reshape(N, 3)
    
    # Ensure the result is valid (all points inside the cube)
    if best_result is None:
        # Fallback: grid-based
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_result = grid[:N].copy()
    
    # Final clip to ensure validity
    best_result = np.clip(best_result, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_result
# EVOLVE-BLOCK-END

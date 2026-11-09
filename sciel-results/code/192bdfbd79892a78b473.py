"""Rewritten program: optimize 100 equal spheres in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
SEED_RADIUS = 0.1113823475125  # best known (Specht)

def _score(centers):
    """Compute the radius score for a given set of centers."""
    # Centers must be in [-0.5, 0.5]^3
    # Distance to nearest wall
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use efficient computation
    from scipy.spatial.distance import pdist
    d = pdist(centers)
    min_pair = np.min(d)
    
    r = min(min_pair / 2.0, min_wall)
    return r

def _neg_score(centers_flat):
    """Negative score for minimization."""
    centers = centers_flat.reshape(N, 3)
    return -_score(centers)

def construct():
    """Construct centers of 100 equal spheres in the unit cube [-0.5, 0.5]^3."""
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s
    
    # Start from a grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_score = _score(best_centers)
    
    # Try multiple random restarts with local optimization
    rng = np.random.default_rng(42)
    
    # We'll use Nelder-Mead or Powell with the negative score
    # But first, let's try a smarter initialization
    
    # Strategy: Start from grid, then do local search
    # Also try some random initializations
    
    attempts = []
    
    # Attempt 1: Grid-based
    attempts.append(best_centers.copy())
    
    # Attempt 2: Slightly perturbed grid
    perturbed = best_centers.copy()
    perturbed += rng.normal(0, 0.01, size=perturbed.shape)
    # Project back to cube
    perturbed = np.clip(perturbed, -0.5, 0.5)
    attempts.append(perturbed)
    
    # Attempt 3: Random initialization
    random_init = rng.uniform(-0.45, 0.45, size=(N, 3))
    attempts.append(random_init)
    
    # Attempt 4: Another random
    random_init2 = rng.uniform(-0.4, 0.4, size=(N, 3))
    attempts.append(random_init2)
    
    # Try to optimize each attempt
    for i, init in enumerate(attempts):
        if time.time() > deadline - 20:
            break
        
        try:
            init_flat = init.flatten()
            # Use Powell method which doesn't require gradients
            result = minimize(
                _neg_score,
                init_flat,
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            opt_centers = result.x.reshape(N, 3)
            
            # Project to valid cube
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            
            opt_score = _score(opt_centers)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
                print(f"Attempt {i}: improved to {best_score:.10f}")
        except Exception as e:
            print(f"Attempt {i} failed: {e}")
            continue
    
    # If we have time, do a final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            best_flat = best_centers.flatten()
            result = minimize(
                _neg_score,
                best_flat,
                method='Powell',
                options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            opt_centers = result.x.reshape(N, 3)
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            opt_score = _score(opt_centers)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
                print(f"Polish: improved to {best_score:.10f}")
        except Exception as e:
            print(f"Polish failed: {e}")
    
    # Final safety check
    if not np.all(np.isfinite(best_centers)):
        # Fall back to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    # Ensure centers are strictly inside or on boundary
    best_centers = np.clip(best_centers, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_centers

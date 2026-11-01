import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.stats.qmc import Sobol
from scipy.optimize import minimize

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        # Allow 145s out of 150s to be safe
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    # cdist is generally faster than manual broadcasting for N=100
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the negative soft-min approximation of the radius.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    We want to maximize soft_min, so we minimize -soft_min.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances: shape (N, 3) -> (3N,)
    wall_dists = 0.5 - np.abs(centers)
    wall_vals = wall_dists.ravel()
    
    # 2. Pairwise distances / 2
    # We only need unique pairs (i, j) where i < j
    # cdist gives full matrix. We can extract upper triangle.
    dists = cdist(centers, centers, metric='euclidean')
    # Get indices for upper triangle (excluding diagonal)
    # iu is precomputed indices for upper triangle
    pair_dists = dists[iu] / 2.0
    
    # Combine all constraint values
    # To avoid overflow in exp(-beta * x), we use the log-sum-exp trick:
    # log(sum(exp(-beta * x))) = max(-beta * x) + log(sum(exp(-beta * x - max(-beta * x))))
    # Let y = -beta * x. We want sum(exp(y)).
    # max_y = max(y) = -beta * min(x)
    
    all_vals = np.concatenate([wall_vals, pair_dists])
    
    # If beta is very high, -beta * all_vals can be very negative.
    # We subtract the minimum of all_vals (which corresponds to max of -beta*all_vals)
    min_val = np.min(all_vals)
    
    # Calculate -beta * (all_vals - min_val). This is <= 0.
    # exp of this is in (0, 1].
    shifted = -beta * (all_vals - min_val)
    
    # sum_exp = sum(exp(shifted))
    sum_exp = np.sum(np.exp(shifted))
    
    # soft_min = min_val - (1/beta) * log(sum_exp)
    # Note: log(sum_exp) is log(sum(exp(-beta*(x-min_val))))
    # log(sum(exp(-beta*x))) = log(sum(exp(-beta*x + beta*min_val))) = beta*min_val + log(sum(exp(-beta*(x-min_val))))
    # So -1/beta * log(sum(exp(-beta*x))) = -min_val - 1/beta * log(sum_exp)
    # Wait, the formula is:
    # min(x) approx -1/beta * log(sum(exp(-beta*x)))
    # Let's re-derive carefully.
    # LSE(y) = log(sum(exp(y))).
    # We want approx min(x).
    # Consider f(x) = -1/beta * log(sum(exp(-beta*x))).
    # As beta -> inf, this goes to min(x).
    # Let x_min = min(x).
    # sum(exp(-beta*x)) = exp(-beta*x_min) * sum(exp(-beta*(x - x_min))).
    # log(sum(exp(-beta*x))) = -beta*x_min + log(sum(exp(-beta*(x - x_min)))).
    # -1/beta * log(...) = x_min - 1/beta * log(sum(exp(-beta*(x - x_min)))).
    
    # So, soft_min = min_val - (1.0 / beta) * np.log(sum_exp)
    
    soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
    
    # We want to maximize soft_min_val, so we minimize -soft_min_val
    return -soft_min_val

def _init_sobol():
    """
    Generate N points using a Sobol sequence.
    """
    sampler = Sobol(d=3, scramble=True, seed=42)
    points = sampler.random(N)
    centers = points * 1.0 - 0.5
    # Add tiny noise to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _init_bcc():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
    nx, ny, nz = 5, 5, 2
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _polish_l_bfgs(centers, time_budget=10.0, iu=None):
    """
    Local search using L-BFGS-B with soft-min objective and increasing beta.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    if iu is None:
        # Precompute upper triangle indices for pairwise distances
        iu = np.triu_indices(N, k=1)
    
    start_time = time.time()
    
    # Beta schedule: start low, increase to sharpen
    # Logarithmic spacing
    beta_start = 20.0
    beta_end = 2000.0
    n_steps = 8
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if _time_left() < 2.0 or (time.time() - start_time) > time_budget:
            break
            
        # Define objective with current beta
        def obj(x):
            return _soft_min_objective(x, beta, iu)
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Run minimize
        # Reduce maxiter as beta increases to avoid getting stuck in narrow valleys
        # or use fixed reasonable maxiter
        max_iter = 100 if beta < 100 else 50
        
        try:
            res = minimize(
                obj, 
                centers.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            r_current = _compute_radius(new_centers)
            
            if r_current > best_r + 1e-12:
                best_r = r_current
                best_centers = new_centers.copy()
                centers = new_centers.copy()
            else:
                # If no improvement, keep old centers but maybe try again with same beta?
                # Or just move to next beta.
                pass
                
        except Exception as e:
            # If optimization fails, break
            break
            
    return best_centers, best_r

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # List of initializers
    inits = [
        ("sobol", _init_sobol),
        ("bcc", _init_bcc),
    ]
    
    # Time allocation
    # We have ~145s.
    # Try each init with a budget.
    # If we have 2 inits, give ~60s each?
    # Or try one, if it's good, spend remaining on it?
    # Let's try both, but keep the best.
    
    time_per_init = 50.0
    
    for name, init_func in inits:
        if _time_left() < time_per_init + 5.0:
            break
            
        centers = init_func()
        r_init = _compute_radius(centers)
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        # Run polish
        # Use the current best centers as starting point for polish?
        # Or polish the specific init?
        # Polishing the specific init might find a better local min for that basin.
        # But if the init is bad, polishing won't help much.
        # Let's polish the current init.
        
        budget = min(time_per_init, _time_left() - 5.0)
        
        if budget > 5.0:
            polished_centers, polished_r = _polish_l_bfgs(centers, time_budget=budget, iu=iu)
            if polished_r > best_radius:
                best_radius = polished_r
                best_centers = polished_centers.copy()
                
    # Final polish with all remaining time on the best candidate found so far
    if best_centers is not None and _time_left() > 5.0:
        final_budget = _time_left() - 2.0
        # Use a higher beta range for final polish?
        # The _polish_l_bfgs function already ramps up beta.
        # We can just call it again. It might find a slightly better spot.
        polished_centers, polished_r = _polish_l_bfgs(best_centers, time_budget=final_budget, iu=iu)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()
            
    if best_centers is None:
        best_centers = _init_sobol()
        
    # Final validation and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

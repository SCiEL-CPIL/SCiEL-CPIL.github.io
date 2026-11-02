import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.stats.qmc import Sobol, Halton
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

def _repulsion_objective(centers_flat, r_target, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the sum of squared violations of the radius constraint r_target.
    Violation = max(0, r_target - distance)
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_violations = np.maximum(0, r_target - wall_dists)
    
    # 2. Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    pair_dists = dists[iu] / 2.0
    pair_violations = np.maximum(0, r_target - pair_dists)
    
    # Sum of squares
    return np.sum(wall_violations**2) + np.sum(pair_violations**2)

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

def _init_grid():
    """
    Build a simple grid initialization.
    """
    # 5x5x4 grid = 100 points
    nx, ny, nz = 5, 5, 4
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
    
    centers = np.array(points[:N])
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _init_halton():
    """
    Generate N points using a Halton sequence.
    """
    sampler = Halton(d=3, scramble=True, seed=42)
    points = sampler.random(N)
    centers = points * 1.0 - 0.5
    # Add tiny noise to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _simulated_annealing(centers, iu, time_budget=15.0):
    """
    Simulated annealing to escape local minima.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    T_start = 0.05
    T_end = 0.001
    n_steps = 500
    
    # Precompute upper triangle indices for fast distance computation
    iu = np.triu_indices(N, k=1)
    
    for step in range(n_steps):
        if _time_left() < 2.0 or (time.time() - start_time) > time_budget:
            break
        
        # Annealing schedule
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Generate perturbation
        noise = rng.normal(0, T, centers.shape)
        perturbed = centers + noise
        perturbed = np.clip(perturbed, -0.4999, 0.4999)
        
        # Quick evaluation
        r_current = _compute_radius(perturbed)
        
        if r_current > best_r:
            best_r = r_current
            best_centers = perturbed.copy()
            centers = perturbed.copy()
        else:
            # Metropolis acceptance
            delta_r = r_current - best_r
            if delta_r > 0 or rng.random() < np.exp(delta_r / T):
                centers = perturbed.copy()
    
    return best_centers, best_r

def _polish_l_bfgs(centers, time_budget=10.0, iu=None):
    """
    Local search using L-BFGS-B with soft-min objective and increasing beta.
    Includes a final high-precision polish pass.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    if iu is None:
        # Precompute upper triangle indices for pairwise distances
        iu = np.triu_indices(N, k=1)
    
    start_time = time.time()
    
    # Beta schedule: start low, increase to sharpen
    # Spend more time at higher beta values for precision
    beta_start = 20.0
    beta_end = 10000.0
    n_steps = 12
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for i, beta in enumerate(betas):
        if _time_left() < 2.0 or (time.time() - start_time) > time_budget:
            break
            
        # Define objective with current beta
        def obj(x):
            return _soft_min_objective(x, beta, iu)
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Run minimize
        # Increase maxiter for higher beta to allow convergence in narrow valleys
        if beta < 50:
            max_iter = 80
            ftol = 1e-12
            gtol = 1e-8
        elif beta < 200:
            max_iter = 120
            ftol = 1e-14
            gtol = 1e-10
        elif beta < 1000:
            max_iter = 150
            ftol = 1e-14
            gtol = 1e-10
        else:
            max_iter = 250
            ftol = 1e-15
            gtol = 1e-12
        
        try:
            res = minimize(
                obj, 
                centers.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            r_current = _compute_radius(new_centers)
            
            if r_current > best_r + 1e-12:
                best_r = r_current
                best_centers = new_centers.copy()
                centers = new_centers.copy()
            else:
                # If no improvement, keep old centers
                pass
                
        except Exception as e:
            # If optimization fails, break
            break
            
    # Final high-precision polish pass with very high beta
    if _time_left() > 5.0 and (time.time() - start_time) < time_budget - 2.0:
        final_beta = 20000.0
        def obj_final(x):
            return _soft_min_objective(x, final_beta, iu)
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj_final, 
                best_centers.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            r_current = _compute_radius(new_centers)
            
            if r_current > best_r + 1e-12:
                best_r = r_current
                best_centers = new_centers.copy()
                
        except Exception as e:
            pass
            
    return best_centers, best_r

def _polish_repulsion(centers, time_budget=10.0, iu=None):
    """
    Local search using L-BFGS-B with hard constraint repulsion.
    Iteratively increases target radius and minimizes violations.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    if iu is None:
        iu = np.triu_indices(N, k=1)
    
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Start with current radius
    r_target = best_r
    
    # We will try to increase r_target in small steps
    # If we can satisfy the constraint (energy ~ 0), we accept the new radius
    # and increase r_target further.
    
    # To avoid infinite loops or getting stuck, we limit the number of steps
    # and the time per step.
    
    n_attempts = 20
    for i in range(n_attempts):
        if _time_left() < 2.0 or (time.time() - start_time) > time_budget:
            break
        
        # Increase target radius slightly
        # The increment should be small enough for the optimizer to handle
        # but large enough to make progress.
        # 1e-4 seems reasonable for this scale.
        r_target_new = r_target * (1 + 1e-4) + 1e-6
        
        def obj(x):
            return _repulsion_objective(x, r_target_new, iu)
        
        try:
            res = minimize(
                obj, 
                centers.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            # Check if we actually achieved the new radius
            r_achieved = _compute_radius(new_centers)
            
            if r_achieved >= r_target_new - 1e-9:
                # Success! Update best and target
                best_r = r_achieved
                best_centers = new_centers.copy()
                centers = new_centers.copy()
                r_target = r_target_new
            else:
                # Failure to reach target. 
                # We might still have improved slightly, so check.
                if r_achieved > best_r:
                    best_r = r_achieved
                    best_centers = new_centers.copy()
                    centers = new_centers.copy()
                    r_target = r_achieved
                else:
                    # Stuck. Break to save time for final polish.
                    break
                    
        except Exception:
            break
            
    # Final polish on the best configuration using soft-min for high precision
    if _time_left() > 2.0 and (time.time() - start_time) < time_budget:
        # Use a high beta soft-min polish to refine the result
        final_beta = 5000.0
        def obj_final(x):
            return _soft_min_objective(x, final_beta, iu)
        
        try:
            res = minimize(
                obj_final, 
                best_centers.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            r_current = _compute_radius(new_centers)
            
            if r_current > best_r:
                best_r = r_current
                best_centers = new_centers.copy()
                
        except Exception:
            pass
            
    return best_centers, best_r

def _perturb_and_reoptimize(centers, iu, time_budget=10.0):
    """
    Perturb the configuration and re-optimize to escape local minima.
    Returns improved centers if perturbation leads to better result.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Try several perturbations with decreasing step sizes
    step_sizes = [0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step_size in step_sizes:
        if _time_left() < 2.0 or (time.time() - start_time) > time_budget:
            break
        
        # Generate perturbation
        noise = rng.normal(0, step_size, centers.shape)
        perturbed = centers + noise
        perturbed = np.clip(perturbed, -0.4999, 0.4999)
        
        # Quick polish with moderate beta
        beta = 1000.0
        def obj(x):
            return _soft_min_objective(x, beta, iu)
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj, 
                perturbed.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            r_current = _compute_radius(new_centers)
            
            if r_current > best_r + 1e-12:
                best_r = r_current
                best_centers = new_centers.copy()
                
        except Exception as e:
            pass
            
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
        ("grid", _init_grid),
        ("halton", _init_halton),
    ]
    
    # Time allocation
    # We have ~145s.
    # Give each init ~25s, then use remaining for SA, perturbation and final polish
    
    time_per_init = 25.0
    
    for name, init_func in inits:
        if _time_left() < time_per_init + 5.0:
            break
            
        centers = init_func()
        r_init = _compute_radius(centers)
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        # Run polish
        budget = min(time_per_init, _time_left() - 5.0)
        
        if budget > 5.0:
            # Use the new repulsion polish which is often more robust
            polished_centers, polished_r = _polish_repulsion(centers, time_budget=budget, iu=iu)
            if polished_r > best_radius:
                best_radius = polished_r
                best_centers = polished_centers.copy()
                
    # Simulated annealing step to escape local minima
    if best_centers is not None and _time_left() > 20.0:
        sa_budget = min(15.0, _time_left() - 15.0)
        sa_centers, sa_r = _simulated_annealing(best_centers, iu, time_budget=sa_budget)
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_centers.copy()
            
    # Perturbation step to escape local minima
    if best_centers is not None and _time_left() > 15.0:
        perturb_budget = min(15.0, _time_left() - 10.0)
        perturbed_centers, perturbed_r = _perturb_and_reoptimize(best_centers, iu, time_budget=perturb_budget)
        if perturbed_r > best_radius:
            best_radius = perturbed_r
            best_centers = perturbed_centers.copy()
            
    # Final polish with all remaining time on the best candidate found so far
    if best_centers is not None and _time_left() > 5.0:
        final_budget = _time_left() - 2.0
        # Use repulsion polish again for final push
        polished_centers, polished_r = _polish_repulsion(best_centers, time_budget=final_budget, iu=iu)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()
            
    if best_centers is None:
        best_centers = _init_sobol()
        
    # Final validation and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

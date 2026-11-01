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
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Soft-min objective: maximize the minimum of (pairwise_dist/2, wall_dist).
    We minimize the negative soft-min.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraints into one array
    all_vals = np.concatenate([pair_dists.ravel(), wall_dists.ravel()])
    
    # Soft min via log-sum-exp trick
    max_val = np.max(all_vals)
    soft_min_val = max_val - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (all_vals - max_val))))
    
    # We want to maximize soft_min, so we minimize -soft_min
    return -soft_min_val

def _init_sobol():
    """
    Generate N points using a Sobol sequence (quasi-random, low discrepancy).
    """
    # Skip the first few points as they can be biased towards corners in some implementations
    sampler = Sobol(d=3, scramble=True, seed=42)
    # Generate slightly more than needed to skip initial points if desired, or just use N
    points = sampler.random(N)
    # Map [0, 1] to [-0.5, 0.5]
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

def _init_random(seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, (N, 3))
    return centers

def _polish_l_bfgs(centers, time_budget=10.0):
    """
    Local search using L-BFGS-B with soft-min objective.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    
    # Start with a moderate beta
    beta = 50.0
    
    while _time_left() > 1.0 and (time.time() - start_time) < time_budget:
        # Run L-BFGS for a fixed number of iterations or until time runs out
        # We'll run it in chunks to allow for beta adaptation
        
        # Define objective with current beta
        def obj(x):
            return _soft_min_objective(x, beta, None)
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Run minimize
        try:
            res = minimize(
                obj, 
                centers.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            centers = res.x.reshape(-1, 3)
            centers = np.clip(centers, -0.4999, 0.4999)
            
            r_current = _compute_radius(centers)
            
            if r_current > best_r + 1e-12:
                best_r = r_current
                best_centers = centers.copy()
                
            # Increase beta for sharper min approximation
            beta *= 1.5
            
            # If we made no progress, reduce beta? Or just break?
            # For now, just keep increasing beta to sharpen the landscape.
            
        except Exception:
            break
        
        # Check time
        if _time_left() < 2.0:
            break
            
    return best_centers, best_r

def _polish_sa(centers, time_budget=10.0):
    """
    Simulated Annealing using soft-min objective for energy.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    
    # SA Parameters
    T = 0.05
    T_min = 1e-4
    decay = 0.995
    beta = 100.0 # Fixed beta for SA energy, high enough to be close to true min
    
    rng = np.random.default_rng(42)
    
    while _time_left() > 1.0 and (time.time() - start_time) < time_budget:
        # Compute current energy (negative soft-min)
        current_energy = _soft_min_objective(centers.ravel(), beta, None)
        
        # Propose move: perturb a random subset of points
        num_points = rng.integers(1, 10)
        indices = rng.choice(N, num_points, replace=False)
        
        # Gaussian perturbation
        perturbations = rng.normal(0, T, (num_points, 3))
        new_centers = centers.copy()
        new_centers[indices] += perturbations
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        # Compute new energy
        new_energy = _soft_min_objective(new_centers.ravel(), beta, None)
        
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            centers = new_centers
            r_current = _compute_radius(centers)
            if r_current > best_r + 1e-12:
                best_r = r_current
                best_centers = centers.copy()
        
        T *= decay
        if T < T_min:
            T = T_min
            # Restart with higher T if stuck? Or just break?
            # For now, keep going with T_min.

    return best_centers, best_r

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    best_centers = None
    best_radius = 0.0
    
    # List of initializers
    inits = [
        ("sobol", _init_sobol),
        ("bcc", _init_bcc),
        ("rand1", lambda: _init_random(1)),
        ("rand2", lambda: _init_random(2)),
        ("rand3", lambda: _init_random(3)),
    ]
    
    # Allocate time. We have ~145s.
    # Give each init a chance.
    
    time_per_init = 20.0
    
    for name, init_func in inits:
        if _time_left() < time_per_init + 5.0:
            break
            
        centers = init_func()
        r_init = _compute_radius(centers)
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        # Run polish: Hybrid SA + L-BFGS
        budget = min(time_per_init, _time_left() - 5.0)
        
        if budget > 2.0:
            # 1. Simulated Annealing for global escape
            sa_budget = budget * 0.4
            polished_centers, polished_r = _polish_sa(centers, time_budget=sa_budget)
            if polished_r > best_radius:
                best_radius = polished_r
                best_centers = polished_centers.copy()
            
            # 2. L-BFGS for local precision
            lbfgs_budget = budget - sa_budget
            if lbfgs_budget > 2.0:
                polished_centers2, polished_r2 = _polish_l_bfgs(best_centers, time_budget=lbfgs_budget)
                if polished_r2 > best_radius:
                    best_radius = polished_r2
                    best_centers = polished_centers2.copy()
                
    # Final polish with all remaining time
    if best_centers is not None and _time_left() > 5.0:
        final_budget = _time_left() - 2.0
        # Use L-BFGS for final high-precision polish
        polished_centers, polished_r = _polish_l_bfgs(best_centers, time_budget=final_budget)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()
            
    if best_centers is None:
        best_centers = _init_sobol()
        
    return best_centers

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
    Objective function for L-BFGS-B.
    Minimizes the negative soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances: shape (N, 3) -> (3N,)
    wall_dists = 0.5 - np.abs(centers)
    wall_vals = wall_dists.ravel()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    pair_dists = dists[iu] / 2.0
    
    all_vals = np.concatenate([wall_vals, pair_dists])
    
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    sum_exp = np.sum(np.exp(shifted))
    soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
    
    return -soft_min_val

def _init_sobol():
    """
    Generate N points using a Sobol sequence.
    """
    sampler = Sobol(d=3, scramble=True, seed=42)
    points = sampler.random(N)
    centers = points * 1.0 - 0.5
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

def _init_halton():
    """
    Build a Halton sequence initialization.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        points.append([halton(i, 2), halton(i, 3), halton(i, 5)])
    
    centers = np.array(points)
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
        iu = np.triu_indices(N, k=1)
    
    start_time = time.time()
    
    beta_start = 20.0
    beta_end = 5000.0
    n_steps = 10
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for i, beta in enumerate(betas):
        if _time_left() < 2.0 or (time.time() - start_time) > time_budget:
            break
            
        def obj(x):
            return _soft_min_objective(x, beta, iu)
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        max_iter = 150 if beta < 100 else 80
        
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
                
        except Exception:
            break
            
    return best_centers, best_r

def _simulated_annealing(centers, iu, time_budget=30.0):
    """
    Simulated annealing to escape local minima.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.99
    
    while _time_left() > 2.0 and (time.time() - start_time) < time_budget:
        if T < T_min:
            break
            
        # Pick a random sphere and perturb it
        idx = np.random.randint(N)
        delta = np.random.normal(0, T, 3)
        new_centers = centers.copy()
        new_centers[idx] += delta
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_r = _compute_radius(new_centers)
        
        # Accept if better or with probability exp((new_r - current_r)/T)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centers = new_centers.copy()
        
        delta_r = new_r - _compute_radius(centers)
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            centers = new_centers
        
        T *= cooling_rate
    
    return best_centers, best_r

def _perturb_and_reoptimize(centers, iu, time_budget=20.0):
    """
    Perturb the best centers and re-optimize.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    while _time_left() > 5.0 and (time.time() - start_time) < time_budget:
        # Perturb a random subset of points
        n_perturb = np.random.randint(5, 20)
        indices = rng.choice(N, n_perturb, replace=False)
        
        perturbed = centers.copy()
        noise = rng.normal(0, 0.01, (n_perturb, 3))
        perturbed[indices] += noise
        perturbed = np.clip(perturbed, -0.499, 0.499)
        
        # Re-optimize
        polished, r = _polish_l_bfgs(perturbed, time_budget=min(10.0, _time_left() - 5.0), iu=iu)
        
        if r > best_r + 1e-12:
            best_r = r
            best_centers = polished.copy()
    
    return best_centers, best_r

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    inits = [
        ("sobol", _init_sobol),
        ("bcc", _init_bcc),
        ("halton", _init_halton),
    ]
    
    time_per_init = 30.0
    
    for name, init_func in inits:
        if _time_left() < time_per_init + 5.0:
            break
            
        centers = init_func()
        r_init = _compute_radius(centers)
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        
        budget = min(time_per_init, _time_left() - 5.0)
        
        if budget > 5.0:
            polished_centers, polished_r = _polish_l_bfgs(centers, time_budget=budget, iu=iu)
            if polished_r > best_radius:
                best_radius = polished_r
                best_centers = polished_centers.copy()
    
    # Simulated annealing
    if best_centers is not None and _time_left() > 20.0:
        sa_budget = min(30.0, _time_left() - 10.0)
        sa_centers, sa_r = _simulated_annealing(best_centers, iu, time_budget=sa_budget)
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_centers.copy()
    
    # Perturb and re-optimize
    if best_centers is not None and _time_left() > 15.0:
        pr_budget = min(20.0, _time_left() - 5.0)
        pr_centers, pr_r = _perturb_and_reoptimize(best_centers, iu, time_budget=pr_budget)
        if pr_r > best_radius:
            best_radius = pr_r
            best_centers = pr_centers.copy()
    
    # Final polish
    if best_centers is not None and _time_left() > 10.0:
        final_budget = _time_left() - 2.0
        polished_centers, polished_r = _polish_l_bfgs(best_centers, time_budget=final_budget, iu=iu)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()
    
    if best_centers is None:
        best_centers = _init_sobol()
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

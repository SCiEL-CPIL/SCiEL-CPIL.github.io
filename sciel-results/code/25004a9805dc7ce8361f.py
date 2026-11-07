# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _random_sphere_start(n, rng):
    """Generate n random points in a sphere with hard-core exclusion."""
    # Estimate radius based on number density
    # BCC density is 2 atoms / a^3. Volume ~ n/2. R ~ (3n/8pi)^(1/3)
    # SC clusters are slightly less dense, so scale up a bit.
    R = 0.8 * n ** (1.0 / 3.0)
    
    points = []
    attempts = 0
    max_attempts = 10000 + 100 * n
    
    while len(points) < n and attempts < max_attempts:
        # Sample from cube and reject if outside sphere
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            attempts += 1
            continue
        
        # Check distance to existing points
        if points:
            pts_arr = np.array(points)
            dists = np.linalg.norm(pts_arr - p, axis=1)
            if dists.min() < 0.35:
                attempts += 1
                continue
        
        points.append(p)
        attempts += 1
        
    if len(points) < n:
        # Fallback: grid
        # This is a very rough fallback, should rarely trigger
        side = int(np.ceil(n ** (1/3))) + 2
        grid = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    grid.append([i*0.4 - side*0.2, j*0.4 - side*0.2, k*0.4 - side*0.2])
        # Shuffle and take n
        rng.shuffle(grid)
        points = grid[:n]
        
    return np.array(points, dtype=float)

def _bcc_start(n, rng, noise_amp=0.05):
    """Generate a BCC-like start."""
    # Simple BCC packing
    # Estimate box size
    side = (n / 2.0) ** (1.0 / 3.0) * 1.1
    half_side = side / 2.0
    n_cells = int(np.ceil(side))
    
    points = []
    # Generate corners and centers
    for i in range(-n_cells//2, n_cells//2 + 1):
        for j in range(-n_cells//2, n_cells//2 + 1):
            for k in range(-n_cells//2, n_cells//2 + 1):
                # Corner
                p_c = np.array([i, j, k])
                # Center
                p_m = np.array([i + 0.5, j + 0.5, k + 0.5])
                
                # Filter by distance from origin to keep it compact
                if np.linalg.norm(p_c) < half_side * 1.2:
                    points.append(p_c)
                if np.linalg.norm(p_m) < half_side * 1.2:
                    points.append(p_m)
                
                if len(points) > n * 2:
                    break
            if len(points) > n * 2:
                break
        if len(points) > n * 2:
            break
            
    if len(points) < n:
        return _random_sphere_start(n, rng)
        
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    
    # Select n closest to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    sel = points[idx]
    
    # Add noise
    sel += rng.normal(0, noise_amp, size=sel.shape)
    
    return sel

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    parameter = kind if kind is not None else instance.get("rho")
    
    start_time = time.time()
    # Evaluator kills at 330s. Be safe with 300s limit.
    deadline = start_time + 295.0
    
    # Deterministic seed based on n and kind
    seed_val = 20240613 ^ (n * 101)
    if kind:
        seed_val ^= hash(kind)
    rng = np.random.default_rng(seed_val)
    
    best_e = np.inf
    best_x = None
    
    def check_validity(x):
        if not np.all(np.isfinite(x)):
            return False
        if n < 2:
            return True
        # Vectorized min distance check
        # For n up to 80, (80, 80, 3) is fine
        diff = x[:, None, :] - x[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(d, np.inf)
        return d.min() > 0.35

    def try_relax(x, tag=''):
        nonlocal best_e, best_x
        if not np.all(np.isfinite(x)):
            return
        try:
            e, p = relax(np.asarray(x, dtype=float), 'sutton_chen', parameter)
            if np.isfinite(e) and np.all(np.isfinite(p)):
                if check_validity(p):
                    if e < best_e:
                        best_e = e
                        best_x = np.array(p, dtype=float)
        except Exception:
            pass

    # Generate initial diverse starts
    # 1. BCC
    x_bcc = _bcc_start(n, rng, noise_amp=0.05)
    try_relax(x_bcc, 'bcc')
    
    # 2. Random Sphere
    x_rand = _random_sphere_start(n, rng)
    try_relax(x_rand, 'rand')
    
    # Basin Hopping
    # Allocate time based on n
    # n=38, 55, 59, 75, 79, 80
    # Smaller n allows more iterations or longer per iteration
    # Let's target a fixed number of successful relaxations if possible, or time-based.
    
    # Time budget for BH: 250 seconds
    bh_deadline = start_time + 250.0
    
    # Step size schedule: Start moderate, increase if stuck, decrease if improving?
    # Standard BH: T decreases.
    # Let's use a simple scheme:
    # T starts at 0.05 * n (arbitrary scaling)
    # Actually, energy scales with n. So T should scale with n.
    # E ~ -n * E_pair.
    # Let's set T = 0.1 * n initially.
    
    T = 0.1 * n
    # Cooling
    cooling_rate = 0.99
    
    # Perturbation magnitude
    # Standard: Gaussian noise on positions.
    # sigma ~ 0.1 - 0.5
    
    sigma = 0.3
    
    current_x = best_x.copy() if best_x is not None else x_rand
    current_e = best_e if best_x is not None else 0.0 # If best_x is None, we haven't relaxed yet?
    # Wait, try_relax was called. If it failed, best_x is None.
    if best_x is None:
        # Fallback if relaxation failed completely
        current_x = x_rand
        # Force one more try?
        try_relax(current_x, 'retry')
        if best_x is None:
            # Return random
            return np.asarray(current_x, dtype=float)
        current_e = best_e
        current_x = best_x.copy()
        
    iters = 0
    while time.time() < bh_deadline:
        iters += 1
        
        # Perturb
        # Random Gaussian noise
        # Also consider random displacement of a random atom (kink)
        # Mix: 50% global noise, 50% local kink
        if rng.random() < 0.5:
            new_x = current_x + rng.normal(0, sigma, size=current_x.shape)
        else:
            new_x = current_x.copy()
            idx = rng.integers(0, n)
            new_x[idx] += rng.normal(0, sigma, size=3)
            
        # Relax
        try:
            e_new, p_new = relax(np.asarray(new_x, dtype=float), 'sutton_chen', parameter)
        except Exception:
            continue
            
        if not np.isfinite(e_new) or not np.all(np.isfinite(p_new)):
            continue
            
        if not check_validity(p_new):
            continue
            
        # Metropolis Accept
        if e_new < current_e:
            # Accept
            current_e = e_new
            current_x = p_new
            # Update global best
            if e_new < best_e:
                best_e = e_new
                best_x = p_new.copy()
                # Reduce noise slightly on improvement? Or keep constant?
                # Standard BH keeps noise constant or increases it if stuck.
                # Let's keep sigma constant for simplicity, or decay it.
                sigma *= 0.995 # Slow decay
                if sigma < 0.05:
                    sigma = 0.05
        else:
            # Metropolis
            delta_e = e_new - current_e
            # Check T > 0
            if T > 1e-6:
                # Avoid overflow
                if delta_e < 1000 * T: 
                     accept_prob = np.exp(-delta_e / T)
                else:
                     accept_prob = 0.0
                
                if rng.random() < accept_prob:
                    current_e = e_new
                    current_x = p_new
                    # Do not update best_x if not better
            
        # Cool
        T *= cooling_rate
        
        # If T is too low, we might be stuck in local minimum.
        # Restart with higher T if no improvement for a while?
        # For now, simple cooling.
        
    # Final check
    if best_x is None:
        # Should not happen if try_relax succeeded at least once
        return np.asarray(current_x, dtype=float)
        
    return np.asarray(best_x, dtype=float)
# EVOLVE-BLOCK-END

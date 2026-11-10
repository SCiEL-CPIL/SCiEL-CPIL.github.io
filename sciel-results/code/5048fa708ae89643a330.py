import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

TOTAL_DEADLINE = 295.0
START_TIME = None

def _energy_and_grad(pos, rho):
    """
    Compute Morse energy and gradient for a cluster.
    V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    """
    n = pos.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Vectorized pairwise differences and distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero
    mask = d < 1e-6
    d[mask] = 1e-6
    
    x = rho * (1.0 - d)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    
    # Energy is half the sum because V_ij is counted twice in the matrix
    energy = 0.5 * np.sum(v)
    
    # Gradient calculation
    # dV/dr = -rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1)
    dvdr = -rho * ex * (ex - 1.0)
    
    # Set diagonal to 0 to avoid self-interaction issues
    np.fill_diagonal(dvdr, 0.0)
    
    # grad_i = sum_j (dV/dr_ij) * (r_i - r_j) / r_ij
    # coeff[i,j] = dvdr[i,j] / d[i,j]
    coeff = dvdr / d
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    
    return energy, grad

def _local_minimize(pos, rho, max_iter=50):
    """
    Perform local minimization using L-BFGS-B with exact gradients.
    Returns (energy, positions).
    """
    def fun(x):
        p = x.reshape(-1, 3)
        e, g = _energy_and_grad(p, rho)
        return e, g.flatten()
    
    try:
        res = minimize(
            fun, pos.flatten(), jac=True, method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-8, 'maxfun': 1000}
        )
        new_pos = res.x.reshape(-1, 3)
        new_e, _ = _energy_and_grad(new_pos, rho)
        return new_e, new_pos
    except:
        return _energy_and_grad(pos, rho)[0], pos

def _greedy_growth(n, rho, start_n=4, start_pos=None):
    """
    Build cluster atom-by-atom by finding the best insertion position.
    """
    if start_pos is None:
        # Start with a small stable cluster, e.g., tetrahedron
        if start_n == 4:
            phi = (1 + np.sqrt(5)) / 2
            start_pos = np.array([
                [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
            ])
            start_pos /= np.linalg.norm(start_pos[0]) # Normalize to unit distance
            # Actually, let's just use a simple seed
            start_pos = np.array([
                [1, 0, 0], [-0.5, np.sqrt(3)/2, 0], [-0.5, -np.sqrt(3)/2, 0], [0, 0, 1]
            ])
            # Relax the seed
            _, start_pos = _local_minimize(start_pos, rho, max_iter=20)
        
    pos = start_pos.copy()
    current_n = pos.shape[0]
    
    # If we need to shrink (shouldn't happen in standard growth, but for safety)
    if n < current_n:
        # Just return the first n atoms (crude, but shouldn't be used if n > start_n)
        pos = pos[:n]
        _, pos = _local_minimize(pos, rho, max_iter=10)
        return pos
        
    while current_n < n:
        # 1. Define candidate positions for the new atom
        # Use a grid of points around the existing cluster
        center = np.mean(pos, axis=0)
        radius = np.max(np.linalg.norm(pos - center, axis=1))
        
        # Search in a shell around the cluster
        # Generate points on a sphere using Fibonacci spiral
        num_candidates = 200
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        candidates = []
        
        # Search multiple radii
        for r_mult in [0.8, 1.0, 1.2, 1.5]:
            r_shell = (radius * 0.5 + 1.0) * r_mult
            for i in range(num_candidates // 4):
                t = (i + 0.5) / (num_candidates // 4)
                z = 1.0 - 2.0 * t
                radius_xy = np.sqrt(max(0, 1.0 - z * z))
                theta = golden_angle * i
                x = radius_xy * np.cos(theta)
                y = radius_xy * np.sin(theta)
                candidates.append(center + r_shell * np.array([x, y, z]))
                
        candidates = np.array(candidates)
        
        # 2. Evaluate energy for each candidate
        best_energy = np.inf
        best_candidate = None
        
        # Batch evaluate to save time? No, n is small enough, but let's be efficient.
        # We can add the candidate to pos and compute energy.
        # Since n is small, computing energy for 200 candidates is fast.
        
        for c in candidates:
            trial_pos = np.vstack([pos, c])
            e, _ = _energy_and_grad(trial_pos, rho)
            if e < best_energy:
                best_energy = e
                best_candidate = c
        
        # 3. Add the best candidate
        pos = np.vstack([pos, best_candidate])
        current_n += 1
        
        # 4. Relax the whole cluster
        _, pos = _local_minimize(pos, rho, max_iter=20)
        
    return pos

def _basin_hopping(pos, rho, time_limit, rng):
    """
    Perform basin hopping starting from a given configuration.
    """
    start_time = time.time()
    best_e, best_pos = _local_minimize(pos, rho, max_iter=50)
    
    # Temperature schedule for moves
    T = 1.0
    
    while time.time() - start_time < time_limit:
        # Generate a perturbed configuration
        # Move a random subset of atoms
        n = pos.shape[0]
        n_moved = max(1, int(n * 0.1))
        idx = rng.choice(n, n_moved, replace=False)
        
        # Gaussian perturbation
        sigma = 0.5
        perturb = np.zeros_like(pos)
        perturb[idx] = rng.normal(0, sigma, (n_moved, 3))
        
        new_pos = best_pos + perturb
        
        # Local minimize
        new_e, new_pos = _local_minimize(new_pos, rho, max_iter=30)
        
        # Metropolis acceptance
        delta_e = new_e - best_e
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            best_e = new_e
            best_pos = new_pos
            
        # Cool down
        T = max(0.05, T * 0.95)
        
        # Check time
        if time.time() - start_time > time_limit - 2.0:
            break
            
    return best_e, best_pos

def construct(instance):
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
        
    n = instance["n"]
    rho = instance["rho"]
    
    elapsed = time.time() - START_TIME
    remaining = TOTAL_DEADLINE - elapsed
    
    # Safety check
    if remaining < 5.0:
        # Fallback: Return a simple valid configuration
        # Grid packing
        side = int(np.ceil(n ** (1/3)))
        pos = []
        count = 0
        for i in range(-side, side+1):
            for j in range(-side, side+1):
                for k in range(-side, side+1):
                    if count < n:
                        pos.append([i, j, k])
                        count += 1
                    else:
                        break
                if count >= n: break
            if count >= n: break
        return np.array(pos[:n])
    
    rng = np.random.default_rng(42 + n) # Seed based on n for reproducibility per instance
    
    best_e = np.inf
    best_pos = None
    
    # Time budget per instance
    # Assume 6 instances, so ~1/6 of total time
    # But we should be dynamic.
    # Let's allocate 40% of remaining time for growth, 60% for basin hopping
    growth_time = min(remaining * 0.3, 20.0)
    hopping_time = remaining - growth_time - 5.0 # 5s buffer
    
    # Phase 1: Greedy Growth
    if n > 4:
        try:
            # Use a random seed for growth to diversify? 
            # Greedy growth is deterministic if seed is fixed. 
            # Let's try a few different initial seeds or growth paths.
            # Actually, greedy growth is very fast and robust.
            
            # Initial seed
            init_seed = np.random.default_rng(123)
            seed_pos = None
            if n <= 10:
                 # For small n, just use the optimal small clusters
                 if n == 1:
                     return np.array([[0,0,0]])
                 if n == 2:
                     return np.array([[0.5, 0, 0], [-0.5, 0, 0]])
            
            # Generate a diverse set of initial structures via greedy growth
            # with slightly different random perturbations during growth?
            # The current _greedy_growth is deterministic.
            # To get diversity, we can perturb the "best candidate" selection slightly
            # or use different initial seeds.
            
            # Let's run greedy growth once.
            pos_grown = _greedy_growth(n, rho)
            
            e_grown, pos_grown = _local_minimize(pos_grown, rho, max_iter=50)
            
            if e_grown < best_e:
                best_e = e_grown
                best_pos = pos_grown
                
        except Exception as e:
            print(f"Growth failed: {e}")
    
    # Phase 2: Basin Hopping
    # Run multiple basin hopping chains
    # Time available for each chain
    num_chains = 3 if n < 50 else 2
    
    chain_time = hopping_time / num_chains
    
    for i in range(num_chains):
        if time.time() - START_TIME > TOTAL_DEADLINE - 5.0:
            break
            
        # Start from best found so far, or a new growth if we have time
        if i == 0:
            start_pos = best_pos.copy()
        else:
            # Generate a new random compact cluster or perturb best
            if rng.random() < 0.5 and best_pos is not None:
                # Perturb best
                perturb_scale = 0.3
                start_pos = best_pos + rng.normal(0, perturb_scale, best_pos.shape)
            else:
                # New random compact
                # Simple grid + noise
                side = int(np.ceil(n ** (1/3)))
                grid = []
                for i in range(-side, side+1):
                    for j in range(-side, side+1):
                        for k in range(-side, side+1):
                            grid.append([i, j, k])
                grid = np.array(grid[:n])
                # Normalize scale
                grid = grid / np.max(np.linalg.norm(grid, axis=1)) * 1.5
                start_pos = grid + rng.normal(0, 0.1, grid.shape)
                
        # Run basin hopping
        e_bh, pos_bh = _basin_hopping(start_pos, rho, chain_time, rng)
        
        if e_bh < best_e:
            best_e = e_bh
            best_pos = pos_bh
    
    if best_pos is None:
        # Final fallback
        return np.zeros((n, 3))
        
    return best_pos


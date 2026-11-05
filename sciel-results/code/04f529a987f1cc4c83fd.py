"""Program for Dzugutov cluster global minima using Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants for Dzugutov potential
A_CONST = 5.82
B_CONST = 1.28
a_CONST = 1.87
b_CONST = 1.94
c_CONST = 1.10
d_CONST = 0.27
m_CONST = 16

def _dzugutov_energy_vectorized(pos, A=A_CONST, B=B_CONST, a=a_CONST, b=b_CONST, c=c_CONST, d=d_CONST, m=m_CONST):
    """
    Compute total Dzugutov potential energy using vectorized operations.
    pos: (n, 3) array
    """
    n = len(pos)
    if n < 2:
        return 0.0
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist = dist[mask]
    
    if len(dist) == 0:
        return 0.0
        
    dist_safe = np.maximum(dist, 1e-6)
    
    energy = 0.0
    mask_a = dist_safe < a
    if np.any(mask_a):
        r_a = dist_safe[mask_a]
        v_a = A * (r_a**(-m) - B) * np.exp(c / (r_a - a))
        energy += np.sum(v_a)
        
    mask_b = (dist_safe >= a) & (dist_safe < b)
    if np.any(mask_b):
        r_b = dist_safe[mask_b]
        v_b = B * np.exp(d / (r_b - b))
        energy += np.sum(v_b)
        
    return energy

def _generate_random_compact(n, seed):
    """Generate a random compact configuration of n atoms."""
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n / (4/3 * np.pi)) ** (1/3)
    
    pos = np.zeros((n, 3))
    pos[0] = [0, 0, 0]
    
    min_dist = 0.5
    for i in range(1, n):
        for attempt in range(100):
            r = radius * rng.random() ** (1/3)
            theta = rng.random() * 2 * np.pi
            phi = np.arccos(2 * rng.random() - 1)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            p = np.array([x, y, z])
            
            dists = np.linalg.norm(pos[:i] - p, axis=1)
            if np.min(dists) >= min_dist:
                pos[i] = p
                break
        else:
            # If failed, place randomly and hope for the best, or expand radius
            p = rng.normal(0, radius, 3)
            pos[i] = p
                 
    return pos

def _basin_hopping(n, seed, max_iter=20, T_start=1.0, T_end=0.1, kick_scale_start=0.5, kick_scale_end=0.1, time_limit=10.0):
    """
    Run Basin Hopping for a cluster of size n.
    Returns (energy, positions).
    """
    start_time = time.time()
    pos = _generate_random_compact(n, seed)
    
    # Initial relaxation
    try:
        e_curr, pos_curr = relax(pos, 'dzugutov', None)
    except Exception:
        return np.inf, pos
    
    e_best = e_curr
    pos_best = pos_curr.copy()
    
    T = T_start
    kick_scale = kick_scale_start
    n_iter = max_iter
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        T = T_start * (T_end / T_start) ** (i / n_iter)
        kick_scale = kick_scale_start * (kick_scale_end / kick_scale_start) ** (i / n_iter)
        
        rng = np.random.default_rng(seed + i)
        kick = rng.normal(0, kick_scale, pos_curr.shape)
        pos_kicked = pos_curr + kick
        
        # Quick check for overlaps
        # If any distance < 0.5, skip relax
        # We can use a fast distance check
        # For simplicity, we'll just try relax. If it fails or is too slow, we'll rely on time limit.
        
        try:
            e_new, pos_new = relax(pos_kicked, 'dzugutov', None)
        except Exception:
            continue
            
        delta_e = e_new - e_curr
        if delta_e < 0:
            e_curr = e_new
            pos_curr = pos_new
            if e_new < e_best:
                e_best = e_new
                pos_best = pos_new.copy()
        else:
            if rng.random() < np.exp(-delta_e / T):
                e_curr = e_new
                pos_curr = pos_new
                
    return e_best, pos_best

def construct(instance):
    """Construct low-energy Dzugutov cluster using Basin Hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 45.0
    
    n_restarts = 4
    seeds = [size * 100 + i for i in range(n_restarts)]
    
    best_energy = np.inf
    best_pos = None
    
    with ThreadPoolExecutor(max_workers=n_restarts) as executor:
        futures = {}
        for seed in seeds:
            # Reduce max_iter if time is tight? 
            # Let's allocate time per restart
            time_per_restart = time_limit / n_restarts
            future = executor.submit(_basin_hopping, size, seed, max_iter=20, time_limit=time_per_restart)
            futures[future] = seed
        
        for future in as_completed(futures):
            seed = futures[future]
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
            except Exception as ex:
                pass
            
            if time.time() - start_time > time_limit * 0.9:
                break

    if best_pos is None:
        pos = _generate_random_compact(size, 0)
        try:
            _, best_pos = relax(pos, potential, parameter)
        except:
            best_pos = pos
            
    return best_pos
# EVOLVE-BLOCK-END

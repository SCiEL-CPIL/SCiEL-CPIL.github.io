"""
Strategy: Parallel Basin Hopping with L-BFGS-B.
Instead of Simulated Annealing, we use Basin Hopping:
1. Generate diverse initial structures (icosahedral, random, etc.).
2. Use scipy.optimize.minimize (L-BFGS-B) to rapidly converge to local minima.
3. Apply random kicks to escape local minima.
4. Re-minimize and accept if energy improves.
5. Run 4 independent basin hopping chains in parallel using ThreadPoolExecutor
   to exploit the 4-thread budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax

# Dzugutov Potential Parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0

def _dzugutov_energy_and_grad(pos):
    """
    Compute energy and gradient of the Dzugutov potential.
    pos: (n, 3) array
    Returns: energy (float), grad (n, 3) array
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask to avoid division by zero and self-interaction
    mask = ~np.eye(n, dtype=bool)
    dist_safe = np.where(mask, dist, 1.0)
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    V = np.zeros((n, n))
    dV = np.zeros((n, n))
    
    # Region 1: r < a
    mask_a = mask & (dist_safe < a)
    r_a = dist_safe[mask_a]
    if len(r_a) > 0:
        term1 = A * (r_a ** (-m) - B)
        term2 = np.exp(c / (r_a - a))
        V_vals = term1 * term2
        
        dterm1 = -m * (r_a ** (-m - 1))
        dterm2 = -c / (r_a - a)**2
        dV_vals = (dterm1 * term2 + term1 * term2 * dterm2)
        
        V[mask_a] = V_vals
        dV[mask_a] = dV_vals
        
    # Region 2: a <= r < b
    mask_b = mask & (dist_safe >= a) & (dist_safe < b)
    r_b = dist_safe[mask_b]
    if len(r_b) > 0:
        V_vals_b = B * np.exp(d / (r_b - b))
        dV_vals_b = V_vals_b * (-d / (r_b - b)**2)
        
        V[mask_b] = V_vals_b
        dV[mask_b] = dV_vals_b
        
    energy = 0.5 * np.sum(V)
    
    factor = dV / dist_safe
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _generate_icosahedron_shell(n_atoms, center=np.zeros(3), radius=1.0):
    """
    Generate points on an icosahedron surface and fill inside.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [center]
    r_shell = radius * (n_atoms / 12.0) ** (1.0/3.0)
    
    for v in verts:
        points.append(center + r_shell * v)
    
    rng = np.random.default_rng(42)
    while len(points) < n_atoms:
        p = rng.uniform(-r_shell, r_shell, 3)
        p = center + p
        dists = np.linalg.norm(np.array(points) - p, axis=1)
        if np.min(dists) > 0.5:
            points.append(p)
            
    return np.array(points[:n_atoms])

def _local_minimize(pos, deadline):
    """
    Perform local minimization using L-BFGS-B.
    """
    # Flatten for scipy
    x0 = pos.flatten()
    
    def fun(x):
        p = x.reshape(-1, 3)
        e, g = _dzugutov_energy_and_grad(p)
        return e, g.flatten()
        
    # Check time
    if time.time() > deadline:
        return pos, _dzugutov_energy_and_grad(pos)[0]
        
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True, options={'maxiter': 50, 'gtol': 1e-5})
        new_pos = res.x.reshape(-1, 3)
        new_e = res.fun
        return new_pos, new_e
    except Exception:
        return pos, _dzugutov_energy_and_grad(pos)[0]

def _basin_hopping_chain(n, seed, deadline):
    """
    Run a single basin hopping chain.
    """
    rng = np.random.default_rng(seed)
    
    # Generate initial structure
    # Mix of icosahedral and random
    if seed % 2 == 0:
        pos = _generate_icosahedron_shell(n, radius=1.0)
    else:
        pos = rng.normal(0, 1.0, (n, 3))
        
    # Ensure valid start
    # Simple check
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # Jitter until valid or give up
        for _ in range(10):
            pos += rng.normal(0, 0.1, pos.shape)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                break
                
    e, _ = _dzugutov_energy_and_grad(pos)
    best_pos = pos.copy()
    best_e = e
    
    kick_size = 0.5
    
    while time.time() < deadline:
        # Kick
        pos_kicked = pos + rng.normal(0, kick_size, pos.shape)
        
        # Check validity of kicked pos
        diff = pos_kicked[:, np.newaxis, :] - pos_kicked[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            continue # Skip invalid kick
            
        # Minimize
        pos_min, e_min = _local_minimize(pos_kicked, deadline)
        
        # Check validity of minimized pos
        diff = pos_min[:, np.newaxis, :] - pos_min[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            continue
            
        if e_min < e:
            pos = pos_min
            e = e_min
            if e_min < best_e:
                best_e = e_min
                best_pos = pos_min.copy()
            kick_size = max(0.1, kick_size * 0.95) # Reduce kick on success
        else:
            kick_size = min(1.5, kick_size * 1.05) # Increase kick on failure
            
    return best_pos, best_e

def construct(instance):
    """
    Main entry point.
    """
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    # We have 4 threads. Let's run 4 basin hopping chains in parallel.
    # Each chain gets its own seed.
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(4):
            seed = 1000 + i
            futures.append(executor.submit(_basin_hopping_chain, n, seed, deadline))
            
        for future in futures:
            try:
                pos, e = future.result()
                results.append((e, pos))
            except Exception:
                pass
                
    if not results:
        # Fallback
        pos = _generate_icosahedron_shell(n, radius=1.0)
        # Try to relax it
        try:
            e, p = relax(pos, 'dzugutov')
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                return p
        except:
            pass
        return pos
        
    # Find the best result
    results.sort(key=lambda x: x[0])
    best_e, best_pos = results[0]
    
    # Final polish with relax if time permits
    if time.time() < deadline - 10:
        try:
            e_polished, pos_polished = relax(best_pos, 'dzugutov')
            diff = pos_polished[:, np.newaxis, :] - pos_polished[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5 and e_polished < best_e:
                return pos_polished
        except:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END

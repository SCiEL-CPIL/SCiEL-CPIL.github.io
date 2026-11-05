"""
Strategy: Basin Hopping with Gradient-Based Local Minimization.
The Dzugutov potential landscape is rugged with many local minima.
Simulated Annealing (current strategy) uses small steps and temperature, 
which can be inefficient in high dimensions. Basin Hopping (Wales, 2002) 
is specifically designed for such landscapes:
1. Locally minimize the structure using a fast gradient-based method (L-BFGS-B).
2. Apply a random displacement (kick) of a certain magnitude.
3. Locally minimize the new structure.
4. Accept the new structure if its energy is lower (Metropolis criterion).
5. Repeat.

This allows the search to "hop" between basins rather than climbing out of 
them via thermal fluctuations, which is more effective for finding global 
minima in cluster potentials.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
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
    
    # Sum energy: sum over i<j
    energy = 0.5 * np.sum(V)
    
    # Gradient calculation
    factor = dV / dist_safe
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _generate_icosahedron_shell(n_atoms, radius=1.0):
    """
    Generate a seed structure based on icosahedral geometry.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    r_shell = radius * (n_atoms / 12.0) ** (1.0/3.0)
    
    for v in verts:
        points.append(r_shell * v)
    
    rng = np.random.default_rng(42)
    while len(points) < n_atoms:
        p = rng.uniform(-r_shell, r_shell, 3)
        dists = np.linalg.norm(np.array(points) - p, axis=1)
        if np.min(dists) > 0.5:
            points.append(p)
            
    return np.array(points[:n_atoms])

def _local_minimize(pos):
    """
    Perform local minimization using L-BFGS-B.
    Returns (energy, positions).
    """
    def func(x):
        p = x.reshape(-1, 3)
        e, g = _dzugutov_energy_and_grad(p)
        return e, g.flatten()
    
    # Flatten positions
    x0 = pos.flatten()
    
    try:
        res = minimize(func, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        pos_min = res.x.reshape(-1, 3)
        e_min = res.fun
        return e_min, pos_min
    except Exception:
        return _dzugutov_energy_and_grad(pos)[0], pos

def _basin_hopping(pos_init, n_iter, kick_mag, T, seed=0):
    """
    Perform Basin Hopping optimization.
    """
    rng = np.random.default_rng(seed)
    current_e, current_pos = _local_minimize(pos_init)
    best_e = current_e
    best_pos = current_pos.copy()
    
    for i in range(n_iter):
        # Apply kick
        noise = rng.normal(0, kick_mag, pos_init.shape)
        new_pos = current_pos + noise
        
        # Simple validity check before expensive minimization
        # If min dist < 0.5, it's invalid. We could try to repair, but rejection is simpler.
        # However, L-BFGS-B might fix it if the gradient pushes them apart.
        # But if they are very close, it might fail.
        # Let's just let L-BFGS-B try. If it fails, we reject.
        
        new_e, new_pos_min = _local_minimize(new_pos)
        
        # Check validity of the minimized structure
        # The evaluator requires min dist >= 0.5.
        # L-BFGS-B minimizes energy. The Dzugutov potential has strong repulsion 
        # for r < a=1.87. So it should naturally push atoms apart.
        # But we must ensure the final output is valid.
        
        if np.isfinite(new_e):
            # Metropolis acceptance
            if new_e < current_e or rng.random() < np.exp(-(new_e - current_e) / T):
                current_e = new_e
                current_pos = new_pos_min
                
                if new_e < best_e:
                    best_e = new_e
                    best_pos = new_pos_min.copy()
                    
    return best_e, best_pos

def construct(instance):
    """
    Main entry point.
    """
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin (300s limit)
    
    # Generate diverse initial structures
    candidates = []
    
    # 1. Icosahedral seed
    try:
        ico_pos = _generate_icosahedron_shell(n, radius=1.0)
        # Quick relax to ensure validity and get a good starting point
        e, p = relax(ico_pos, 'dzugutov')
        candidates.append(p)
    except:
        pass
        
    # 2. Random seeds
    for _ in range(2):
        rng = np.random.default_rng()
        pos = rng.uniform(-1, 1, (n, 3))
        pos /= np.linalg.norm(pos, axis=1, keepdims=True)
        pos *= (n ** (1/3)) * 0.8
        try:
            e, p = relax(pos, 'dzugutov')
            candidates.append(p)
        except:
            pass
            
    if not candidates:
        # Fallback
        pos = np.zeros((n, 3))
        side = int(np.ceil(n ** (1/3)))
        k = 0
        for i in range(side):
            for j in range(side):
                for l in range(side):
                    if k < n:
                        pos[k] = [i, j, l] * 0.6
                        k += 1
        return pos[:n]

    best_overall_energy = np.inf
    best_overall_pos = candidates[0]
    
    # Run Basin Hopping on each candidate
    # Allocate time evenly
    n_candidates = len(candidates)
    time_per_candidate = (deadline - time.time()) / n_candidates
    
    for idx, pos_init in enumerate(candidates):
        t_start_cand = time.time()
        
        # Parameters for Basin Hopping
        # Kick magnitude: roughly the interatomic distance scale
        kick_mag = 0.5
        T = 0.1 # Temperature for Metropolis acceptance
        
        # Estimate steps
        # One BH step involves a minimization (50 iters L-BFGS-B)
        # L-BFGS-B on 60 atoms is fast, maybe 1-5ms per step?
        # Let's run as many steps as possible within time_per_candidate
        n_steps = 50 # Initial guess
        t_test_start = time.time()
        # Test one step to estimate cost
        _local_minimize(pos_init)
        t_step = time.time() - t_test_start
        
        if t_step < 0.001: t_step = 0.001 # Avoid division by zero
        max_steps = int(time_per_candidate / (t_step * 1.5)) # 1.5x safety factor
        max_steps = max(10, min(max_steps, 500)) # Cap steps to avoid infinite loop
        
        best_e, best_pos = _basin_hopping(pos_init, max_steps, kick_mag, T, seed=idx)
        
        # Final polish with the provided relax function (which might be a more robust optimizer)
        try:
            e_polished, pos_polished = relax(best_pos, 'dzugutov')
            # Check validity
            diff = pos_polished[:, np.newaxis, :] - pos_polished[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                if e_polished < best_e:
                    best_e = e_polished
                    best_pos = pos_polished
        except:
            pass
            
        if best_e < best_overall_energy:
            best_overall_energy = best_e
            best_overall_pos = best_pos
            
        if time.time() > deadline:
            break
            
    return best_overall_pos
# EVOLVE-BLOCK-END

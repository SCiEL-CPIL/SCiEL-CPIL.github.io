"""
Strategy: Simulated Annealing with Poly-tetrahedral Seeding.
The Dzugutov potential favors icosahedral/polytetrahedral order.
We generate initial structures based on icosahedral shells and random
perturbations, then apply Simulated Annealing to escape local minima.
We use vectorized NumPy operations for speed to maximize search depth
within the 300s limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
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
    
    # Only consider pairs i < j to avoid double counting and self-interaction
    # We can use the full matrix but divide by 2 at the end, or use triu indices.
    # Using triu indices is more memory efficient for large n? No, n is small (38-60).
    # Let's use the full matrix approach for simplicity, then mask out i=j.
    
    V = np.zeros((n, n))
    dV = np.zeros((n, n))
    
    # Region 1: r < a
    mask_a = mask & (dist_safe < a)
    r_a = dist_safe[mask_a]
    # V(r) = A * (r^-m - B) * exp(c / (r - a))
    term1 = A * (r_a ** (-m) - B)
    term2 = np.exp(c / (r_a - a))
    V_vals = term1 * term2
    
    # Gradient dV/dr
    # d/dr [ (r^-m - B) * exp(c/(r-a)) ]
    # = (-m * r^(-m-1)) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c / (r-a)^2)
    dterm1 = -m * (r_a ** (-m - 1))
    dterm2 = -c / (r_a - a)**2
    dV_vals = (dterm1 * term2 + term1 * term2 * dterm2)
    
    # Assign to V and dV matrices
    V[mask_a] = V_vals
    dV[mask_a] = dV_vals
    
    # Region 2: a <= r < b
    mask_b = mask & (dist_safe >= a) & (dist_safe < b)
    r_b = dist_safe[mask_b]
    # V(r) = B * exp(d / (r - b))
    V_vals_b = B * np.exp(d / (r_b - b))
    
    # Gradient dV/dr
    # d/dr [ B * exp(d/(r-b)) ] = B * exp(d/(r-b)) * (-d / (r-b)^2)
    dV_vals_b = V_vals_b * (-d / (r_b - b)**2)
    
    V[mask_b] = V_vals_b
    dV[mask_b] = dV_vals_b
    
    # Sum energy: sum over i<j
    # V is symmetric. Sum of upper triangle.
    energy = 0.5 * np.sum(V)
    
    # Gradient calculation:
    # F_i = sum_{j != i} dV/dr_ij * (r_i - r_j) / r_ij
    # dV/dr_ij is in dV[i,j].
    # Let's compute the force vector for each i.
    # diff[i,j] = pos[i] - pos[j]
    # grad[i] = sum_j ( dV[i,j] / dist[i,j] ) * diff[i,j]
    
    # dV is zero where mask is false (including i=j).
    # We need to handle division by zero in diff/dist.
    # dist_safe is 1.0 where mask is false.
    # So dV/dist_safe is 0 where mask is false.
    
    factor = dV / dist_safe
    # factor[i,j] is the scalar multiplier for diff[i,j]
    
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _generate_icosahedron_shell(n_atoms, center=np.zeros(3), radius=1.0):
    """
    Generate points on an icosahedron surface and fill inside.
    This is a rough heuristic for polytetrahedral packing.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [center]
    # Place atoms on the icosahedron vertices first
    # Then fill remaining atoms randomly within the sphere, enforcing min distance
    rng = np.random.default_rng(42)
    
    # Scale icosahedron to fit n_atoms roughly
    # Volume ~ n * (4/3) pi r_min^3 ?
    # Just use a reasonable radius
    r_shell = radius * (n_atoms / 12.0) ** (1.0/3.0)
    
    for v in verts:
        points.append(center + r_shell * v)
    
    # Add remaining atoms randomly
    while len(points) < n_atoms:
        p = rng.uniform(-r_shell, r_shell, 3)
        p = center + p
        # Check distance to existing points
        dists = np.linalg.norm(np.array(points) - p, axis=1)
        if np.min(dists) > 0.5: # Min separation constraint
            points.append(p)
            
    return np.array(points[:n_atoms])

def _simulated_annealing(pos, n_steps, T_init, T_final, seed=0):
    """
    Perform Simulated Annealing.
    """
    rng = np.random.default_rng(seed)
    current_energy, current_grad = _dzugutov_energy_and_grad(pos)
    best_energy = current_energy
    best_pos = pos.copy()
    
    # Adaptive step size
    step_size = 0.1
    T = T_init
    
    for i in range(n_steps):
        # Cooling schedule
        T = T_init * (T_final / T_init) ** (i / n_steps)
        
        # Propose move
        # Random perturbation
        noise = rng.normal(0, step_size, pos.shape)
        new_pos = pos + noise
        
        # Check min distance constraint (simple rejection)
        # Efficient check: only check new points against old? No, all pairs.
        # For n=60, 1770 pairs. Fast enough.
        # If any pair < 0.5, reject? Or just let energy be huge?
        # The potential has a hard repulsion for r < a=1.87.
        # But the problem statement says "atoms must be at least 0.5 apart".
        # If we violate this, is it invalid? Yes, "invalid result ... scores 0".
        # So we MUST enforce 0.5 separation.
        
        # Fast rejection: compute min dist
        # Using numpy broadcasting for min dist is O(N^2)
        diff = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            continue # Reject invalid move
        
        new_energy, _ = _dzugutov_energy_and_grad(new_pos)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = pos.copy()
            # Reduce step size if accepting
            step_size *= 0.999
        else:
            # Increase step size slightly if rejecting to explore more?
            step_size *= 1.001
            
        # Keep step size within bounds
        step_size = np.clip(step_size, 0.01, 1.0)
        
    return best_pos, best_energy

def construct(instance):
    """
    Main entry point.
    """
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    # Generate multiple diverse initial guesses
    candidates = []
    
    # 1. Icosahedral seed
    try:
        ico_pos = _generate_icosahedron_shell(n, radius=1.0)
        candidates.append(ico_pos)
    except:
        pass
        
    # 2. Random compact seeds
    for _ in range(3):
        rng = np.random.default_rng()
        pos = rng.uniform(-1, 1, (n, 3))
        # Normalize to roughly same volume
        pos /= np.linalg.norm(pos, axis=1, keepdims=True)
        pos *= (n ** (1/3)) * 0.8
        candidates.append(pos)
        
    # 3. "Polytetrahedral" random: cluster around a central point
    rng = np.random.default_rng(123)
    center = np.zeros(3)
    pos = rng.normal(0, 0.5, (n, 3))
    candidates.append(pos)
    
    best_overall_energy = np.inf
    best_overall_pos = None
    
    # Run SA on each candidate, or just the best few?
    # We have limited time. Let's run SA on the icosahedral one first, 
    # then if time permits, others.
    
    # Sort candidates by initial energy
    scored_candidates = []
    for pos in candidates:
        # Ensure min distance 0.5
        # If not, relax quickly or reject
        # Let's try to relax briefly to get a valid start
        try:
            e, p = relax(pos, 'dzugutov')
            # Check validity of relaxed pos
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                scored_candidates.append((e, p))
        except:
            pass
            
    if not scored_candidates:
        # Fallback: return random valid points if all relaxations failed
        # (Unlikely, but safe)
        pos = np.zeros((n, 3))
        # Grid-like placement
        side = int(np.ceil(n ** (1/3)))
        k = 0
        for i in range(side):
            for j in range(side):
                for l in range(side):
                    if k < n:
                        pos[k] = [i, j, l] * 0.6
                        k += 1
        return pos[:n]
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run SA on the top candidates
    # Budget: ~280s total. 
    # Let's allocate time per candidate.
    # If we have 4 candidates, ~60s each.
    # SA steps: maybe 5000-10000 steps?
    # Let's estimate cost of one step.
    # Energy calc is vectorized, ~O(N^2). N=60 -> 3600 ops. Very fast.
    # Python overhead might be higher.
    
    # Let's do a timed loop
    for e_init, pos_init in scored_candidates:
        if time.time() > deadline:
            break
            
        # Run SA
        # Increase steps if we have time?
        # Start with 2000 steps, see how long it takes.
        n_steps = 2000
        T_init = 0.5
        T_final = 0.01
        
        t_start = time.time()
        pos_sa, e_sa = _simulated_annealing(pos_init, n_steps, T_init, T_final, seed=int(time.time()*1000) % 10000)
        t_elapsed = time.time() - t_start
        
        if e_sa < best_overall_energy:
            best_overall_energy = e_sa
            best_overall_pos = pos_sa
            
        # If we have time, do a second pass with more steps or lower T?
        # Or just move to next candidate.
        # Let's try to do a "polish" with relax on the SA result
        try:
            e_polished, pos_polished = relax(pos_sa, 'dzugutov')
            # Check validity
            diff = pos_polished[:, np.newaxis, :] - pos_polished[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5 and e_polished < best_overall_energy:
                best_overall_energy = e_polished
                best_overall_pos = pos_polished
        except:
            pass

    if best_overall_pos is None:
        # Should not happen if scored_candidates is not empty
        return scored_candidates[0][1]
        
    return best_overall_pos
# EVOLVE-BLOCK-END

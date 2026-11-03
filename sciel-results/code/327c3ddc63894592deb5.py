import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _smooth_objective(params, target_radius, penalty_weight):
    """
    Smooth surrogate objective function.
    We want to maximize the "space" available. 
    A common smooth approximation for packing is to minimize a penalty for overlaps.
    However, to drive circles to be as large as possible, we can use a potential energy 
    that penalizes if distance < 2 * target_radius, but also attracts them?
    
    Actually, a better smooth objective for maximizing min-radius or sum-radius 
    is to minimize:
    E = sum_{i<j} max(0, 2*r_target - dist(i,j))^2 + boundary_penalty
    
    But this keeps circles at 2*r_target. To make them larger, we can iteratively 
    increase r_target, but that's complex.
    
    Alternative: 
    Minimize E = sum_{i<j} ( (2*r_est - dist(i,j))^2 if dist < 2*r_est else 0 )
    This doesn't encourage expansion.
    
    Let's use a "pressure" model. 
    If we fix positions, the max radius is determined by the closest neighbor.
    Let r_i = min( boundary_i, min_j (dist_ij - r_j) ) ... this is coupled.
    
    Let's stick to the standard smooth packing objective:
    Minimize sum of squared violations of the constraint dist(i,j) >= 2*r_min
    where r_min is a variable we might optimize? No, let's optimize positions 
    to maximize the minimum distance to boundaries and neighbors, assuming equal radii 
    first, then refine.
    
    Actually, the most effective smooth objective for circle packing is:
    f(x) = sum_{i<j} max(0, 2*r_0 - ||x_i - x_j||)^2 + sum_i max(0, r_0 - min(x_i, 1-x_i, y_i, 1-y_i))^2
    
    We can run this with a fixed r_0. To maximize sum of radii, we want the circles 
    to be as large as possible. 
    A better strategy:
    1. Find a good layout for N circles of radius r=0.15 (or similar).
    2. Use the LP to get the actual max radii for that layout.
    
    The smooth objective should guide the centers into a dense, non-overlapping 
    configuration that allows for large radii.
    """
    pos = params.reshape(N, 2)
    
    # Vectorized pairwise distances
    # pos: (N, 2)
    # diff: (N, N, 2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # We only care about i < j
    # Mask for i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Smooth penalty for overlap assuming a reference radius r_ref
    # We want circles to be separated by at least 2*r_ref.
    # If they are closer, penalize quadratically.
    # This creates a "repulsive" force.
    overlap = np.maximum(0, 2 * target_radius - d_ij)
    energy_overlap = np.sum(overlap ** 2)
    
    # Boundary penalty
    # Distance to boundary for each circle
    d_bound = np.minimum(np.minimum(pos[:, 0], pos[:, 1]), 
                         1.0 - np.minimum(pos[:, 0], pos[:, 1]))
    # We want d_bound >= target_radius
    bound_viol = np.maximum(0, target_radius - d_bound)
    energy_bound = np.sum(bound_viol ** 2)
    
    # We also want to encourage circles to be "tight" against each other 
    # to maximize potential radius? 
    # The repulsive force above pushes them apart to exactly 2*r_ref.
    # If the box is large enough, they will stop at 2*r_ref.
    # If we increase r_ref, they push harder.
    # So, we can use a fixed r_ref that is slightly larger than the expected 
    # final radius to ensure a dense packing.
    
    total_energy = energy_overlap + penalty_weight * energy_bound
    
    return total_energy

def _get_optimal_radii(pos, n):
    """
    Solve the LP to find optimal radii for fixed positions.
    Maximize sum(r_i)
    Subject to:
      r_i + r_j <= dist(i,j)
      r_i <= boundary_dist(i)
      r_i >= 0
    """
    # Boundary distances
    b = np.minimum(np.minimum(pos[:, 0], pos[:, 1]), 
                   1.0 - np.minimum(pos[:, 0], pos[:, 1]))
    b = np.maximum(b, 0) # Ensure non-negative
    
    # Pairwise distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = pos[i_idx] - pos[j_idx]
    d_ij = np.linalg.norm(diff, axis=1)
    
    # LP Formulation
    # Variables: r_0 ... r_{n-1}
    # Objective: Minimize -sum(r)
    c = -np.ones(n)
    
    # Constraints: r_i + r_j <= d_ij
    A_ub = np.zeros((len(d_ij), n))
    for k in range(len(d_ij)):
        A_ub[k, i_idx[k]] = 1.0
        A_ub[k, j_idx[k]] = 1.0
    b_ub = d_ij
    
    # Bounds: 0 <= r_i <= b_i
    bounds = [(0, b[i]) for i in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            # Fallback: Greedy
            radii = np.zeros(n)
            for _ in range(50):
                changed = False
                for i in range(n):
                    max_r = b[i]
                    for j in range(n):
                        if i != j:
                            dist = np.linalg.norm(pos[i] - pos[j])
                            max_r = min(max_r, dist - radii[j])
                    if max_r > radii[i] + 1e-6:
                        radii[i] = max_r
                        changed = True
                if not changed:
                    break
            return np.sum(radii), radii
    except Exception:
        radii = np.zeros(n)
        return 0.0, radii

def _is_valid(pos, radii, n, tol=1e-6):
    """Check validity of packing."""
    # Boundary
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    # Overlap
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = pos[i_idx] - pos[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    sums = radii[i_idx] + radii[j_idx]
    if np.any(dists < sums - tol):
        return False
    return True

def _simulated_annealing(pos, target_radius, penalty_weight, start_temp=1.0, end_temp=0.001, steps=5000, seed=42):
    """
    Simulated Annealing to find a good configuration.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_energy = _smooth_objective(current_pos.flatten(), target_radius, penalty_weight)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    step_size = 0.05
    total_steps = steps
    
    for i in range(total_steps):
        # Annealing schedule
        temp = start_temp * (end_temp / start_temp) ** (i / total_steps)
        
        # Propose new position: move a random circle
        idx = rng.randint(N)
        delta = rng.uniform(-step_size, step_size, 2)
        
        new_pos = current_pos.copy()
        new_pos[idx] += delta
        # Keep within bounds [0,1]
        new_pos[idx] = np.clip(new_pos[idx], 0, 1)
        
        new_energy = _smooth_objective(new_pos.flatten(), target_radius, penalty_weight)
        
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.rand() < np.exp(-delta_e / (temp + 1e-9)):
            current_pos = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
        # Reduce step size slightly? Or keep constant? 
        # Constant step size with decreasing temp usually works well.
        
    return best_pos, best_energy

def _gradient_polish(pos, target_radius, penalty_weight, time_limit=20):
    """
    Refine positions using gradient-based optimization on the smooth objective.
    """
    start_time = time.time()
    x0 = pos.flatten()
    
    # Define objective and gradient
    def obj(x):
        return _smooth_objective(x, target_radius, penalty_weight)
    
    # Numerical gradient is slow, but L-BFGS-B handles it internally.
    # We can provide a finite difference gradient if needed, but let's let scipy do it.
    
    bounds = [(0, 1)] * (2 * N)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success:
            return res.x.reshape(N, 2)
        else:
            return pos
    except Exception:
        return pos

def _initial_grid(n):
    """Create a grid layout as initial guess."""
    # 5x6 grid = 30, take 26
    rows = 5
    cols = 6
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            positions.append([x, y])
    
    if len(positions) < n:
        rng = np.random.RandomState(42)
        while len(positions) < n:
            positions.append(rng.uniform(0.1, 0.9, 2))
            
    return np.array(positions[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Initial positions
    pos0 = _initial_grid(N)
    
    # We will try a few different target radii for the smooth optimization.
    # The optimal radius for 26 circles is likely around 0.15-0.18.
    # Let's start with 0.15 and maybe try 0.12 for a looser pack, then polish.
    
    target_radii = [0.15, 0.12, 0.18]
    penalty_weights = [10.0, 1.0]
    
    for r_target in target_radii:
        if time.time() > deadline - 30:
            break
            
        for p_weight in penalty_weights:
            if time.time() > deadline - 30:
                break
                
            # 1. Simulated Annealing
            pos_sa, _ = _simulated_annealing(pos0, r_target, p_weight, steps=3000, seed=int(r_target*100))
            
            # 2. Gradient Polish
            pos_polished = _gradient_polish(pos_sa, r_target, p_weight, time_limit=10)
            
            # 3. Calculate exact radii
            sum_r, radii = _get_optimal_radii(pos_polished, N)
            
            # 4. Validate and Fix
            # Check validity
            if _is_valid(pos_polished, radii, N):
                if sum_r > best_score:
                    best_score = sum_r
                    result = np.zeros((N, 3))
                    result[:, :2] = pos_polished
                    result[:, 2] = radii
                    best_result = result
            else:
                # If invalid, try to fix radii by reducing them (conservative)
                # This shouldn't happen if LP was correct, but numerical errors might occur
                radii_fixed = radii * 0.99
                if _is_valid(pos_polished, radii_fixed, N):
                    if np.sum(radii_fixed) > best_score:
                        best_score = np.sum(radii_fixed)
                        result = np.zeros((N, 3))
                        result[:, :2] = pos_polished
                        result[:, 2] = radii_fixed
                        best_result = result

    # Fallback
    if best_result is None:
        pos = _initial_grid(N)
        radii = np.full(N, 0.05)
        # Quick fix
        for i in range(N):
            max_r = min(pos[i,0], pos[i,1], 1-pos[i,0], 1-pos[i,1])
            radii[i] = min(radii[i], max_r)
        best_result = np.zeros((N, 3))
        best_result[:, :2] = pos
        best_result[:, 2] = radii
        best_score = np.sum(radii)
        
    return best_result

# EVOLVE-BLOCK-END

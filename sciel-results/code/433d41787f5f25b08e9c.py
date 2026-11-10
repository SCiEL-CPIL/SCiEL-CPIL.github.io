import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

N = 50
DEADLINE = 140.0  # Safety margin before 150s limit

def compute_radius_fast(positions):
    """Fast computation of maximum common radius using cdist."""
    min_bound = 0.5 - np.max(np.abs(positions))
    if min_bound <= 0:
        return 0.0
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_bound, min_dist / 2.0)

def compute_radius(positions):
    """Compute the maximum common radius for given circle centers."""
    return compute_radius_fast(positions)

def soft_min_radius(x, temperature=0.005):
    """
    Smooth approximation of the min radius using log-sum-exp.
    """
    pos = x.reshape(N, 2)
    
    # Boundary terms
    bound_terms = np.concatenate([
        0.5 - np.abs(pos[:, 0]),
        0.5 - np.abs(pos[:, 1])
    ])
    
    # Pairwise distance terms / 2
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_terms = dists[iu] / 2.0
    
    all_terms = np.concatenate([bound_terms, pair_terms])
    neg_terms = -all_terms / temperature
    soft_min = -temperature * logsumexp(neg_terms)
    
    return soft_min

def generate_hexagonal_initial():
    """Generate a hexagonal packing initialization."""
    positions = []
    sqrt3 = np.sqrt(3.0)
    
    best_pos = None
    best_r = 0
    
    for n_rows in range(5, 10):
        n_cols = int(np.ceil(N / n_rows))
        
        s_x = 1.0 / (n_cols + 0.5)
        s_y = s_x * sqrt3 / 2.0
        
        total_height = (n_rows - 1) * s_y + s_x
        if total_height > 1.0:
            scale = 1.0 / total_height
            s_x *= scale
            s_y *= scale
        
        pos_list = []
        for row in range(n_rows):
            y = -0.5 + s_x/2 + row * s_y
            if y > 0.5 - s_x/2:
                continue
            n_in_row = n_cols if row % 2 == 0 else n_cols - 1
            if row % 2 == 1:
                x_start = -0.5 + s_x
            else:
                x_start = -0.5 + s_x/2
            
            for col in range(n_in_row):
                x = x_start + col * s_x
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pos_list.append([x, y])
                if len(pos_list) >= N:
                    break
            if len(pos_list) >= N:
                break
        
        if len(pos_list) >= N:
            pos_arr = np.array(pos_list[:N])
            r = compute_radius_fast(pos_arr)
            if r > best_r:
                best_r = r
                best_pos = pos_arr.copy()
    
    if best_pos is not None:
        return best_pos
    
    n_side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(n_side):
        for j in range(n_side):
            if len(positions) >= N:
                break
            x = -0.5 + (j + 0.5) * (1.0 / n_side)
            y = -0.5 + (i + 0.5) * (1.0 / n_side)
            positions.append([x, y])
    
    return np.array(positions[:N])

def polish(positions, time_limit=5.0, use_soft=False):
    """Local optimization using scipy minimize to polish the configuration."""
    start_time = time.time()
    positions = positions.copy()
    current_r = compute_radius_fast(positions)
    
    best_pos = positions.copy()
    best_r = current_r
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Phase 1: Use soft-min objective for smoother gradient landscape
    if use_soft:
        def soft_objective(x):
            if time.time() - start_time > time_limit:
                return 0
            pos = x.reshape(N, 2)
            if np.any(np.abs(pos) > 0.5):
                return 0
            return -soft_min_radius(x, temperature=0.003)
        
        try:
            result = minimize(soft_objective, positions.flatten(), method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 200, 'maxls': 30, 'ftol': 1e-10})
            new_pos = result.x.reshape(N, 2)
            if np.all(np.abs(new_pos) <= 0.5):
                new_r = compute_radius_fast(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
        except Exception:
            pass
    
    # Phase 2: Use exact min radius with tighter tolerance
    def exact_objective(x):
        if time.time() - start_time > time_limit:
            return 0
        pos = x.reshape(N, 2)
        if np.any(np.abs(pos) > 0.5):
            return 0
        return -compute_radius_fast(pos)
    
    n_restarts = 2
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            x0 = best_pos.flatten()
        else:
            noise_scale = 0.005
            x0 = best_pos.flatten() + np.random.uniform(-noise_scale, noise_scale, 2*N)
            x0 = np.clip(x0, -0.5, 0.5)
        
        try:
            result = minimize(exact_objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 300, 'maxls': 40, 'ftol': 1e-12})
            new_pos = result.x.reshape(N, 2)
            if np.all(np.abs(new_pos) <= 0.5):
                new_r = compute_radius_fast(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
        except Exception:
            pass
    
    # Phase 3: Very tight final polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        try:
            result = minimize(exact_objective, best_pos.flatten(), method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-14})
            new_pos = result.x.reshape(N, 2)
            if np.all(np.abs(new_pos) <= 0.5):
                new_r = compute_radius_fast(new_pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = new_pos.copy()
        except Exception:
            pass
    
    return best_pos

def simulated_annealing(positions, time_limit=10.0):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    current_pos = positions.copy()
    current_r = compute_radius_fast(current_pos)
    best_pos = current_pos.copy()
    best_r = current_r
    
    T_start = 0.05
    T_end = 0.001
    cooling = 0.995
    T = T_start
    
    while time.time() - start_time < time_limit and T > T_end:
        # Random perturbation
        noise_scale = T * 0.5
        new_pos = current_pos + np.random.normal(0, noise_scale, (N, 2))
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        new_r = compute_radius_fast(new_pos)
        delta = new_r - current_r
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_pos = new_pos
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pos = current_pos.copy()
        
        T *= cooling
    
    return best_pos

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Step 1: Generate initial hexagonal packing
    positions = generate_hexagonal_initial()
    current_r = compute_radius_fast(positions)
    best_pos = positions.copy()
    best_r = current_r
    
    # Step 2: Polish with soft-min then exact local optimization
    positions = polish(positions, time_limit=20.0, use_soft=True)
    current_r = compute_radius_fast(positions)
    if current_r > best_r:
        best_r = current_r
        best_pos = positions.copy()
    
    # Step 3: Simulated annealing to explore different configurations
    remaining_time = deadline - time.time()
    if remaining_time > 20.0:
        sa_pos = simulated_annealing(best_pos, time_limit=min(25.0, remaining_time - 15.0))
        sa_r = compute_radius_fast(sa_pos)
        if sa_r > best_r:
            best_r = sa_r
            best_pos = sa_pos.copy()
        
        # Polish the SA result
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            polished = polish(best_pos, time_limit=min(15.0, remaining_time - 5.0), use_soft=False)
            pol_r = compute_radius_fast(polished)
            if pol_r > best_r:
                best_r = pol_r
                best_pos = polished.copy()
    
    # Step 4: Try a few random perturbations + polish to escape local minima
    remaining_time = deadline - time.time()
    n_perturbations = 3
    for i in range(n_perturbations):
        if time.time() - start_time > deadline - 15.0:
            break
        
        noise_scale = 0.02 / (i + 1)
        perturbed = best_pos.copy()
        perturbed += np.random.uniform(-noise_scale, noise_scale, (N, 2))
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        polish_time = min(10.0, (deadline - time.time()) / (n_perturbations - i))
        if polish_time < 2.0:
            break
        
        result_pos = polish(perturbed, time_limit=polish_time, use_soft=False)
        result_r = compute_radius_fast(result_pos)
        if result_r > best_r:
            best_r = result_r
            best_pos = result_pos.copy()
    
    # Step 5: Final high-precision polish
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        best_pos = polish(best_pos, time_limit=min(remaining_time - 2.0, 15.0), use_soft=False)
        final_r = compute_radius_fast(best_pos)
        if final_r > best_r:
            best_r = final_r
    
    # Ensure valid output
    best_pos = np.clip(best_pos, -0.5, 0.5)
    return best_pos

import numpy as np
import time
import math
from scipy.optimize import least_squares

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair, min_bound)
    return score

def soft_min_value(points, k=100.0):
    """
    Compute the smooth minimum of pairwise distances and boundary distances.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    pair_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    d_min = np.min(all_dists)
    exp_terms = np.exp(-k * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return d_min
    
    soft_min = d_min - (1.0 / k) * np.log(sum_exp)
    return soft_min

def active_set_newton(points, r_target, max_iter=50, tol=1e-10):
    """
    Active-set Newton refinement: identify binding constraints and solve
    the small linear system to push all distances >= r_target.
    """
    points = points.copy()
    n = len(points)
    
    for it in range(max_iter):
        # Compute all pairwise distances and boundary distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        # Find binding constraints (distance < r_target + tol)
        pair_idx = np.triu_indices(n, k=1)
        pair_dists = dists[pair_idx]
        
        # Pairwise binding constraints
        pair_binding = np.where(pair_dists < r_target + tol)[0]
        # Boundary binding constraints
        bound_binding = np.where(bound_dists < r_target + tol)[0]
        
        if len(pair_binding) == 0 and len(bound_binding) == 0:
            break
        
        # Build active constraint system
        # For each binding pair (i,j): (p_i - p_j) · (p_i - p_j) / dist = r_target
        # Gradient w.r.t. p_i: (p_i - p_j) / dist
        # Gradient w.r.t. p_j: -(p_i - p_j) / dist
        # For boundary constraint i: -p_i / |p_i| (to push inward)
        
        # We want to find delta such that all binding constraints are satisfied.
        # Linearize: for pair (i,j) with dist d < r_target:
        #   (p_i - p_j + delta_i - delta_j) · (p_i - p_j) / d >= r_target
        #   => (delta_i - delta_j) · (p_i - p_j) / d >= r_target - d
        # For boundary i with |p_i| > 1 - r_target:
        #   |p_i + delta_i| <= 1 - r_target
        #   => delta_i · p_i / |p_i| <= (1 - r_target) - |p_i|
        
        # Build A matrix and b vector for the active constraints
        # We'll use a simple approach: for each binding constraint, add a row
        # to the system A @ delta_flat = b
        
        A_rows = []
        b_vals = []
        
        # Pairwise constraints
        for idx in pair_binding:
            i, j = pair_idx[0][idx], pair_idx[1][idx]
            d = dists[i, j]
            if d < 1e-12:
                continue
            direction = (points[i] - points[j]) / d
            # (delta_i - delta_j) · direction >= r_target - d
            # Row for i: +direction, for j: -direction
            row = np.zeros(2 * n)
            row[2*i:2*i+2] = direction
            row[2*j:2*j+2] = -direction
            A_rows.append(row)
            b_vals.append(r_target - d)
        
        # Boundary constraints
        for i in bound_binding:
            ri = rad[i]
            if ri < 1e-12:
                continue
            direction = points[i] / ri
            # delta_i · direction <= (1 - r_target) - ri
            # Row: -direction for i (to make it >= form)
            row = np.zeros(2 * n)
            row[2*i:2*i+2] = -direction
            A_rows.append(row)
            b_vals.append(ri - (1 - r_target))
        
        if len(A_rows) == 0:
            break
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        
        # Solve A @ delta = b (least squares if overdetermined)
        try:
            delta, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            break
        
        delta = delta.reshape(n, 2)
        
        # Damping: take a small step to avoid overshooting
        step_size = 0.5
        points = points + step_size * delta
        
        # Project inside container
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
        
        # Check convergence
        current_score = compute_score(points)
        if abs(current_score - r_target) < tol:
            break
    
    return points, compute_score(points)

def simulated_annealing(points, start_temp=0.05, end_temp=0.001, max_iter=5000, time_limit=None):
    """
    Simulated annealing to escape local minima.
    """
    if time_limit is None:
        time_limit = 30.0
    
    start_time = time.time()
    current = points.copy()
    current_score = compute_score(current)
    best = current.copy()
    best_score = current_score
    
    n = len(current)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        # Random perturbation
        move_idx = np.random.randint(n)
        step = np.random.normal(0, temp, size=2)
        new_point = current[move_idx].copy()
        new_point += step
        
        # Keep inside container
        r_new = np.linalg.norm(new_point)
        if r_new > 1.0:
            new_point /= r_new
        
        # Compute new score (only need to check affected distances)
        candidate = current.copy()
        candidate[move_idx] = new_point
        new_score = compute_score(candidate)
        
        # Accept or reject
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / max(temp, 1e-10)):
            current = candidate
            current_score = new_score
            if new_score > best_score:
                best = candidate.copy()
                best_score = new_score
    
    return best, best_score

def polish_least_squares(points, time_budget, k=100.0, R_target=0.1):
    """
    Polish using scipy.optimize.least_squares.
    """
    start_time = time.time()
    
    def residuals(x_flat):
        x = x_flat.reshape(N, 2)
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
        
        s_val = soft_min_value(x, k)
        return np.array([R_target - s_val])

    x0 = points.flatten()
    
    try:
        result = least_squares(
            residuals, 
            x0, 
            method='trf', 
            max_nfev=20000,
            xtol=1e-12, 
            ftol=1e-12
        )
        
        x_opt = result.x.reshape(N, 2)
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception as e:
        print(f"LSQ failed: {e}")
        return points, compute_score(points)

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Optimization Phase with multiple strategies
    # Phase 1: LSQ polish on initial grid
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        polished, polished_score = polish_least_squares(best_points, remaining * 0.3)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    # Phase 2: Simulated annealing to explore
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10.0:
        sa_points, sa_score = simulated_annealing(best_points, start_temp=0.03, end_temp=0.001, 
                                                   max_iter=3000, time_limit=min(remaining * 0.4, 30))
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
    
    # Phase 3: LSQ polish after SA
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        polished, polished_score = polish_least_squares(best_points, remaining * 0.3)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    # Phase 4: Active-set Newton refinement
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        # Use current best score as target, slightly higher to push further
        r_target = best_score * 1.001
        new_points, new_score = active_set_newton(best_points, r_target, max_iter=30)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
    
    # Phase 5: Another round of perturbation + polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 8.0:
        noise = np.random.normal(0, 0.005, size=best_points.shape)
        perturbed = best_points + noise
        rad = np.linalg.norm(perturbed, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            perturbed[mask] /= rad[mask, np.newaxis]
        
        polished, polished_score = polish_least_squares(perturbed, remaining * 0.4)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
        
        # Final Newton refinement
        remaining = time_limit - (time.time() - start_time)
        if remaining > 1.0:
            r_target = best_score * 1.001
            new_points, new_score = active_set_newton(best_points, r_target, max_iter=20)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
    
    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    return best_points

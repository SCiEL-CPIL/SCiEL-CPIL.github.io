import time
from math import log, floor, sqrt
import random
import numpy as np
from scipy.optimize import minimize

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set_from_params(params, m):
    """
    Generate a set from geometric parameters.
    params: [log_r, log_s] where r = exp(log_r), s = exp(log_s)
    m: number of non-zero elements
    Returns a sorted list of unique non-negative integers containing 0.
    """
    log_r, log_s = params
    r = np.exp(log_r)
    s = np.exp(log_s)
    
    # Generate geometric sequence
    # k = 1 to m
    ks = np.arange(1, m + 1)
    vals = s * (r ** ks)
    
    # Floor to integers
    U = np.floor(vals).astype(int)
    
    # Filter out non-positive values (though s>0, r>1 usually ensures positive)
    U = U[U > 0]
    
    # Convert to list, add 0, sort, unique
    U_list = [0] + list(U)
    U_list = sorted(set(U_list))
    
    return U_list

def negative_score(params, m):
    """
    Objective function for minimization (minimizing negative score).
    """
    U = geometric_set_from_params(params, m)
    if len(U) < 2:
        return 1000.0
    
    s = score_set(U)
    if s < 0:
        return 1000.0
    
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # 1. Initial coarse grid search for geometric parameters
    # We search over m (size) and initial guesses for log_r, log_s
    # r is typically between 1.5 and 3.0 -> log_r in [0.4, 1.1]
    # s is typically between 1.0 and 10.0 -> log_s in [0.0, 2.3]
    
    m_values = [5, 6, 7, 8, 9, 10, 12, 15]
    
    # Coarse grid
    log_r_grid = np.linspace(0.3, 1.2, 10)
    log_s_grid = np.linspace(0.0, 2.5, 10)
    
    for m in m_values:
        if time.time() > deadline - 60:
            break
        for lr in log_r_grid:
            for ls in log_s_grid:
                U = geometric_set_from_params([lr, ls], m)
                if len(U) < 2:
                    continue
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    # 2. Local refinement using Nelder-Mead on continuous parameters
    # Use the best grid points and some random points as seeds for optimization
    
    # Identify the parameters that gave the best score in the grid
    # We'll just re-run a small grid to find the best param set, or keep the best U and back-calculate
    # For simplicity, we will run optimization from a few diverse starting points
    
    start_points = []
    
    # Add points from the coarse grid that scored well
    for m in m_values:
        for lr in log_r_grid:
            for ls in log_s_grid:
                U = geometric_set_from_params([lr, ls], m)
                if len(U) < 2:
                    continue
                s = score_set(U)
                if s > best_score - 0.01: # Close to best
                    start_points.append(([lr, ls], m))
    
    # Add some random points
    for _ in range(20):
        m = random.choice(m_values)
        lr = random.uniform(0.4, 1.1)
        ls = random.uniform(0.0, 2.0)
        start_points.append(([lr, ls], m))
    
    # If we found a good set, try to estimate its geometric parameters
    if len(best_set) > 2:
        # Estimate r and s from the best set
        # This is tricky, so we rely on the grid/random seeds
        pass

    # Run Nelder-Mead optimization for each seed
    for seed_params, m in start_points:
        if time.time() > deadline - 10:
            break
        
        try:
            # Bounds are implicit in the function, but we can guide it
            # x0 = [log_r, log_s]
            res = minimize(
                negative_score, 
                seed_params, 
                args=(m,), 
                method='Nelder-Mead', 
                options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200}
            )
            
            if res.success:
                best_params = res.x
                U = geometric_set_from_params(best_params, m)
                if len(U) >= 2:
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U
        except Exception:
            continue

    # 3. Final Polish: Element-wise local search on the best geometric set
    # This corrects for rounding errors in the geometric approximation
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 100:
            improved = False
            iteration += 1
            n = len(current)
            
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try small deltas
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved and n < 20 and time.time() < deadline - 1:
                max_val = max(current)
                # Try values near the gap or max
                for new_val in range(1, max_val + 10):
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                        
            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))

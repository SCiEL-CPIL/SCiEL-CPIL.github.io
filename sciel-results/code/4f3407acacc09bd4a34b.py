import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N_ORIG = 12
M = 40
N_OPT = 6
FULL_MASK_OPT = (1 << N_OPT) - 1
N_MASKS = 1 << N_OPT  # 64

def _count_pairs(masks):
    """Count comparable pairs in a list of masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _solve_ilp():
    """
    Formulate as ILP:
    x_i in {0,1} for each of the 64 masks
    Maximize sum of z_{ij} over comparable pairs
    subject to:
      z_{ij} <= x_i, z_{ij} <= x_j, z_{ij} >= x_i + x_j - 1
      sum_i x_i = M
    """
    # Get list of comparable pairs (i < j)
    pairs = []
    for i in range(N_MASKS):
        for j in range(i + 1, N_MASKS):
            if (i & j) == i or (i & j) == j:
                pairs.append((i, j))
    
    n_pairs = len(pairs)
    n_vars = N_MASKS + n_pairs
    
    # Objective: maximize sum of z's -> minimize -sum of z's
    c = np.zeros(n_vars)
    for p_idx in range(n_pairs):
        c[N_MASKS + p_idx] = -1.0
    
    # Constraints:
    n_constraints = 1 + 3 * n_pairs
    A = np.zeros((n_constraints, n_vars))
    lb = np.zeros(n_constraints)
    ub = np.zeros(n_constraints)
    
    # Constraint 1: sum x_i = M
    A[0, :N_MASKS] = 1.0
    lb[0] = M
    ub[0] = M
    
    # Constraints for each pair
    for p_idx, (i, j) in enumerate(pairs):
        base = 1 + 3 * p_idx
        # z_p - x_i <= 0
        A[base, N_MASKS + p_idx] = 1.0
        A[base, i] = -1.0
        lb[base] = -np.inf
        ub[base] = 0.0
        
        # z_p - x_j <= 0
        A[base + 1, N_MASKS + p_idx] = 1.0
        A[base + 1, j] = -1.0
        lb[base + 1] = -np.inf
        ub[base + 1] = 0.0
        
        # z_p - x_i - x_j <= -1
        A[base + 2, N_MASKS + p_idx] = 1.0
        A[base + 2, i] = -1.0
        A[base + 2, j] = -1.0
        lb[base + 2] = -np.inf
        ub[base + 2] = -1.0
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds and integrality
    integrality = np.zeros(n_vars)
    integrality[:N_MASKS] = 1
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    except Exception:
        return None, 0
    
    if result is not None and result.status == 0:
        x = result.x[:N_MASKS]
        selected = [i for i in range(N_MASKS) if x[i] > 0.5]
        score = _count_pairs(selected)
        return selected, score
    else:
        return None, 0

def _local_search_fallback():
    """Fallback local search if ILP fails or is slow."""
    all_masks = list(range(N_MASKS))
    best_score = -1
    best_mask_set = None
    
    for restart in range(50):
        random.seed(restart)
        current_set = set(random.sample(all_masks, M))
        current_score = _count_pairs(list(current_set))
        
        if current_score > best_score:
            best_score = current_score
            best_mask_set = set(current_set)
        
        T = 5.0
        no_improve = 0
        
        for _ in range(3000):
            if no_improve > 1500:
                break
            
            remove_mask = random.choice(list(current_set))
            while True:
                add_mask = random.randint(0, FULL_MASK_OPT)
                if add_mask not in current_set:
                    break
            
            delta = 0
            for m in current_set:
                if m == remove_mask:
                    continue
                old_comp = 1 if ((remove_mask & m) == remove_mask or (remove_mask & m) == m) else 0
                delta -= old_comp
                new_comp = 1 if ((add_mask & m) == add_mask or (add_mask & m) == m) else 0
                delta += new_comp
            
            if delta > 0 or random.random() < math.exp(delta / max(T, 0.1)):
                current_set.remove(remove_mask)
                current_set.add(add_mask)
                current_score += delta
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_mask_set = set(current_set)
            else:
                no_improve += 1
            T *= 0.998
    
    return list(best_mask_set), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_masks = None
    best_score = 0
    
    # Try ILP first
    if time.time() < deadline - 30:
        try:
            masks, score = _solve_ilp()
            if masks is not None:
                best_masks = masks
                best_score = score
        except Exception:
            pass
    
    # If ILP didn't give good result or failed, try local search
    if best_score < 100 or time.time() < deadline - 5:
        try:
            masks_ls, score_ls = _local_search_fallback()
            if score_ls > best_score:
                best_masks = masks_ls
                best_score = score_ls
        except Exception:
            pass
    
    if best_masks is None:
        best_masks = list(range(M))
    
    result = []
    for mask in best_masks:
        result.append([b for b in range(N_OPT) if (mask >> b) & 1])
    
    return result

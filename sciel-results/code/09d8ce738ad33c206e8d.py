# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _precompute_mappings():
    """Precompute index mappings for the four linear forms."""
    y_idx, x_idx = np.indices((10, 10))
    # x - y: range -9 to 9, shift by 9 -> 0 to 18
    map_m1 = (x_idx - y_idx + 9).astype(int)
    # x: range 0 to 9
    map_x = x_idx.astype(int)
    # x + y: range 0 to 18
    map_p1 = (x_idx + y_idx).astype(int)
    # x + 2y: range 0 to 27
    map_p2 = (x_idx + 2 * y_idx).astype(int)
    return map_m1, map_x, map_p1, map_p2

MAP_M1, MAP_X, MAP_P1, MAP_P2 = _precompute_mappings()
NUM_BINS_M1 = 19
NUM_BINS_X = 10
NUM_BINS_P1 = 19
NUM_BINS_P2 = 28

def _entropy_and_grad_from_counts(counts):
    """
    Computes Shannon entropy (bits) and gradient w.r.t counts.
    counts: array of shape (num_bins,)
    Returns: H, grad_counts (shape num_bins,)
    """
    total = counts.sum()
    if total < 1e-12:
        return 0.0, np.zeros_like(counts)
    
    probs = counts / total
    # H = - sum p log2 p
    # dH/dp_i = -1/ln(2) * (log2(p_i) + 1)
    # dH/dcounts_i = (1/total) * dH/dp_i
    
    non_zero = probs > 1e-12
    if not np.any(non_zero):
        return 0.0, np.zeros_like(counts)
    
    p_nz = probs[non_zero]
    H = -np.sum(p_nz * np.log2(p_nz))
    
    grad_p = np.zeros_like(counts)
    grad_p[non_zero] = -1.0 / np.log(2.0) * (np.log2(p_nz) + 1.0)
    
    grad_counts = grad_p / total
    return H, grad_counts

def _fast_ratio(P):
    """
    Fast computation of D/M for a valid probability matrix P.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total = P.sum()
    if total <= 1e-9:
        return 0.0

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()
    
    flat_P = P.flatten()
    
    counts_m1 = np.bincount(MAP_M1.flatten(), weights=flat_P, minlength=NUM_BINS_M1)
    counts_x = np.bincount(MAP_X.flatten(), weights=flat_P, minlength=NUM_BINS_X)
    counts_p1 = np.bincount(MAP_P1.flatten(), weights=flat_P, minlength=NUM_BINS_P1)
    counts_p2 = np.bincount(MAP_P2.flatten(), weights=flat_P, minlength=NUM_BINS_P2)
    
    H_m1, _ = _entropy_and_grad_from_counts(counts_m1)
    H_x, _ = _entropy_and_grad_from_counts(counts_x)
    H_p1, _ = _entropy_and_grad_from_counts(counts_p1)
    H_p2, _ = _entropy_and_grad_from_counts(counts_p2)
    
    M = max(H_x, H_p1, H_p2)
    if M <= 1e-9:
        return 0.0
        
    return H_m1 / M

def _objective_and_grad_z(z):
    """
    Objective: -ratio(P(z)) where P is softmax(z).
    Returns value and gradient w.r.t z.
    """
    # Softmax
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    sum_exp = np.sum(exp_z)
    P = exp_z / sum_exp
    
    # Reshape P to 10x10 for mapping usage, though we can use flat
    P_mat = P.reshape(10, 10)
    
    # Compute counts
    flat_P = P
    counts_m1 = np.bincount(MAP_M1.flatten(), weights=flat_P, minlength=NUM_BINS_M1)
    counts_x = np.bincount(MAP_X.flatten(), weights=flat_P, minlength=NUM_BINS_X)
    counts_p1 = np.bincount(MAP_P1.flatten(), weights=flat_P, minlength=NUM_BINS_P1)
    counts_p2 = np.bincount(MAP_P2.flatten(), weights=flat_P, minlength=NUM_BINS_P2)
    
    # Compute entropies and gradients w.r.t counts
    H_m1, grad_counts_m1 = _entropy_and_grad_from_counts(counts_m1)
    H_x, grad_counts_x = _entropy_and_grad_from_counts(counts_x)
    H_p1, grad_counts_p1 = _entropy_and_grad_from_counts(counts_p1)
    H_p2, grad_counts_p2 = _entropy_and_grad_from_counts(counts_p2)
    
    M = max(H_x, H_p1, H_p2)
    
    if M < 1e-9:
        return 0.0, np.zeros_like(z)
    
    ratio = H_m1 / M
    value = -ratio
    
    # Gradient of ratio w.r.t P
    # f = H_m1 / M
    # df/dP = (1/M) dH_m1/dP - (H_m1 / M^2) dM/dP
    
    # dH_m1/dP:
    # grad_counts_m1 is dH_m1/dcounts_m1.
    # counts_m1[b] = sum_{i: map_m1(i)=b} P_i
    # dcounts_m1[b]/dP_i = 1 if map_m1(i)=b else 0
    # dH_m1/dP_i = grad_counts_m1[map_m1(i)]
    
    grad_H_m1_flat = grad_counts_m1[MAP_M1.flatten()]
    
    # dM/dP:
    # M = max(H_x, H_p1, H_p2)
    # We use the gradient of the active term. If tie, average? 
    # For stability, let's compute all and pick the max, or use a smooth max.
    # Given L-BFGS-B, picking the max is usually okay if we restart.
    
    grad_H_x_flat = grad_counts_x[MAP_X.flatten()]
    grad_H_p1_flat = grad_counts_p1[MAP_P1.flatten()]
    grad_H_p2_flat = grad_counts_p2[MAP_P2.flatten()]
    
    # Determine active gradient
    if H_x >= H_p1 and H_x >= H_p2:
        grad_M_flat = grad_H_x_flat
    elif H_p1 >= H_x and H_p1 >= H_p2:
        grad_M_flat = grad_H_p1_flat
    else:
        grad_M_flat = grad_H_p2_flat
        
    grad_ratio_flat = (1.0 / M) * grad_H_m1_flat - (H_m1 / (M * M)) * grad_M_flat
    
    # Chain rule: grad_ratio w.r.t z
    # P_i = exp(z_i) / S
    # dP_i/dz_j = P_i (delta_ij - P_j)
    # grad_obj_z_j = sum_i (grad_obj_P_i * dP_i/dz_j)
    # = sum_i (grad_obj_P_i * P_i * (delta_ij - P_j))
    # = grad_obj_P_j * P_j - P_j * sum_i (grad_obj_P_i * P_i)
    
    dot_term = np.sum(grad_ratio_flat * P)
    grad_obj_z = P * (grad_ratio_flat - dot_term)
    
    # We are minimizing -ratio, so gradient is -grad_ratio
    grad_obj_z = -grad_obj_z
    
    return value, grad_obj_z

def _local_search(P_init, time_limit):
    """
    Simple local search on the probability matrix.
    """
    best_P = P_init.copy()
    best_score = _fast_ratio(best_P)
    rng = np.random.RandomState(123)
    start_time = time.time()
    
    for _ in range(2000):
        if time.time() > time_limit:
            break
        i, j = rng.randint(10), rng.randint(10)
        old_val = best_P[i, j]
        if old_val < 1e-6:
            continue
            
        for factor in [0.5, 0.8, 1.2, 1.5, 0.0, 0.2]:
            new_P = best_P.copy()
            new_P[i, j] = old_val * factor
            s = new_P.sum()
            if s < 1e-9:
                continue
            new_P = new_P / s
            new_score = _fast_ratio(new_P)
            if new_score > best_score:
                best_P = new_P
                best_score = new_score
    return best_P, best_score

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 array P of finite weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_P = np.ones((10, 10)) / 100.0
    best_score = _fast_ratio(best_P)
    
    # Initial candidates
    candidates_z = [np.zeros(100)]
    
    # Add some structured initial guesses
    # 1. Uniform
    # 2. Diagonal x+y=const
    z_diag = np.zeros(100)
    for y in range(10):
        for x in range(10):
            if x + y == 9:
                z_diag[y*10+x] = 10
            else:
                z_diag[y*10+x] = -10
    candidates_z.append(z_diag)
    
    # 3. Anti-diagonal x-y=const
    z_adia = np.zeros(100)
    for y in range(10):
        for x in range(10):
            if x - y == 0:
                z_adia[y*10+x] = 10
            else:
                z_adia[y*10+x] = -10
    candidates_z.append(z_adia)
    
    # 4. Random sparse
    rng = np.random.RandomState(42)
    for _ in range(5):
        z_r = rng.randn(100) * 5
        candidates_z.append(z_r)
        
    for z_init in candidates_z:
        if time.time() > deadline - 50:
            break
        try:
            res = minimize(
                _objective_and_grad_z, 
                z_init, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            z_opt = res.x
            # Convert z_opt to P
            z_max = np.max(z_opt)
            exp_z = np.exp(z_opt - z_max)
            P_opt = exp_z / np.sum(exp_z)
            P_opt = P_opt.reshape(10, 10)
            
            # Validate and normalize
            P_opt[P_opt < 0] = 0
            P_opt = P_opt / P_opt.sum()
            
            score_opt = _fast_ratio(P_opt)
            if score_opt > best_score:
                best_P = P_opt
                best_score = score_opt
        except Exception as e:
            pass
    
    # Local search polish on the best found
    if time.time() < deadline - 10:
        best_P, best_score = _local_search(best_P, deadline - 5)
        
    # Final check
    best_P[best_P < 0] = 0
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    else:
        best_P = np.ones((10, 10)) / 100.0
        
    return best_P

# EVOLVE-BLOCK-END

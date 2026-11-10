import numpy as np
import time
import sys
from scipy.optimize import minimize

def _precompute_tables():
    """Precompute index mappings for fast bincount-based ratio evaluation."""
    n = 10
    y_idx, x_idx = np.indices((n, n))
    
    # x - y range: -9 to 9 (19 values)
    diff = x_idx - y_idx
    diff_idx = (diff + 9).ravel()
    
    # x range: 0 to 9 (10 values)
    x_idx_flat = x_idx.ravel()
    
    # x + y range: 0 to 18 (19 values)
    sum_idx = (x_idx + y_idx).ravel()
    
    # x + 2y range: 0 to 27 (28 values)
    sum2_idx = (x_idx + 2 * y_idx).ravel()
    
    return diff_idx, x_idx_flat, sum_idx, sum2_idx

# Pre-compute indices at module level for speed
_DIFF_IDX, _X_IDX, _SUM_IDX, _SUM2_IDX = _precompute_tables()

def _fast_ratio(P):
    """
    Compute D/M ratio efficiently using precomputed indices.
    P is a flat array of 100 elements (row-major, P[y, x]).
    Returns 0.0 if invalid.
    """
    # Check finite
    if not np.all(np.isfinite(P)):
        return 0.0
    
    # Clip negatives
    P = np.maximum(P, 0.0)
    
    total = P.sum()
    if total <= 1e-9:
        return 0.0
    
    # Normalize
    P = P / total
    
    # Compute distributions
    # H_{-1}: x - y
    dist_diff = np.bincount(_DIFF_IDX, weights=P, minlength=19)
    p_diff = dist_diff[dist_diff > 0]
    H_diff = -np.sum(p_diff * np.log2(p_diff))
    
    # H_0: x
    dist_x = np.bincount(_X_IDX, weights=P, minlength=10)
    p_x = dist_x[dist_x > 0]
    H_x = -np.sum(p_x * np.log2(p_x))
    
    # H_1: x + y
    dist_sum = np.bincount(_SUM_IDX, weights=P, minlength=19)
    p_sum = dist_sum[dist_sum > 0]
    H_sum = -np.sum(p_sum * np.log2(p_sum))
    
    # H_2: x + 2y
    dist_sum2 = np.bincount(_SUM2_IDX, weights=P, minlength=28)
    p_sum2 = dist_sum2[dist_sum2 > 0]
    H_sum2 = -np.sum(p_sum2 * np.log2(p_sum2))
    
    M = max(H_x, H_sum, H_sum2)
    
    if M <= 1e-9:
        return 0.0
        
    return H_diff / M

def _softmax(x):
    """Numerically stable softmax."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def _objective_and_grad(z):
    """
    Objective for SLSQP: Maximize H_diff - lambda * max(H_x, H_sum, H_sum2)
    We minimize -H_diff + lambda * M.
    Actually, to use SLSQP constraints, we minimize -H_diff subject to H_x <= H_diff, H_sum <= H_diff, H_sum2 <= H_diff.
    Wait, SLSQP minimizes. So we minimize -H_diff.
    Constraints:
    H_diff - H_x >= 0
    H_diff - H_sum >= 0
    H_diff - H_sum2 >= 0
    
    This ensures M <= H_diff, so H_diff / M >= 1.
    Wait, the goal is to maximize H_diff / M.
    If we enforce H_diff >= M, then the ratio is >= 1.
    But we want to maximize the ratio.
    Maximizing H_diff/M is equivalent to minimizing log(M) - log(H_diff).
    
    Let's stick to the direct ratio formulation if possible, but SLSQP needs smooth constraints.
    The max function is non-smooth.
    
    Alternative:
    Minimize -log(H_diff) + log(M).
    This is smooth if M is determined by a unique max.
    If multiple are max, gradient is undefined.
    
    Let's try the constraint approach:
    Minimize -H_diff
    Subject to:
    H_diff - H_x >= 0
    H_diff - H_sum >= 0
    H_diff - H_sum2 >= 0
    
    This finds a P where H_diff is at least as large as the others.
    Does this maximize the ratio?
    If we push H_diff up and M down, the ratio goes up.
    The constraint H_diff >= M ensures the ratio is >= 1.
    Minimizing -H_diff (maximizing H_diff) under these constraints might not maximize the ratio if M can be made very small.
    Example: H_diff = 3, M = 3 => Ratio 1.
    H_diff = 4, M = 4 => Ratio 1.
    H_diff = 4, M = 2 => Ratio 2.
    
    So maximizing H_diff is not enough. We need to minimize M or maximize H_diff/M directly.
    
    Let's use the objective: f(z) = -H_diff / M.
    This is non-smooth at points where the argmax of M changes.
    However, SLSQP can handle some non-smoothness or we can use the log form.
    
    Let's try minimizing -log(H_diff) + log(M).
    Gradient of -log(H_diff) w.r.t z:
    H_diff = -sum p_i log p_i.
    dH/dz = J^T * dH/dp.
    dH/dp_i = -log p_i - 1.
    
    M = max(H0, H1, H2).
    dM/dz = dH_k/dz where k is argmax.
    
    If we assume a specific k is the max, we can compute the gradient.
    We can try all 3 permutations of which entropy is the max and solve the constrained optimization for each.
    
    Let's implement a helper that computes entropies and their gradients w.r.t logits z.
    """
    P = _softmax(z)
    
    # Compute distributions
    dist_diff = np.bincount(_DIFF_IDX, weights=P, minlength=19)
    dist_x = np.bincount(_X_IDX, weights=P, minlength=10)
    dist_sum = np.bincount(_SUM_IDX, weights=P, minlength=19)
    dist_sum2 = np.bincount(_SUM2_IDX, weights=P, minlength=28)
    
    def entropy_and_grad(dist, idx):
        p = dist
        H = 0.0
        # Gradient of H w.r.t P_flat
        # H = -sum p_i log p_i
        # dH/dP_flat_j = sum_i (dH/dp_i) * (dp_i/dP_flat_j)
        # dp_i/dP_flat_j = 1 if idx[j] == i else 0
        # So dH/dP_flat_j = -log p_{idx[j]} - 1 (if p_{idx[j]} > 0)
        
        # Vectorized:
        # We need a vector G_flat of length 100 such that G_flat[j] = -log p_{idx[j]} - 1
        # But we only have p (the binned probs).
        # We can map P_flat to binned probs? No, we have dist.
        # We need to unmap the gradient.
        
        # Let's compute dH/dP_flat directly.
        # For each cell j, it contributes to bin idx[j].
        # dH/dP_j = -log(P_bin[idx[j]]) - 1.
        
        # We need P_bin for each cell.
        # We can create an array P_cell_bin of length 100 where P_cell_bin[j] = dist[idx[j]].
        P_cell_bin = dist[idx]
        
        # Avoid log(0)
        P_cell_bin_safe = np.maximum(P_cell_bin, 1e-15)
        
        grad_P_flat = -np.log2(P_cell_bin_safe) - 1.0 / np.log(2)
        
        # If P_cell_bin is 0, the derivative is technically -inf, but we clipped.
        # In practice, if a bin is empty, moving mass into it creates a new term.
        # The formula -log p - 1 is valid for p > 0.
        
        H = -np.sum(p * np.log2(p))
        return H, grad_P_flat
    
    H_diff, g_diff_P = entropy_and_grad(dist_diff, _DIFF_IDX)
    H_x, g_x_P = entropy_and_grad(dist_x, _X_IDX)
    H_sum, g_sum_P = entropy_and_grad(dist_sum, _SUM_IDX)
    H_sum2, g_sum2_P = entropy_and_grad(dist_sum2, _SUM2_IDX)
    
    # Gradient w.r.t z
    # P = softmax(z)
    # dP_i/dz_j = P_i (delta_ij - P_j)
    # grad_z = J^T grad_P
    # grad_z_i = sum_j (dP_j/dz_i) grad_P_j
    # = sum_j P_j (delta_ji - P_i) grad_P_j
    # = P_i grad_P_i - P_i sum_j P_j grad_P_j
    # = P_i (grad_P_i - dot(P, grad_P))
    
    def grad_z_from_grad_P(g_P):
        dot = np.dot(P, g_P)
        return P * (g_P - dot)
    
    g_diff_z = grad_z_from_grad_P(g_diff_P)
    g_x_z = grad_z_from_grad_P(g_x_P)
    g_sum_z = grad_z_from_grad_P(g_sum_P)
    g_sum2_z = grad_z_from_grad_P(g_sum2_P)
    
    entropies = np.array([H_x, H_sum, H_sum2])
    grads = np.array([g_x_z, g_sum_z, g_sum2_z])
    
    # M = max(entropies)
    max_idx = np.argmax(entropies)
    M = entropies[max_idx]
    
    if M <= 1e-9 or H_diff <= 1e-9:
        return 1e10, np.zeros(100)
    
    # Objective: -log(H_diff) + log(M)
    obj = -np.log(H_diff) + np.log(M)
    
    # Gradient: -1/H_diff * g_diff_z + 1/M * g_M_z
    # g_M_z is the gradient of the max entropy.
    # Assuming unique max for now.
    g_M_z = grads[max_idx]
    
    grad_obj = -g_diff_z / H_diff + g_M_z / M
    
    return obj, grad_obj

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600 - 10
    
    # Initial best: Uniform
    best_P = np.ones((n, n)) / (n * n)
    best_score = _fast_ratio(best_P.ravel())
    
    # Multi-start SLSQP
    # We will try to optimize -log(H_diff) + log(M)
    # This is equivalent to minimizing log(M/H_diff) = -log(H_diff/M)
    
    starts = []
    
    # 1. Uniform
    starts.append(np.zeros(100))
    
    # 2. Random starts
    np.random.seed(42)
    for _ in range(20):
        starts.append(np.random.randn(100) * 0.5)
        
    # 3. Structured starts
    x_idx, y_idx = np.indices((n, n))
    
    # Diagonal
    diag_mask = (x_idx == y_idx).ravel()
    s = np.zeros(100)
    s[diag_mask] = 1.0
    starts.append(s)
    
    # Anti-diagonal
    anti_mask = (x_idx + y_idx == 9).ravel()
    s = np.zeros(100)
    s[anti_mask] = 1.0
    starts.append(s)
    
    # Corners
    s = np.zeros(100)
    s[0] = 1.0
    s[99] = 1.0
    s[9] = 1.0
    s[90] = 1.0
    starts.append(s)
    
    # Rows
    for r in range(0, 10, 2):
        s = np.zeros(100)
        s[r*10:(r+1)*10] = 1.0
        starts.append(s)
        
    # Cols
    for c in range(0, 10, 2):
        s = np.zeros(100)
        s[c::10] = 1.0
        starts.append(s)
    
    for i, start_x in enumerate(starts):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _objective_and_grad,
                start_x,
                method='SLSQP',
                jac=True, # We return (obj, grad)
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                P_candidate = _softmax(res.x)
                score = _fast_ratio(P_candidate)
                
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.reshape((n, n))
                    
        except Exception as e:
            continue

    # If we have time, do a few more random restarts
    while time.time() < deadline:
        start_x = np.random.randn(100) * 1.0
        try:
            res = minimize(
                _objective_and_grad,
                start_x,
                method='SLSQP',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                P_candidate = _softmax(res.x)
                score = _fast_ratio(P_candidate)
                
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.reshape((n, n))
        except:
            pass

    return best_P

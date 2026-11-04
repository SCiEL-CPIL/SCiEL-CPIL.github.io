"""Multi-Start Iterated Local Search with spectral-projection initialization for QAP."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _spectral_init(a, b, k=10, rng=None):
    """Generate an initial permutation using spectral decomposition of a and b."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(a)
    
    # Symmetrize
    a_sym = 0.5 * (a + a.T)
    b_sym = 0.5 * (b + b.T)
    
    # Center
    a_centered = a_sym - a_sym.mean(axis=1, keepdims=True)
    a_centered = a_centered - a_centered.mean(axis=0, keepdims=True)
    
    b_centered = b_sym - b_sym.mean(axis=1, keepdims=True)
    b_centered = b_centered - b_centered.mean(axis=0, keepdims=True)
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_centered)
        evals_b, evecs_b = np.linalg.eigh(b_centered)
    except np.linalg.LinAlgError:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p.tolist()
    
    # Sort by eigenvalue descending
    idx_a = np.argsort(-evals_a)
    idx_b = np.argsort(-evals_b)
    evals_a = evals_a[idx_a]
    evecs_a = evecs_a[:, idx_a]
    evals_b = evals_b[idx_b]
    evecs_b = evecs_b[:, idx_b]
    
    k_eff = min(k, n)
    Ua_top = evecs_a[:, :k_eff]
    Ub_top = evecs_b[:, :k_eff]
    
    weights_a = np.abs(evals_a[:k_eff]) + 1e-10
    weights_b = np.abs(evals_b[:k_eff]) + 1e-10
    W = np.sqrt(weights_a * weights_b)
    C = -(Ua_top * W) @ Ub_top.T
    
    row_ind, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p.tolist()


def _evaluate(a, b, p):
    """Compute the QAP objective value for permutation p."""
    b_perm = b[p, :]
    cost = np.sum(a * b_perm)
    return int(cost)


def _local_search(a, b, p, max_iters=5000, time_limit=None):
    """Iterated Local Search with vectorized steepest descent and reversal kicks."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _evaluate(a, b, p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    rng = np.random.default_rng()
    
    # Precompute for fast delta computation
    b_perm = b[p, :]
    row_sum = np.sum(a * b_perm, axis=1)
    col_sum = np.sum(a * b_perm, axis=0)
    
    # Precompute C_row[i, j] = sum_k a[i, k] * b[j, p[k]] for all i, j
    # C_col[j, i] = sum_k a[k, i] * b[p[k], j] for all i, j
    # These can be computed as:
    # C_row[i, :] = a[i, :] @ b[:, p]  -> a[i, :] @ B_col
    # where B_col[:, k] = b[:, p[k]]
    # Actually B_col = b[:, p]
    # C_row[i, j] = sum_k a[i, k] * b[j, p[k]] = (a[i, :] @ b[:, p])[j]
    # So C_row = a @ b[:, p]
    
    # C_col[j, i] = sum_k a[k, i] * b[p[k], j]
    # = (a[:, i] @ b[p, :])[j]
    # So C_col = (a.T @ b[p, :]).T ... let me think
    # C_col[j, i] = sum_k a[k, i] * b[p[k], j]
    # = (a[:, i] @ b[p, :])[j]
    # Let A_col_i = a[:, i], then C_col[j, i] = A_col_i @ b[p, :]
    # So C_col = (a.T @ b[p, :]).T? No.
    # a.T @ b[p, :] has shape (n, n) where (a.T @ b[p, :])[i, j] = sum_k a[k, i] * b[p[k], j]
    # That's exactly C_col[j, i]... wait, (a.T @ b[p, :])[i, j] = sum_k a.T[i, k] * b[p, :][k, j] = sum_k a[k, i] * b[p[k], j]
    # So (a.T @ b[p, :])[i, j] = C_col[j, i]
    # Therefore C_col = (a.T @ b[p, :]).T
    
    def _compute_C_row():
        return a @ b[:, p]
    
    def _compute_C_col():
        return (a.T @ b[p, :]).T
    
    C_row = _compute_C_row()
    C_col = _compute_C_col()
    
    def _apply_swap(i, j):
        """Apply swap and update row_sum, col_sum, current_cost, C_row, C_col. Returns delta."""
        nonlocal current_cost, p, row_sum, col_sum, C_row, C_col
        pi = p[i]
        pj = p[j]
        
        # Compute delta
        # Old: sum_k a[i,k]*b[pi, p[k]] + sum_k a[k,j]*b[p[k], pj]
        # New: sum_k a[i,k]*b[pj, p[k]] + sum_k a[k,j]*b[p[k], pi]
        # delta = C_row[i, pj] - C_row[i, pi] + C_col[pj, j] - C_col[pi, j]
        # But we need to be careful: when k=i or k=j, the terms are double-counted or need adjustment
        # Let's use the precomputed values but adjust for the diagonal terms
        
        # Actually, let's compute delta directly using the precomputed matrices
        # C_row[i, pi] = sum_k a[i, k] * b[pi, p[k]]
        # C_row[i, pj] = sum_k a[i, k] * b[pj, p[k]]
        # C_col[pi, j] = sum_k a[k, j] * b[p[k], pi]
        # C_col[pj, j] = sum_k a[k, j] * b[p[k], pj]
        
        # The delta for swapping i and j:
        # Old contribution from i: sum_k a[i,k]*b[pi, p[k]] = C_row[i, pi]
        # New contribution from i: sum_k a[i,k]*b[pj, p[k]] = C_row[i, pj]
        # Old contribution to j: sum_k a[k,j]*b[p[k], pj] = C_col[pj, j]
        # New contribution to j: sum_k a[k,j]*b[p[k], pi] = C_col[pi, j]
        
        # But wait, when k=i: a[i,i]*b[pi, pi] is in both old and new for i's row
        # When k=j: a[i,j]*b[pi, pj] is in old for i's row, a[i,j]*b[pj, pi] is in new
        # Similarly for j's column
        
        # Let's just compute delta as:
        delta = C_row[i, pj] - C_row[i, pi] + C_col[pi, j] - C_col[pj, j]
        
        # Apply the swap
        p[i], p[j] = pj, pi
        current_cost += delta
        
        # Update row_sum
        row_sum[i] = C_row[i, pj]
        row_sum[j] = C_row[j, pi]
        
        # Update col_sum
        col_sum[i] = C_col[pj, i]
        col_sum[j] = C_col[pi, j]
        
        # Update C_row and C_col
        # C_row[i, :] changes because p[i] changed from pi to pj
        # C_row[i, j] = sum_k a[i, k] * b[j, p[k]]
        # The only change is in the term where k=i: a[i, i] * b[j, p[i]] = a[i, i] * b[j, pj] instead of a[i, i] * b[j, pi]
        # So C_row[i, :] += a[i, i] * (b[:, pj] - b[:, pi])
        # Similarly C_row[j, :] += a[j, j] * (b[:, pi] - b[:, pj])
        
        C_row[i, :] += a[i, i] * (b[:, pj] - b[:, pi])
        C_row[j, :] += a[j, j] * (b[:, pi] - b[:, pj])
        
        # C_col[:, i] changes because p[i] changed
        # C_col[j, i] = sum_k a[k, i] * b[p[k], j]
        # The change is in the term where k=i: a[i, i] * b[p[i], j] = a[i, i] * b[pj, j] instead of a[i, i] * b[pi, j]
        # So C_col[:, i] += a[i, i] * (b[pj, :] - b[pi, :]).T
        # Similarly C_col[:, j] += a[j, j] * (b[pi, :] - b[pj, :]).T
        
        C_col[:, i] += a[i, i] * (b[pj, :] - b[pi, :]).T
        C_col[:, j] += a[j, j] * (b[pi, :] - b[pj, :]).T
        
        return delta
    
    def _find_best_swap():
        """Find the best improving swap using precomputed matrices."""
        nonlocal p
        min_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                delta = C_row[i, pj] - C_row[i, pi] + C_col[pi, j] - C_col[pj, j]
                if delta < min_delta:
                    min_delta = delta
                    best_i = i
                    best_j = j
        
        return min_delta, best_i, best_j
    
    for iteration in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Find best swap
        min_delta, best_i, best_j = _find_best_swap()
        
        if min_delta < 0 and best_i >= 0:
            # Apply best swap
            d = _apply_swap(best_i, best_j)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improving swap found, perform a kick
            if rng.random() < 0.5:
                # Single swap
                i, j = rng.choice(n, size=2, replace=False)
                if i != j:
                    d = _apply_swap(int(i), int(j))
            else:
                # Multi-swap kick
                num_swaps = rng.integers(2, 5)
                for _ in range(num_swaps):
                    i, j = rng.choice(n, size=2, replace=False)
                    if i != j:
                        d = _apply_swap(int(i), int(j))
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p.tolist(), best_cost


def solve(a, b):
    """Multi-Start Iterated Local Search for QAP."""
    n = len(a)
    
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    time_limit = 115
    
    best_p = None
    best_cost = np.inf
    
    num_starts = 5
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit - 30:
            break
        
        if start_idx == 0:
            p_init = _spectral_init(a, b, k=10)
        elif start_idx == 1:
            p_init = _spectral_init(a, b, k=5)
        elif start_idx == 2:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p_arr = np.empty(n, dtype=np.int64)
            p_arr[heavy] = central
            p_init = p_arr.tolist()
        elif start_idx == 3:
            p_init = list(np.random.default_rng(start_idx).permutation(n))
        else:
            p_init = _spectral_init(a, b, k=15)
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_init, max_iters=5000, time_limit=remaining_time - 2)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    if best_p is None:
        best_p = _spectral_init(a, b, k=10)
    
    return best_p
# EVOLVE-BLOCK-END

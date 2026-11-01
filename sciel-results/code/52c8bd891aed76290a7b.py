"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    # Try the structured 4-block rank-11 scheme (total rank 44)
    best = None
    best_rank = n * m * p

    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _four_block_decomposition(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass
    
    # Try to find a better sub-problem decomposition
    if time.time() < deadline - 30:
        try:
            sub_U, sub_V, sub_W, sub_r = _search_subproblem_rank(2, 3, 2, 11, deadline)
            if sub_U is not None and sub_r < 11:
                # Build the full decomposition using this sub-problem
                U2 = np.zeros((n * m, sub_r * 4))
                V2 = np.zeros((m * p, sub_r * 4))
                W2 = np.zeros((p * n, sub_r * 4))
                
                for b in range(4):
                    k_offset = b * 2
                    col_offset = b * sub_r
                    U2[:, col_offset:col_offset+sub_r] = sub_U
                    V2[:, col_offset:col_offset+sub_r] = np.zeros((m*p, sub_r))
                    W2[:, col_offset:col_offset+sub_r] = np.zeros((p*n, sub_r))
                    
                    # Map V and W indices
                    for jj in range(m):
                        for kk in range(2):
                            V2[jj * p + k_offset + kk, col_offset:col_offset+sub_r] = sub_V[jj * 2 + kk, :]
                    for kk in range(2):
                        for ii in range(n):
                            W2[(k_offset + kk) * n + ii, col_offset:col_offset+sub_r] = sub_W[kk * n + ii, :]
                
                if _verify(n, m, p, U2, V2, W2):
                    if sub_r * 4 < best_rank:
                        best = (U2, V2, W2)
                        best_rank = sub_r * 4
        except Exception:
            pass

    # Try further refinement on the 44-rank solution (attempt rank 43, 42, ...)
    if best is not None and time.time() < deadline - 10:
        try:
            U2, V2, W2, r2 = _refine_decomposition(n, m, p, best[0], best[1], best[2], best_rank, deadline)
            if r2 < best_rank and _verify(n, m, p, U2, V2, W2):
                best = (U2, V2, W2)
                best_rank = r2
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    Total rank: 44.
    """
    n, m, p = 2, 3, 8
    block_size = p // 4  # 2
    
    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44
    
    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44
    
    for b in range(4):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block
        
        # Strassen 7 terms for 2x2 * 2x2 part
        strassen_U = [
            {0: 1, 4: 1},      # M1
            {3: 1, 4: 1},      # M2
            {0: 1},            # M3
            {4: 1},            # M4
            {0: 1, 1: 1},      # M5
            {3: 1, 0: -1},     # M6
            {1: 1, 4: -1},     # M7
        ]
        
        strassen_V = [
            {0 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},  # M1
            {0 * p + k_offset + 0: 1},                           # M2
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: -1}, # M3
            {0 * p + k_offset + 1: 1, 0 * p + k_offset + 0: -1}, # M4
            {1 * p + k_offset + 1: 1},                           # M5
            {0 * p + k_offset + 0: 1, 0 * p + k_offset + 1: 1},  # M6
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},  # M7
        ]
        
        strassen_W = [
            {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: 1},  # M1
            {(k_offset + 0) * n + 1: 1},                             # M2
            {(k_offset + 1) * n + 0: 1, (k_offset + 1) * n + 1: 1},  # M3
            {(k_offset + 0) * n + 0: 1, (k_offset + 0) * n + 1: 1},  # M4
            {(k_offset + 0) * n + 0: -1, (k_offset + 1) * n + 0: 1}, # M5
            {(k_offset + 1) * n + 1: 1},                             # M6
            {(k_offset + 0) * n + 0: 1},                             # M7
        ]
        
        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val
        
        # Outer product part: A_right(2x1) * B_bottom(1x2) - rank 4
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk  # 7,8,9,10
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1
    
    return U, V, W, total_rank


def _two_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 2 blocks of size 4.
    For each 2x3 x 3x4 sub-problem, use a decomposition.
    """
    n, m, p = 2, 3, 8
    block_size = p // 2  # 4
    
    # For 2x3 x 3x4, try to find a good decomposition
    # Split into two 2x3 x 3x2 parts, each rank 11 -> total 22 per block, 44 total
    # Or try a smarter approach
    
    # Use the same 4-block structure but grouped differently
    # Actually, let's try: for each 2x3 x 3x4, use rank 20 (4 * 5)
    # We can think of 3x4 as two 3x2 blocks, each needing rank 11 for the full 2x3x3x2
    # But that's not quite right.
    
    # Let me try a different approach: treat the 2x3 x 3x4 as a single unit
    # and find a decomposition.
    
    # For now, fall back to 4-block
    return None, None, None, 999


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using the bilinear form:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    Vectorized for speed.
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    rank = U.shape[1]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)  # U[i*m+j, c]
    V_reshaped = V.reshape(m, p, rank)  # V[l*p+k, c]
    W_reshaped = W.reshape(p, n, rank)  # W[k*n+i, c]
    
    # u_val[i, c] = sum_j U[i*m+j, c] * A[i,j]
    u_vals = np.einsum('ijc,ij->ic', U_reshaped, A)  # (n, rank)
    
    # v_val[l, k, c] = sum_l V[l*p+k, c] * B[l,k]
    v_vals = np.einsum('lkp,lk->kp', V_reshaped, B)  # (p, rank)
    
    # C_comp[i, k] = sum_c W[k*n+i, c] * u_vals[i, c] * v_vals[k, c]
    C_comp = np.einsum('kic,ic,kc->ik', W_reshaped, u_vals, v_vals)
    
    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True


def _refine_decomposition(n, m, p, U, V, W, current_rank, deadline):
    """
    Try to reduce the rank by finding linear dependencies among columns.
    Uses a greedy approach: try removing each column one at a time and verify.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank
    
    # Try removing each column one at a time and verify.
    # Iterate multiple times until no more removals are possible or time runs out.
    removed_any = True
    while removed_any and time.time() < deadline - 5:
        removed_any = False
        for c_remove in range(rank - 1, -1, -1):
            if time.time() >= deadline - 5:
                break
            
            # Create new matrices without column c_remove
            mask = [c for c in range(rank) if c != c_remove]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            # Quick check: verify the new decomposition
            if _verify(n, m, p, U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed_any = True
                if rank <= 1:
                    return U, V, W, rank
    
    return U, V, W, rank


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            # Round to nearest half-integer
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

def _search_subproblem_rank(n_sub, m_sub, p_sub, max_rank, deadline):
    """
    Try to find a rank-r decomposition for the n_sub x m_sub x p_sub sub-problem.
    Uses a greedy/iterative approach.
    """
    from fractions import Fraction
    
    # The tensor T[i,j,k] = 1 for all (i,j,k) in the sub-problem
    # We want U (n_sub*m_sub x r), V (m_sub*p_sub x r), W (p_sub*n_sub x r)
    # such that sum_c U[i*m_sub+j, c] * V[j*p_sub+k, c] * W[k*n_sub+i, c] = 1
    
    best_U, best_V, best_W, best_r = None, None, None, max_rank + 1
    
    for r in range(1, max_rank + 1):
        if time.time() >= deadline - 5:
            break
        
        # Try to find a rank-r decomposition
        # Use a simple initialization and iterative refinement
        np.random.seed(42)
        
        for attempt in range(20):
            if time.time() >= deadline - 5:
                break
            
            U = np.random.randn(n_sub * m_sub, r)
            V = np.random.randn(m_sub * p_sub, r)
            W = np.random.randn(p_sub * n_sub, r)
            
            # Iterative refinement: for each (i,j,k), adjust the columns
            # to better satisfy the constraint
            converged = False
            for it in range(50):
                if time.time() >= deadline - 5:
                    break
                
                max_err = 0.0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            # Current value
                            val = 0.0
                            for c in range(r):
                                val += U[i * m_sub + j, c] * V[j * p_sub + k, c] * W[k * n_sub + i, c]
                            err = 1.0 - val
                            max_err = max(max_err, abs(err))
                
                if max_err < 1e-6:
                    converged = True
                    break
                
                # Simple gradient step (approximate)
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            val = 0.0
                            for c in range(r):
                                val += U[i * m_sub + j, c] * V[j * p_sub + k, c] * W[k * n_sub + i, c]
                            err = 1.0 - val
                            for c in range(r):
                                # Adjust U, V, W to reduce error
                                grad_U = V[j * p_sub + k, c] * W[k * n_sub + i, c]
                                grad_V = U[i * m_sub + j, c] * W[k * n_sub + i, c]
                                grad_W = U[i * m_sub + j, c] * V[j * p_sub + k, c]
                                U[i * m_sub + j, c] += 0.01 * err * grad_U
                                V[j * p_sub + k, c] += 0.01 * err * grad_V
                                W[k * n_sub + i, c] += 0.01 * err * grad_W
                
                if max_err < 1e-6:
                    converged = True
                    break
            
            if converged:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                if _verify_subproblem(n_sub, m_sub, p_sub, U_r, V_r, W_r):
                    best_U, best_V, best_W, best_r = U_r, V_r, W_r, r
                    break
        
        if best_r <= r:
            break
    
    return best_U, best_V, best_W, best_r


def _verify_subproblem(n_sub, m_sub, p_sub, U, V, W):
    """Verify a sub-problem decomposition."""
    rank = U.shape[1]
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                val = 0.0
                for c in range(rank):
                    val += U[i * m_sub + j, c] * V[j * p_sub + k, c] * W[k * n_sub + i, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


# EVOLVE-BLOCK-END

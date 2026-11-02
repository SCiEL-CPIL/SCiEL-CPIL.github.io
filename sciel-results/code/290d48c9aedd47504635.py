import numpy as np
import time
from scipy.linalg import null_space
# EVOLVE-BLOCK-START

# The 7 Strassen linear combinations for A, B, and C.
_A_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (0, 0, -1)],
    [(0, 1, 1), (1, 1, -1)]
]

_B_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(0, 1, 1), (1, 1, -1)],
    [(1, 0, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (1, 1, 1)]
]

_C_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 1, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1)]
]

def _generate_strassen_4x4():
    """
    Generates U, V, W for 4x4 Strassen decomposition directly.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Outer Strassen loop
    for t_outer in range(7):
        # Inner Strassen loop
        for t_inner in range(7):
            r_idx = t_outer * 7 + t_inner
            
            # A (U) part
            for (br, bc, sign) in _A_COMBOS[t_outer]:
                # The block is 2x2.
                # Local indices in the 2x2 block are 0,1.
                # Global row i = br*2 + ir, Global col k = bc*2 + ik
                # U index: i * 4 + k
                # The inner Strassen term t_inner applies to the 2x2 sub-block.
                # We need to know which entries of the 2x2 sub-block are touched by t_inner.
                # This is equivalent to _base_strassen_2x2 but we can reuse the logic.
                # Let's just use the base combos for the inner level.
                for (ir, ik, sign_inner) in _A_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    u_idx = i_global * n + k_global
                    U[u_idx, r_idx] += sign * sign_inner
            
            # B (V) part
            for (br, bc, sign) in _B_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _B_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    v_idx = i_global * n + k_global
                    V[v_idx, r_idx] += sign * sign_inner
            
            # C (W) part
            for (br, bc, sign) in _C_COMBOS[t_outer]:
                for (ir, ik, sign_inner) in _C_COMBOS[t_inner]:
                    # C block (br, bc) of size 2x2.
                    # Inner term t_inner touches (ir, ik) in the 2x2 block.
                    # Global row i = br*2 + ir, Global col k = bc*2 + ik
                    # W index: k * n + i
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    w_idx = k_global * n + i_global
                    W[w_idx, r_idx] += sign * sign_inner

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))

def _try_reduce_rank(U, V, W, n, m, p, deadline=None):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 terms.
    Returns (U_new, V_new, W_new) if a lower rank is found, else None.
    """
    rank = U.shape[1]
    if deadline is not None and time.time() > deadline:
        return None
    
    # For each pair of rank indices (a, b), check if term b can be expressed
    # as a linear combination of the other terms.
    # We do this by checking if the column b of the "tensor" is in the span
    # of the other columns.
    
    # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Actually, let's think differently. The decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # If we fix all but one term, the residual is a rank-1 tensor in (i,j,k).
    # Specifically, if term r is removed, the residual is:
    # R[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # This is a rank-1 tensor. The question is whether this rank-1 tensor can be
    # expressed as a linear combination of the other rank-1 tensors.
    #
    # A simpler approach: check if any two columns (U_col, V_col, W_col) are
    # linearly dependent in a way that allows elimination.
    
    # Alternative approach: try to find a non-trivial linear relation among the
    # 49 "rank-1 tensors" T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r].
    # If sum_r c_r * T_r = 0 for some non-zero c, then we can potentially
    # eliminate one term.
    
    # Flatten each rank-1 tensor to a vector of length n*m*p = 64.
    U_flat = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_flat = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_flat = np.asarray(W, dtype=float).reshape(p, n, rank)
    
    # Build matrix M where column r is the vectorized T_r.
    M = np.zeros((n*m*p, rank), dtype=float)
    for r in range(rank):
        T_r = U_flat[:, :, r] * V_flat[:, :, r] * W_flat[:, :, r]
        M[:, r] = T_r.reshape(n*m*p)
    
    # Find the null space of M. If there's a non-trivial null vector c,
    # then sum_r c_r * T_r = 0.
    # We can try to eliminate the term with the largest |c_r|.
    
    # Compute null space
    try:
        ns = null_space(M, rcond=1e-10)
    except Exception:
        return None
    
    if ns.shape[1] == 0:
        return None
    
    # For each null vector, try to eliminate one term.
    for ns_col in range(ns.shape[1]):
        if deadline is not None and time.time() > deadline:
            break
        
        c = ns[:, ns_col]
        # Find the index with the largest absolute coefficient
        max_idx = np.argmax(np.abs(c))
        if abs(c[max_idx]) < 1e-10:
            continue
        
        # Try to eliminate term max_idx.
        # New U = U[:, [0..max_idx-1, max_idx+1..rank-1]]
        # But we also need to adjust the remaining terms to compensate.
        # Actually, if sum_r c_r * T_r = 0, then T_max = -(1/c_max) * sum_{r != max} c_r * T_r
        # This means the decomposition with term max removed still works if we
        # can express it differently. But the standard decomposition uses
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
        # 
        # The issue is that the "rank-1 tensors" are not independent in the
        # decomposition sense. We need a different approach.
        
        # Let's try a different strategy: directly search for a rank-48
        # decomposition by relaxing the Strassen structure.
        # This is complex, so let's try a simpler heuristic:
        # Check if any two columns of U, V, or W are linearly dependent.
        
        # Check U columns
        U_mat = np.asarray(U, dtype=float)
        V_mat = np.asarray(V, dtype=float)
        W_mat = np.asarray(W, dtype=float)
        
        # For each pair (a, b), check if U[:, b] is in span of U[:, a] and
        # similarly for V and W. If so, we might be able to merge.
        # This is a heuristic and may not work.
        
        pass
    
    # If no simple reduction found, try a more systematic approach:
    # Solve for a rank-48 decomposition by setting up a linear system.
    # This is expensive, so only try if we have time.
    
    return None

def _construct_rank48_by_search(n, m, p, deadline):
    """
    Attempt to construct a rank-48 decomposition by searching for a
    linear dependence in the Strassen decomposition.
    """
    U, V, W = _generate_strassen_4x4()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    # Try to reduce rank
    result = _try_reduce_rank(U, V, W, n, m, p, deadline)
    if result is not None:
        U_new, V_new, W_new = result
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = 4, 4, 4
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, generate the rank-49 Strassen decomposition
    U, V, W = _generate_strassen_4x4()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce to rank 48
    if time.time() < deadline:
        result = _construct_rank48_by_search(n, m, p, deadline)
        if result is not None:
            U, V, W = result
    
    # Return the best result found
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

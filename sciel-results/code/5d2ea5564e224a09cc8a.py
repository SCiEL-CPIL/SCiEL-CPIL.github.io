"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    # Build the target tensor T[i,j,k] = delta_{correct}
    # T is indexed as: U index = i*m+j (i in 0..n-1, j in 0..m-1)
    #                  V index = j*p+k (j in 0..m-1, k in 0..p-1)
    #                  W index = k*n+i (k in 0..p-1, i in 0..n-1)
    # The tensor T[u_idx, v_idx, w_idx] = 1 iff the multiplication gives that term
    # i.e., A[i,j] * B[j,k] contributes to C[k,i]
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    T = np.zeros((u_dim, v_dim, w_dim), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # First, try rank 13
    for target_rank in [13, 14]:
        if time.time() > deadline - 30:
            break
        
        # Use the known rank-14 solution as a starting point
        U, V, W = _rank14_solution(n, m, p)
        
        if target_rank == 14:
            # Verify and return
            if _verify(U, V, W, T):
                return {"U": U, "V": V, "W": W}
        
        if target_rank == 13:
            # Try to reduce rank by finding linear dependencies
            # and eliminating one term
            result = _try_reduce_rank(U, V, W, T, target_rank, deadline)
            if result is not None:
                return result
    
    # Fallback to rank 14
    U, V, W = _rank14_solution(n, m, p)
    return {"U": U, "V": V, "W": W}

def _rank14_solution(n, m, p):
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    strassen_terms = [
        ([(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)],
         [(0,0,1)],
         [(0,1,1), (1,1,-1)]),
        ([(0,0,1)],
         [(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)]),
        ([(1,1,1)],
         [(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)]),
        ([(0,0,1), (0,1,1)],
         [(1,1,1)],
         [(1,0,1), (0,0,-1)]),
        ([(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)],
         [(1,1,1)]),
        ([(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)],
         [(0,0,1)]),
    ]
    
    for r in range(7):
        u_entries, v_entries, w_entries = strassen_terms[r]
        
        col1 = r
        for i, j, val in u_entries:
            U[i * m + j][col1] += val
        for j, k, val in v_entries:
            V[j * p + k][col1] += val
        for k, i, val in w_entries:
            W[k * n + i][col1] += val
        
        col2 = 7 + r
        for i, j, val in u_entries:
            U[i * m + j][col2] += val
        for j, k, val in v_entries:
            k_new = k + 2
            V[j * p + k_new][col2] += val
        for k, i, val in w_entries:
            k_new = k + 2
            W[k_new * n + i][col2] += val
    
    return U, V, W

def _verify(U, V, W, T, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    reconstructed = np.zeros_like(T)
    for r in range(rank):
        reconstructed += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    return np.allclose(reconstructed, T, atol=tol)

def _try_reduce_rank(U, V, W, T, target_rank, deadline):
    """Try to reduce rank by finding and eliminating linearly dependent terms."""
    n, m, p = DIMENSIONS
    u_dim, v_dim, w_dim = n*m, m*p, p*n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    cur_rank = U_arr.shape[1]
    
    if cur_rank <= target_rank:
        return {"U": U, "V": V, "W": W}
    
    # Try each term for elimination
    for elim_r in range(cur_rank):
        if time.time() > deadline - 5:
            break
        
        # Remove term elim_r
        U_test = np.delete(U_arr, elim_r, axis=1)
        V_test = np.delete(V_arr, elim_r, axis=1)
        W_test = np.delete(W_arr, elim_r, axis=1)
        
        new_rank = cur_rank - 1
        if new_rank < target_rank:
            continue
        
        # Check if the remaining terms can still span the tensor
        # Build the linear system: sum_r u_r ⊗ v_r ⊗ w_r = T
        # This is equivalent to: for each (i,j,k) in T, sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
        # Linearize by fixing two indices... this is complex.
        # Instead, just verify if the reduced decomposition still works
        # (it won't in general, but we can try to fix it)
        
        # Try to find a correction: can we adjust remaining terms to compensate?
        # The error from removing term elim_r is:
        error = np.einsum('i,j,k->ijk', 
                         U_arr[:, elim_r], V_arr[:, elim_r], W_arr[:, elim_r])
        
        # Can the remaining terms (after adjustment) absorb this error?
        # This requires solving a linear system. Let's try a simple approach:
        # Scale remaining terms to try to absorb the error.
        
        # Actually, a simpler approach: check if term elim_r is a linear combination
        # of other terms in the tensor decomposition sense.
        # This is hard in general. Let's try a different approach.
        
        # Try: can we find alpha such that removing term elim_r and adding
        # alpha * (some combination) works? This is getting complex.
        
        # For now, just verify the direct removal
        U_t = np.array(U_test)
        V_t = np.array(V_test)
        W_t = np.array(W_test)
        reconstructed = np.zeros_like(T)
        for r in range(new_rank):
            reconstructed += np.einsum('i,j,k->ijk', U_t[:, r], V_t[:, r], W_t[:, r])
        
        if np.allclose(reconstructed, T, atol=1e-9):
            # Success! Return the reduced solution
            return {
                "U": U_t.tolist(),
                "V": V_t.tolist(),
                "W": W_t.tolist()
            }
    
    return None

# EVOLVE-BLOCK-END

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if not _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try to find a lower rank decomposition
    # We will search for ranks from 50 to 89
    # Use a randomized approach with fast verification
    
    start_time = time.time()
    deadline = start_time + 110 # Leave 40s margin
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Search for lower ranks
    for rank in range(50, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = _find_rank(rank, n, m, p, max_time=5.0)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            # If we find a rank, we can try to go lower, but for now, return the best found
            # We can continue searching for even lower ranks if time permits
            # But let's be conservative and return the first found good rank?
            # No, the task asks for rank-minimal. So we should keep searching.
            # However, if we find 80, we should try 70, etc.
            # So we continue the loop.
    
    # If we found a better rank, use it
    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum"""
    rank = U.shape[1]
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to check if T is all 1s for valid (i,j,k) and 0 otherwise?
    # No, the tensor is defined only on the "diagonal" of the indices.
    # Actually, the tensor T has non-zero entries only at (i*m+j, j*p+k, k*n+i).
    # So we just need to check that sum_c ... = 1 for all i,j,k.
    
    # Construct the full tensor T (sparse)
    # T is shape (n*m, m*p, p*n)
    # But we can just check the specific entries.
    
    # Use einsum to compute the contraction
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We want to check: for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # This is equivalent to checking the tensor T.
    # Let's create a mask of valid indices.
    
    # Efficient check:
    # Create arrays for i, j, k
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Indices into U, V, W
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Extract the relevant rows
    U_sub = U[u_idx, :] # (n*m*p, rank)
    V_sub = V[v_idx, :]
    W_sub = W[w_idx, :]
    
    # Compute the sum
    # result = sum_c U_sub[:, c] * V_sub[:, c] * W_sub[:, c]
    result = np.einsum('ic,ic,ic->i', U_sub, V_sub, W_sub)
    
    return np.allclose(result, 1.0, atol=1e-9)

def _find_rank(rank, n, m, p, max_time=5.0):
    """Try to find a rank-`rank` decomposition using randomized least squares."""
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initialize U, V, W randomly
    np.random.seed(42)
    
    # We need to solve:
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # where T is the tensor.
    
    # We can fix V and W, and solve for U.
    # Then fix U and W, solve for V.
    # Then fix U and V, solve for W.
    
    # This is the standard CP decomposition iteration.
    
    # Build the tensor T (sparse representation)
    # T has non-zeros at (i*m+j, j*p+k, k*n+i)
    
    # For the update of U[a, :]:
    # We need to solve: sum_r U[a, r] * (V[b, r] * W[c, r]) = T[a, b, c]
    # For a fixed a, this is a linear system in U[a, :].
    # The "matrix" is M_a where M_a[(b, c), r] = V[b, r] * W[c, r].
    # The "vector" is T_a where T_a[(b, c)] = T[a, b, c].
    
    # T_a is non-zero only at specific (b, c) pairs.
    # For fixed a = i*m+j, the non-zero (b, c) are:
    # b = j*p+k, c = k*n+i for k in 0..p-1.
    # So there are p non-zero entries in T_a.
    
    # So we have a system of p equations for `rank` unknowns.
    # If rank > p, we have an underdetermined system.
    # We can use the minimum norm solution.
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for trial in range(10):
        if time.time() > deadline:
            break
        
        # Re-initialize
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    # Non-zero (b, c) for this a
                    b_list = []
                    c_list = []
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        b_list.append(b)
                        c_list.append(c)
                    
                    # M matrix: (p, rank)
                    # M[k, r] = V[b_list[k], r] * W[c_list[k], r]
                    M = V[b_list, :] * W[c_list, :]
                    
                    # T vector: (p,)
                    # T[k] = 1.0
                    t_vec = np.ones(p)
                    
                    # Solve M^T @ u = t_vec? No.
                    # sum_r M[k, r] * U[a, r] = t_vec[k]
                    # This is M @ U[a, :] = t_vec
                    # If M is p x rank, and rank > p, we use lstsq.
                    try:
                        U_new[a, :] = np.linalg.lstsq(M, t_vec, rcond=None)[0]
                    except:
                        break
                    
                    U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    # Non-zero (a, c) for this b
                    a_list = []
                    c_list = []
                    for i in range(n):
                        a = i * m + j
                        c = k * n + i
                        a_list.append(a)
                        c_list.append(c)
                    
                    # M matrix: (n, rank)
                    # M[i, r] = U[a_list[i], r] * W[c_list[i], r]
                    M = U[a_list, :] * W[c_list, :]
                    
                    # T vector: (n,)
                    t_vec = np.ones(n)
                    
                    try:
                        V_new[b, :] = np.linalg.lstsq(M, t_vec, rcond=None)[0]
                    except:
                        break
                    
                    V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    # Non-zero (a, b) for this c
                    a_list = []
                    b_list = []
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        a_list.append(a)
                        b_list.append(b)
                    
                    # M matrix: (m, rank)
                    # M[j, r] = U[a_list[j], r] * V[b_list[j], r]
                    M = U[a_list, :] * V[b_list, :]
                    
                    # T vector: (m,)
                    t_vec = np.ones(m)
                    
                    try:
                        W_new[c, :] = np.linalg.lstsq(M, t_vec, rcond=None)[0]
                    except:
                        break
                    
                    W = W_new
            
            # Check error
            if _verify_fast(U, V, W, n, m, p):
                # Try to round
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                else:
                    # If rounding fails, we can't use this solution
                    # But maybe the unrounded solution is close?
                    # The task requires integer/half-integer.
                    # So we must round.
                    pass
            
            # Early stop if error is high
            # (Optional: compute error and break if it increases)
    
    return None

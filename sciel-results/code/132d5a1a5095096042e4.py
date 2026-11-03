import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matmul blocks
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def verify_fast(U_arr, V_arr, W_arr, n=2, m=4, p=4, tol=1e-9):
    """Fast verification using vectorized numpy operations."""
    rank = U_arr.shape[1]
    # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Vectorize over r
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    T_recon = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U_arr[:, r]  # shape (n*m,)
        v_col = V_arr[:, r]  # shape (m*p,)
        w_col = W_arr[:, r]  # shape (p*n,)
        # Reshape to (n, m), (m, p), (p, n)
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        # Contribution to T[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # This is an outer-product-like operation
        # For each i: u_mat[i,:] * (v_mat * w_mat.T)[j,k] ... 
        # Actually: T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # Let's compute via einsum or manual
        for i in range(n):
            # u_mat[i,:] is shape (m,)
            # Need sum over j: u_mat[i,j] * v_mat[j,k] * w_mat[k,i] for all k
            # = (u_mat[i,:] * (v_mat @ w_mat[:,i])) ... no
            # For fixed i: T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # = u_mat[i,j] * (v_mat[j,:] * w_mat[:,i])[k]
            # So T[i] += u_mat[i] * (v_mat * w_mat[:,i].T) ... 
            # w_mat[:,i] is shape (p,), v_mat is (m,p)
            # v_mat * w_mat[:,i].T broadcasts to (m,p)
            contrib = u_mat[i] * (v_mat * w_mat[:, i].T)  # shape (m, p)
            T_recon[i] += contrib
    
    max_err = np.max(np.abs(T_recon - 1.0))
    return max_err < tol, max_err

def try_rank_reduction(U, V, W, n=2, m=4, p=4):
    """Try to reduce rank by greedily removing redundant terms."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    # Multiple passes of greedy removal
    for pass_num in range(8):
        if time_up():
            break
        removed_any = False
        # Try removing each term
        for r in range(current_rank):
            if time_up():
                break
            mask = [i for i in range(current_rank) if i != r]
            new_U = U_arr[:, mask]
            new_V = V_arr[:, mask]
            new_W = W_arr[:, mask]
            
            valid, err = verify_fast(new_U, new_V, new_W, n, m, p)
            if valid:
                U_arr = new_U
                V_arr = new_V
                W_arr = new_W
                current_rank = len(mask)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank

def try_pairwise_merge(U, V, W, n=2, m=4, p=4):
    """Try to merge pairs of rank-1 terms into a single term (rank reduction by 1).
    
    If we have two terms r1, r2, we ask: can we find coefficients such that
    the sum of the two rank-1 tensors equals a single rank-1 tensor?
    This is generally not possible, but we can try to find a linear combination
    that cancels out.
    
    Alternative: try to find a linear dependence among the rank-1 tensors.
    If sum_{r in S} c_r * T_r = 0 for some subset S and coefficients,
    then we can potentially remove terms.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    # Try removing individual terms (already done in try_rank_reduction)
    # Try removing pairs: check if removing two terms still works
    for pass_num in range(5):
        if time_up():
            break
        removed_any = False
        for r1 in range(current_rank):
            if time_up():
                break
            for r2 in range(r1 + 1, current_rank):
                if time_up():
                    break
                mask = [i for i in range(current_rank) if i != r1 and i != r2]
                new_U = U_arr[:, mask]
                new_V = V_arr[:, mask]
                new_W = W_arr[:, mask]
                
                valid, err = verify_fast(new_U, new_V, new_W, n, m, p)
                if valid:
                    U_arr = new_U
                    V_arr = new_V
                    W_arr = new_W
                    current_rank = len(mask)
                    removed_any = True
                    break
            if removed_any:
                break
        
        if not removed_any:
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank

def linear_algebraic_reduction(U, V, W, n=2, m=4, p=4):
    """Use linear algebra to find linear dependencies among rank-1 terms.
    
    The tensor T = sum_r U_r ⊗ V_r ⊗ W_r where U_r is a vector of length n*m,
    V_r is length m*p, W_r is length p*n.
    
    If there's a linear relation sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0,
    we might be able to reduce rank. But this is hard to find directly.
    
    Instead, we can check: for a given subset of terms, can we solve for
    the remaining terms to still produce the target tensor?
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    # For each term r, check if it's "redundant" in the sense that
    # its contribution can be expressed as a linear combination of others.
    # This is equivalent to checking if the tensor T - T_r is in the span
    # of the other rank-1 tensors.
    
    # A practical approach: for each term, zero it out and see if the
    # remaining terms can be adjusted to still produce T.
    # This is a nonlinear problem, so we use a simple heuristic:
    # check if the term's contribution is small relative to others.
    
    # Better: use the fact that if we have a valid decomposition,
    # we can try to find a linear relation among the columns of
    # the "flattened" tensor.
    
    # Flatten the tensor to a matrix of size (n*m*p) x rank
    # where each column r is the vectorization of U_r ⊗ V_r ⊗ W_r
    # This matrix has rank at most rank, and if rank < n*m*p,
    # there are linear dependencies.
    
    nmp = n * m * p
    T_matrix = np.zeros((nmp, current_rank), dtype=float)
    
    for r in range(current_rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        
        # The contribution to T[i,j,k] is u_col[i,j] * v_col[j,k] * w_col[k,i]
        # Vectorize this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_matrix[i * m * p + j * p + k, r] = u_col[i, j] * v_col[j, k] * w_col[k, i]
    
    # Check rank of T_matrix
    # If rank(T_matrix) < current_rank, there are linear dependencies
    try:
        matrix_rank = np.linalg.matrix_rank(T_matrix, tol=1e-10)
    except:
        matrix_rank = current_rank
    
    if matrix_rank < current_rank:
        # There are linear dependencies. Find them.
        # The null space of T_matrix gives coefficients c such that sum c_r T_r = 0
        # We can use this to reduce rank.
        # For simplicity, just do greedy removal (already done elsewhere)
        pass
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank

def random_search(rank, n=2, m=4, p=4, iterations=500, seed=None):
    """Random search for a decomposition of given rank."""
    if seed is not None:
        np.random.seed(seed)
    
    for it in range(iterations):
        if time_up():
            break
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        valid, err = verify_fast(U, V, W, n, m, p)
        if valid:
            return U.tolist(), V.tolist(), W.tolist(), rank
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank via greedy removal
    if not time_up():
        U2, V2, W2, rank2 = try_rank_reduction(U, V, W, n, m, p)
        if rank2 < best_rank:
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try pairwise removal
    if not time_up() and best_rank > 24:
        U3, V3, W3, rank3 = try_pairwise_merge(U, V, W, n, m, p)
        if rank3 < best_rank:
            valid, err = verify_decomposition(U3, V3, W3, n, m, p)
            if valid:
                best_rank = rank3
                best_result = {"U": U3, "V": V3, "W": W3}
    
    # Try random search for lower ranks
    if not time_up() and best_rank > 24:
        for r in range(min(best_rank - 1, 27), 20, -1):
            if time_up():
                break
            result = random_search(r, n, m, p, iterations=200)
            if result is not None:
                U4, V4, W4, rank4 = result
                valid, err = verify_decomposition(U4, V4, W4, n, m, p)
                if valid and rank4 < best_rank:
                    best_rank = rank4
                    best_result = {"U": U4, "V": V4, "W": W4}
                break
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")

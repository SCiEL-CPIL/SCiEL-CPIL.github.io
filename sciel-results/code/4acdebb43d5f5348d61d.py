"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses a robust vectorized ALS approach to search for lower ranks, 
    falling back to the schoolbook rank 60 solution.
    """
    start_time = time.time()
    deadline = 110  # Keep 10s safety margin before 120s limit
    n, m, p = 3, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 12, 20, 15
    
    # Schoolbook baseline (Rank 60)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_solution(U, V, W, rank):
        """Compute max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :rank], V[v_idx, :rank] * W[w_idx, :rank])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def solve_als_vectorized(rank, seed, max_iters=500, time_budget=10.0):
        """
        Solve for U, V, W using Vectorized ALS.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        reg = 1e-6
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Precompute index mappings for vectorization
        # For U update: for each (i,j), we need V[j*p+k, :] * W[k*n+i, :] for all k
        # We can precompute the indices.
        
        for it in range(max_iters):
            if time.time() - start_time > start_time + time_budget:
                break
            
            # Update U
            # For each (i,j), A_u is (p, rank), b_u is (p,)
            # A_u[k, :] = V[j*p+k, :] * W[k*n+i, :]
            # We can vectorize over i, j
            # Let's just loop over i, j as p is small (5)
            for i in range(n):
                for j in range(m):
                    u_idx = i*m + j
                    # Construct A_u
                    # k ranges 0..p-1
                    # v_indices = j*p + np.arange(p)
                    # w_indices = (np.arange(p)*n + i)
                    v_idx_arr = j*p + np.arange(p)
                    w_idx_arr = np.arange(p)*n + i
                    A_u = V[v_idx_arr, :] * W[w_idx_arr, :] # shape (p, rank)
                    b_u = np.ones(p)
                    # Solve (A_u^T A_u + reg I) u = A_u^T b_u
                    try:
                        U[u_idx, :] = np.linalg.solve(A_u.T @ A_u + reg * np.eye(rank), A_u.T @ b_u)
                    except np.linalg.LinAlgError:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j*p + k
                    # A_v[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    u_idx_arr = np.arange(n)*m + j
                    w_idx_arr = k*n + np.arange(n)
                    A_v = U[u_idx_arr, :] * W[w_idx_arr, :] # shape (n, rank)
                    b_v = np.ones(n)
                    try:
                        V[v_idx, :] = np.linalg.solve(A_v.T @ A_v + reg * np.eye(rank), A_v.T @ b_v)
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k*n + i
                    # A_w[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    u_idx_arr = i*m + np.arange(m)
                    v_idx_arr = np.arange(m)*p + k
                    A_w = U[u_idx_arr, :] * V[v_idx_arr, :] # shape (m, rank)
                    b_w = np.ones(m)
                    try:
                        W[w_idx, :] = np.linalg.solve(A_w.T @ A_w + reg * np.eye(rank), A_w.T @ b_w)
                    except np.linalg.LinAlgError:
                        pass
            
            # Check error
            curr_err = check_solution(U, V, W, rank)
            if curr_err < best_err:
                best_err = curr_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
                
        return best_err, best_U, best_V, best_W

    # Try ranks from low to high, but limit the search to a reasonable range
    # and time.
    # We know 60 works. Let's try 50, 45, 40...
    # If we find a valid one, we break.
    
    # The theoretical lower bound for 3x4 * 4x5 is at least max(n*m, m*p, p*n) = 20
    # But actually, for matrix multiplication, the rank is known to be at most n*m*p
    # and at least max(n*m, m*p, p*n).
    # Let's try a wider range of ranks, starting from a reasonable lower bound.
    # For 3x4*4x5, known best is likely around 40-50.
    
    ranks_to_try = [30, 35, 40, 45, 50, 55]
    
    for rank in ranks_to_try:
        if time.time() - start_time > deadline - 5:
            break
        
        found = False
        # Try more seeds and allocate time more carefully
        seeds = [42, 123, 999, 2024, 555, 777, 888, 111, 222, 333]
        
        for seed in seeds:
            if time.time() - start_time > deadline - 5:
                break
            
            time_left = deadline - (time.time() - start_time) - 2
            if time_left < 1:
                break
            
            # Allocate time per attempt
            err, U_c, V_c, W_c = solve_als_vectorized(rank, seed, max_iters=300, time_budget=time_left/len(seeds))
            
            if err < 1e-10:
                best_rank = rank
                best_U, best_V, best_W = U_c, V_c, W_c
                found = True
                break
        
        if found:
            break
            
    # Final verification
    if best_rank < rank_schoolbook:
        final_err = check_solution(best_U, best_V, best_W, best_rank)
        if final_err > 1e-8:
            # Fallback
            best_rank = rank_schoolbook
            best_U, best_V, best_W = U_sb, V_sb, W_sb

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    U_out = to_list(best_U[:, :best_rank])
    V_out = to_list(best_V[:, :best_rank])
    W_out = to_list(best_W[:, :best_rank])
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

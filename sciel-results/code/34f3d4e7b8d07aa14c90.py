"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses a robust vectorized ALS approach with block-structure exploitation.
    """
    start_time = time.time()
    deadline = 135  # Keep 15s safety margin before 150s limit
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

    def check_solution_fast(U, V, W, rank):
        """Faster check using vectorized operations."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :rank]
                for k in range(p):
                    v_row = V[j*p+k, :rank]
                    w_row = W[k*n+i, :rank]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def solve_als(rank, seed, max_iters=1000, time_budget=8.0):
        """
        Solve for U, V, W using ALS with regularization.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        reg = 1e-4
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Precompute index arrays
        # For U update: for each (i,j), need V[j*p+k,:] * W[k*n+i,:] for k=0..p-1
        # For V update: for each (j,k), need U[i*m+j,:] * W[k*n+i,:] for i=0..n-1
        # For W update: for each (k,i), need U[i*m+j,:] * V[j*p+k,:] for j=0..m-1
        
        for it in range(max_iters):
            if time.time() - start_time > start_time + time_budget:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    u_idx = i*m + j
                    v_idx_arr = j*p + np.arange(p)
                    w_idx_arr = np.arange(p)*n + i
                    A_u = V[v_idx_arr, :] * W[w_idx_arr, :]  # (p, rank)
                    b_u = np.ones(p)
                    try:
                        U[u_idx, :] = np.linalg.solve(A_u.T @ A_u + reg * np.eye(rank), A_u.T @ b_u)
                    except np.linalg.LinAlgError:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j*p + k
                    u_idx_arr = np.arange(n)*m + j
                    w_idx_arr = k*n + np.arange(n)
                    A_v = U[u_idx_arr, :] * W[w_idx_arr, :]  # (n, rank)
                    b_v = np.ones(n)
                    try:
                        V[v_idx, :] = np.linalg.solve(A_v.T @ A_v + reg * np.eye(rank), A_v.T @ b_v)
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k*n + i
                    u_idx_arr = i*m + np.arange(m)
                    v_idx_arr = np.arange(m)*p + k
                    A_w = U[u_idx_arr, :] * V[v_idx_arr, :]  # (m, rank)
                    b_w = np.ones(m)
                    try:
                        W[w_idx, :] = np.linalg.solve(A_w.T @ A_w + reg * np.eye(rank), A_w.T @ b_w)
                    except np.linalg.LinAlgError:
                        pass
            
            # Check error periodically
            if it % 10 == 0:
                curr_err = check_solution_fast(U, V, W, rank)
                if curr_err < best_err:
                    best_err = curr_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
                
        # Final check
        curr_err = check_solution_fast(best_U, best_V, best_W, rank)
        if curr_err < best_err:
            best_err = curr_err
        
        return best_err, best_U, best_V, best_W

    def polish(U, V, W, rank, time_budget=5.0):
        """
        High-precision polish pass with tighter tolerance.
        """
        start_polish = time.time()
        reg = 1e-8
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(500):
            if time.time() - start_polish > time_budget:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    u_idx = i*m + j
                    v_idx_arr = j*p + np.arange(p)
                    w_idx_arr = np.arange(p)*n + i
                    A_u = V[v_idx_arr, :] * W[w_idx_arr, :]
                    b_u = np.ones(p)
                    try:
                        U[u_idx, :] = np.linalg.solve(A_u.T @ A_u + reg * np.eye(rank), A_u.T @ b_u)
                    except np.linalg.LinAlgError:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j*p + k
                    u_idx_arr = np.arange(n)*m + j
                    w_idx_arr = k*n + np.arange(n)
                    A_v = U[u_idx_arr, :] * W[w_idx_arr, :]
                    b_v = np.ones(n)
                    try:
                        V[v_idx, :] = np.linalg.solve(A_v.T @ A_v + reg * np.eye(rank), A_v.T @ b_v)
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k*n + i
                    u_idx_arr = i*m + np.arange(m)
                    v_idx_arr = np.arange(m)*p + k
                    A_w = U[u_idx_arr, :] * V[v_idx_arr, :]
                    b_w = np.ones(m)
                    try:
                        W[w_idx, :] = np.linalg.solve(A_w.T @ A_w + reg * np.eye(rank), A_w.T @ b_w)
                    except np.linalg.LinAlgError:
                        pass
                        
            if it % 50 == 0:
                curr_err = check_solution_fast(U, V, W, rank)
                if curr_err < 1e-12:
                    break
        
        return U, V, W

    # Try ranks from low to high
    # Theoretical lower bound is max(n*m, m*p, p*n) = 20
    # Known best for 3x4*4x5 is likely around 40-50
    ranks_to_try = [25, 30, 35, 40, 45, 50, 55]
    
    for rank in ranks_to_try:
        if time.time() - start_time > deadline - 10:
            break
        
        found = False
        # Try multiple seeds
        seeds = [42, 123, 999, 2024, 555, 777, 888, 111, 222, 333, 444, 555, 666, 777, 888]
        
        for seed in seeds:
            if time.time() - start_time > deadline - 10:
                break
            
            time_left = deadline - (time.time() - start_time) - 5
            if time_left < 2:
                break
            
            # Allocate time per attempt
            time_per_attempt = min(6.0, time_left / max(1, len(seeds) - seeds.index(seed)))
            
            err, U_c, V_c, W_c = solve_als(rank, seed, max_iters=800, time_budget=time_per_attempt)
            
            if err < 1e-8:
                # Polish
                U_p, V_p, W_p = polish(U_c, V_c, W_c, rank, time_budget=3.0)
                final_err = check_solution_fast(U_p, V_p, W_p, rank)
                
                if final_err < 1e-10:
                    best_rank = rank
                    best_U, best_V, best_W = U_p, V_p, W_p
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

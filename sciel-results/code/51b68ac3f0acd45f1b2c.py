import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook (rank 72)
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # ALS tensor decomposition: minimize ||T - sum_c U_c * V_c * W_c||_F
    # where U_c, V_c, W_c are rank-1 slices
    
    def als_decompose(T, init_rank, n_iter=100, tol=1e-10, seed=42):
        """
        ALS decomposition of a 3-way tensor into sum of rank-1 tensors.
        Returns U, V, W as 2D arrays of shape (dim1, rank), (dim2, rank), (dim3, rank)
        """
        rng = np.random.RandomState(seed)
        d1, d2, d3 = T.shape
        
        # Initialize with random values
        U = rng.randn(d1, init_rank) * 0.1
        V = rng.randn(d2, init_rank) * 0.1
        W = rng.randn(d3, init_rank) * 0.1
        
        prev_err = np.inf
        for it in range(n_iter):
            # Update U: for each column c, U[:, c] = argmin ||T - sum_c U_c * V_c * W_c||
            # This is equivalent to solving a least squares problem
            for c in range(init_rank):
                # Compute residual without term c
                # T_approx = sum_{c' != c} U[:, c'] * V[:, c'] * W[:, c']
                # We need to find U[:, c] such that T - T_approx is best approximated by U[:, c] * V[:, c] * W[:, c]
                
                # Efficient approach: 
                # Let B = V[:, c] * W[:, c] (outer product, shape d2 x d3)
                # We want U[:, c] such that U[:, c] * B approximates T - T_approx
                # This is a tensor least squares problem
                
                # Alternative: use the fact that for fixed V_c and W_c,
                # the optimal U_c satisfies:
                # U_c = (sum_{i,j,k} (T[i,j,k] - T_approx[i,j,k]) * V_c[j] * W_c[k]) / (sum_{j,k} V_c[j]^2 * W_c[k]^2)
                
                # Compute T_approx without term c
                T_approx = np.zeros_like(T)
                for cp in range(init_rank):
                    if cp != c:
                        T_approx += np.einsum('i,j,k->ijk', U[:, cp], V[:, cp], W[:, cp])
                
                residual = T - T_approx
                
                # Optimal U[:, c] = sum_{j,k} residual[:, j, k] * V_c[j] * W_c[k] / (||V_c||^2 * ||W_c||^2)
                num = np.einsum('ijk,j,k->i', residual, V[:, c], W[:, c])
                denom = np.dot(V[:, c], V[:, c]) * np.dot(W[:, c], W[:, c])
                if abs(denom) > 1e-15:
                    U[:, c] = num / denom
                else:
                    U[:, c] = rng.randn(d1) * 0.1
                
                # Update V[:, c]
                T_approx = np.zeros_like(T)
                for cp in range(init_rank):
                    if cp != c:
                        T_approx += np.einsum('i,j,k->ijk', U[:, cp], V[:, cp], W[:, cp])
                
                residual = T - T_approx
                num = np.einsum('ijk,i,k->j', residual, U[:, c], W[:, c])
                denom = np.dot(U[:, c], U[:, c]) * np.dot(W[:, c], W[:, c])
                if abs(denom) > 1e-15:
                    V[:, c] = num / denom
                else:
                    V[:, c] = rng.randn(d2) * 0.1
                
                # Update W[:, c]
                T_approx = np.zeros_like(T)
                for cp in range(init_rank):
                    if cp != c:
                        T_approx += np.einsum('i,j,k->ijk', U[:, cp], V[:, cp], W[:, cp])
                
                residual = T - T_approx
                num = np.einsum('ijk,i,j->k', residual, U[:, c], V[:, c])
                denom = np.dot(U[:, c], U[:, c]) * np.dot(V[:, c], V[:, c])
                if abs(denom) > 1e-15:
                    W[:, c] = num / denom
                else:
                    W[:, c] = rng.randn(d3) * 0.1
            
            # Compute error
            T_approx = np.zeros_like(T)
            for c in range(init_rank):
                T_approx += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            err = np.linalg.norm(T - T_approx)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W
    
    def round_to_half_integer(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    def try_rank(target_rank, seed=42, n_iter=200):
        """Try to find a decomposition of given rank using ALS."""
        U, V, W = als_decompose(T, target_rank, n_iter=n_iter, seed=seed)
        
        # Round to half-integers
        U_r = round_to_half_integer(U)
        V_r = round_to_half_integer(V)
        W_r = round_to_half_integer(W)
        
        # Convert to lists
        U_list = [list(U_r[i]) for i in range(n * m)]
        V_list = [list(V_r[i]) for i in range(m * p)]
        W_list = [list(W_r[i]) for i in range(p * n)]
        
        if verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
        return None
    
    # Try various ranks with different seeds and iterations
    for target_rank in range(50, 66):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = try_rank(target_rank, seed=seed, n_iter=150)
            if result is not None:
                U_list, V_list, W_list = result
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                print(f"ALS found rank {target_rank} with seed {seed}")
                break
        
        if best_rank < rank_schoolbook and best_rank <= target_rank:
            break
    
    # If ALS didn't find anything better, try the block decomposition as fallback
    if best_rank >= 66:
        # Block-structured decomposition using Strassen for 2x2x2 blocks
        # n=3 as 2+1, m=4 as 2+2, p=6 as 2+2+2
        # 2x2x2 blocks: 1*2*3 = 6 blocks x rank 7 = 42
        # 1x2x2 blocks: 1*2*3 = 6 blocks x rank 4 = 24
        # Total: 66
        
        def strassen_2x2x2():
            """Return the 7 Strassen terms as (U_4, V_4, W_4) tuples for 2x2x2."""
            terms = []
            # M1 = (A11+A22)(B11+B22)
            terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
            # M2 = (A21+A22)B11
            terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
            # M3 = A11(B12-B22)
            terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
            # M4 = A22(B21-B11)
            terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))
            # M5 = (A11+A12)B22
            terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
            # M6 = (A21-A11)(B11+B12)
            terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
            # M7 = (A12-A22)(B21+B22)
            terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
            return terms
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        n_blocks = [[0, 1], [2]]
        m_blocks = [[0, 1], [2, 3]]
        p_blocks = [[0, 1], [2, 3], [4, 5]]
        
        strassen_terms = strassen_2x2x2()
        
        for ni, n_blk in enumerate(n_blocks):
            for mi, m_blk in enumerate(m_blocks):
                for pi, p_blk in enumerate(p_blocks):
                    if len(n_blk) == 2 and len(m_blk) == 2 and len(p_blk) == 2:
                        # 2x2x2: use Strassen
                        for u_vals, v_vals, w_vals in strassen_terms:
                            U_cols.append(u_vals)
                            V_cols.append(v_vals)
                            W_cols.append(w_vals)
                    else:
                        # 1x2x2: direct rank 4
                        for k_idx in range(2):
                            for j_idx in range(2):
                                u_vals = [0, 0]
                                u_vals[k_idx] = 1
                                v_vals = [0, 0, 0, 0]
                                v_vals[k_idx * 2 + j_idx] = 1
                                w_vals = [0, 0]
                                w_vals[j_idx] = 1
                                U_cols.append(u_vals)
                                V_cols.append(v_vals)
                                W_cols.append(w_vals)
        
        rank_found = len(U_cols)
        
        # Build full matrices
        U_mat = np.zeros((n * m, rank_found))
        V_mat = np.zeros((m * p, rank_found))
        W_mat = np.zeros((p * n, rank_found))
        
        for c in range(rank_found):
            for idx, val in enumerate(U_cols[c]):
                if val != 0:
                    U_mat[idx, c] = val
            for idx, val in enumerate(V_cols[c]):
                if val != 0:
                    V_mat[idx, c] = val
            for idx, val in enumerate(W_cols[c]):
                if val != 0:
                    W_mat[idx, c] = val
        
        U_res = [list(map(int, U_mat[i])) for i in range(n * m)]
        V_res = [list(map(int, V_mat[i])) for i in range(m * p)]
        W_res = [list(map(int, W_mat[i])) for i in range(p * n)]
        
        if verify(U_res, V_res, W_res, n, m, p):
            if rank_found < best_rank:
                best = {"U": U_res, "V": V_res, "W": W_res}
                best_rank = rank_found
    
    return best

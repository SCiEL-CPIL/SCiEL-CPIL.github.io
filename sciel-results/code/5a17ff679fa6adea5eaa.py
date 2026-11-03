import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (24, rank)
    
    # Start with the known rank-88 solution
    rank_start = 88
    deadline = time.time() + 110 # Safety margin
    
    def build_initial_solution(rank):
        """
        Reconstruct the rank-88 solution and pad with zeros if rank > 88.
        If rank < 88, we cannot use this directly, but for the initial step 
        we assume we are trying to reduce from 88. 
        However, the optimizer works on a fixed rank. 
        We will try to optimize the rank-88 solution to see if it can be 
        pushed to have zero entries in some columns, effectively reducing rank.
        """
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # Helper to add a local 2x2x2 Strassen block
        def add_strassen_block(i_base, j_base, k_base, s_offset):
            # Local indices i, j, k in {0, 1}
            # Global indices:
            # i_g = i_base + i
            # j_g = j_base + j
            # k_g = k_base + k
            
            # Strassen factors (7 of them)
            # P1 = (A00 + A11)(B00 + B11)
            # P2 = (A10 + A11)B00
            # P3 = A00(B01 - B11)
            # P4 = A11(B10 - B00)
            # P5 = (A00 + A01)B11
            # P6 = (A10 - A00)(B00 + B01)
            # P7 = (A01 - A11)(B10 + B11)
            
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            factors = [
                # (u, v, w) where u,v,w are length 4 vectors for local 2x2 blocks
                # u: A00, A01, A10, A11
                # v: B00, B01, B10, B11
                # w: C00, C01, C10, C11 -> indices 0,1,2,3 correspond to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
                # Wait, W index is k*n + i.
                # For local block, n=2.
                # k=0, i=0 -> 0
                # k=0, i=1 -> 1
                # k=1, i=0 -> 2
                # k=1, i=1 -> 3
                
                # P1
                (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
                # P2
                (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
                # P3
                (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
                # P4
                (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])),
                # P5
                (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
                # P6
                (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
                # P7
                (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))
            ]
            
            for s_idx, (u, v, w) in enumerate(factors):
                s = s_offset + s_idx
                for i in range(2):
                    for j in range(2):
                        u_idx = (i_base + i) * m + (j_base + j)
                        U[u_idx, s] += u[i * 2 + j]
                        
                for j in range(2):
                    for k in range(2):
                        v_idx = (j_base + j) * p + (k_base + k)
                        V[v_idx, s] += v[j * 2 + k]
                        
                for k in range(2):
                    for i in range(2):
                        w_idx = (k_base + k) * n + (i_base + i)
                        W[w_idx, s] += w[k * 2 + i]
                        
        # Part 1: 2x4x8 block (i in 0..1)
        # 8 blocks of 2x2x2
        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        add_strassen_block(0, 0, 6, 21)
        add_strassen_block(0, 2, 0, 28)
        add_strassen_block(0, 2, 2, 35)
        add_strassen_block(0, 2, 4, 42)
        add_strassen_block(0, 2, 6, 49)
        
        # Part 2: 1x4x8 block (i in 2)
        # Schoolbook for i=2
        s_offset = 56
        s_idx = 0
        for j in range(4):
            for k in range(8):
                s = s_offset + s_idx
                s_idx += 1
                
                # U[2 * 4 + j, s] = 1
                U[2 * m + j, s] = 1.0
                
                # V[j * 8 + k, s] = 1
                V[j * p + k, s] = 1.0
                
                # W[k * 3 + 2, s] = 1
                W[k * n + 2, s] = 1.0
                
        return U, V, W

    U0, V0, W0 = build_initial_solution(rank_start)
    
    # Flatten for optimization
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    size_U = n * m * rank_start
    size_V = m * p * rank_start
    size_W = p * n * rank_start
    
    def unpack(x):
        U = x[:size_U].reshape((n*m, rank_start))
        V = x[size_U:size_U+size_V].reshape((m*p, rank_start))
        W = x[size_U+size_V:].reshape((p*n, rank_start))
        return U, V, W
    
    def error_func(x):
        if time.time() > deadline:
            return float('inf')
        U, V, W = unpack(x)
        
        # Compute tensor T[i, j, k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        # We want T[i, j, k] = 1 for all i, j, k.
        # Error = sum_{i,j,k} (T[i,j,k] - 1)^2
        
        # Efficient computation using einsum
        # T_ijk = sum_s U_{ij,s} V_{jk,s} W_{ki,s}
        # Note: W is indexed by k*n+i. Let's rename W'_{k,i} = W[k*n+i, s].
        # So W_matrix has shape (p, n, rank).
        
        # Reshape U to (n, m, rank)
        U_3d = U.reshape((n, m, rank_start))
        # Reshape V to (m, p, rank)
        V_3d = V.reshape((m, p, rank_start))
        # Reshape W to (p, n, rank)
        W_3d = W.reshape((p, n, rank_start))
        
        # T_{i, j, k} = sum_s U_{i,j,s} V_{j,k,s} W_{k,i,s}
        # Using einsum: 'ijs,jks,kis->ijk'
        T = np.einsum('ijs,jks,kis->ijk', U_3d, V_3d, W_3d)
        
        # Target is all 1s
        # Error is sum of squares
        return np.sum((T - 1.0)**2)
    
    # Check initial error
    err0 = error_func(x0)
    if err0 > 1e-6:
        # Fallback to schoolbook if initial solution is broken (should not happen)
        rank_school = n * m * p
        U_base = np.zeros((n * m, rank_school), dtype=float)
        V_base = np.zeros((m * p, rank_school), dtype=float)
        W_base = np.zeros((p * n, rank_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_base[i * m + j, col] = 1
                    V_base[j * p + k, col] = 1
                    W_base[k * n + i, col] = 1
                    col += 1
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
        
    # Optimization
    # We want to see if we can drive some columns to zero.
    # L-BFGS-B is good for this.
    
    best_U, best_V, best_W = U0, V0, W0
    best_err = err0
    
    # We might try to optimize to reduce the "effective rank" by making some columns small.
    # However, standard minimization of error doesn't encourage sparsity.
    # A common trick is to add a regularization term that penalizes the sum of squares of the columns?
    # No, that would distort the solution.
    # Instead, we can just run the optimizer. If it finds a solution with error ~0, 
    # we can then check if any columns are close to zero and remove them.
    
    # Let's run a short optimization.
    # Given the 120s limit, we need to be careful.
    # The problem size is 12*88 + 32*88 + 24*88 = 6080 variables.
    # L-BFGS-B should be fast enough for a few iterations.
    
    try:
        res = minimize(
            error_func, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
        )
        if res.success or res.fun < best_err:
            U_opt, V_opt, W_opt = unpack(res.x)
            if res.fun < 1e-8:
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_err = res.fun
    except Exception:
        pass
    
    # After optimization, check for rank reduction
    # If a column in U, V, W is all zeros (or very small), we can remove it.
    # We need to check if the remaining factors still decompose the tensor.
    
    # Heuristic for rank reduction:
    # Check each column s. If ||U[:,s]||, ||V[:,s]||, ||W[:,s]|| are all small, remove it.
    # Actually, for a valid decomposition, if one factor column is zero, the term contributes nothing.
    # So we can simply drop columns where the max abs value is < 1e-6.
    
    def reduce_rank(U, V, W):
        cols_to_keep = []
        for s in range(U.shape[1]):
            u_norm = np.max(np.abs(U[:, s]))
            v_norm = np.max(np.abs(V[:, s]))
            w_norm = np.max(np.abs(W[:, s]))
            if u_norm > 1e-5 and v_norm > 1e-5 and w_norm > 1e-5:
                cols_to_keep.append(s)
        
        if len(cols_to_keep) < U.shape[1]:
            U_new = U[:, cols_to_keep]
            V_new = V[:, cols_to_keep]
            W_new = W[:, cols_to_keep]
            return U_new, V_new, W_new
        return U, V, W

    U_final, V_final, W_final = reduce_rank(best_U, best_V, best_W)
    
    # Final verification
    def check_solution(U, V, W, rank):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    if check_solution(U_final, V_final, W_final, U_final.shape[1]):
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fallback to original rank 88
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

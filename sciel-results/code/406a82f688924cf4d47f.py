import time
import numpy as np
import itertools

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix.
    
    Strategy:
    1. Use a Greedy Rank-1 Subtraction approach with improved candidate generation.
    2. Initialize with structured candidates based on matrix multiplication patterns.
    3. At each step, select the rank-1 component that most reduces the residual tensor norm.
    4. Use local optimization to polish the solution.
    5. If no exact solution is found within time, fall back to the schoolbook rank-64 decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    dim_u = n * m  # 8
    dim_v = m * p  # 32
    dim_w = p * n  # 16
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Helper to check if entries are half-integers
    def is_half_integer(M, tol=1e-6):
        M2 = M * 2
        return np.all(np.abs(M2 - np.round(M2)) < tol)

    # Helper to compute tensor contraction efficiently
    def compute_contraction(u, v, w):
        u_r = u.reshape(n, m)
        v_r = v.reshape(m, p)
        w_r = w.reshape(p, n)
        return np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
    
    # Helper to compute inner product with residual
    def inner_product(residual, u, v, w):
        u_r = u.reshape(n, m)
        v_r = v.reshape(m, p)
        w_r = w.reshape(p, n)
        temp1 = np.einsum('ijk,ij,jk->ik', residual, u_r, v_r)
        return np.einsum('ik,ki->', temp1, w_r)
    
    # Generate structured candidate vectors based on matrix multiplication patterns
    def generate_structured_candidates():
        candidates = []
        
        # Pattern 1: Identity-like patterns
        for i in range(n):
            for j in range(m):
                u = np.zeros(dim_u)
                u[i * m + j] = 1
                for k in range(p):
                    v = np.zeros(dim_v)
                    v[j * p + k] = 1
                    w = np.zeros(dim_w)
                    w[k * n + i] = 1
                    candidates.append((u, v, w))
        
        # Pattern 2: Strassen-like combinations
        # For 2x4 * 4x8, we can try various combinations
        for i1 in range(n):
            for i2 in range(n):
                for j1 in range(m):
                    for j2 in range(m):
                        for k1 in range(p):
                            for k2 in range(p):
                                # Try sum/difference patterns
                                for sign1 in [1, -1]:
                                    for sign2 in [1, -1]:
                                        u = np.zeros(dim_u)
                                        u[i1 * m + j1] = sign1
                                        u[i2 * m + j2] = sign2
                                        
                                        v = np.zeros(dim_v)
                                        v[j1 * p + k1] = sign1
                                        v[j2 * p + k2] = sign2
                                        
                                        w = np.zeros(dim_w)
                                        w[k1 * n + i1] = sign1
                                        w[k2 * n + i2] = sign2
                                        
                                        if np.any(u != 0) and np.any(v != 0) and np.any(w != 0):
                                            candidates.append((u, v, w))
        
        return candidates
    
    # Greedy search
    def greedy_search(max_rank=30, time_limit=130.0):
        best_rank = max_rank + 1
        best_U, best_V, best_W = None, None, None
        
        # Generate initial candidates
        structured_candidates = generate_structured_candidates()
        
        for restart in range(20):
            if time.time() > time_limit:
                break
            
            residual = T.copy()
            U_cols = []
            V_cols = []
            W_cols = []
            
            # Shuffle candidates for diversity
            np.random.shuffle(structured_candidates)
            
            for r in range(max_rank):
                if time.time() > time_limit:
                    break
                
                # Find best u, v, w from candidates
                best_val = -np.inf
                best_u, best_v, best_w = None, None, None
                
                # Also try random candidates
                trials = min(100, max(50, 500 - r * 10))
                
                for trial in range(trials):
                    if trial < len(structured_candidates):
                        u, v, w = structured_candidates[trial]
                    else:
                        # Random candidate with sparse structure
                        u = np.zeros(dim_u)
                        v = np.zeros(dim_v)
                        w = np.zeros(dim_w)
                        
                        # Add 1-3 non-zero entries to each
                        for _ in range(np.random.randint(1, 4)):
                            idx = np.random.randint(0, dim_u)
                            u[idx] = np.random.choice([-1, 1])
                        
                        for _ in range(np.random.randint(1, 4)):
                            idx = np.random.randint(0, dim_v)
                            v[idx] = np.random.choice([-1, 1])
                        
                        for _ in range(np.random.randint(1, 4)):
                            idx = np.random.randint(0, dim_w)
                            w[idx] = np.random.choice([-1, 1])
                    
                    if np.all(u == 0) or np.all(v == 0) or np.all(w == 0):
                        continue
                    
                    # Compute inner product
                    val = inner_product(residual, u, v, w)
                    
                    if val > best_val:
                        best_val = val
                        best_u = u.copy()
                        best_v = v.copy()
                        best_w = w.copy()
                
                if best_u is None:
                    break
                
                # Calculate optimal alpha
                norm_sq = np.dot(best_u, best_u) * np.dot(best_v, best_v) * np.dot(best_w, best_w)
                if norm_sq < 1e-9:
                    continue
                
                alpha = best_val / norm_sq
                
                # Subtract the rank-1 component
                component = compute_contraction(best_u, best_v, best_w)
                residual = residual - alpha * component
                
                # Store the columns
                U_cols.append(best_u)
                V_cols.append(best_v)
                W_cols.append(alpha * best_w)
                
                # Check if residual is zero
                if np.allclose(residual, 0, atol=1e-9):
                    U_mat = np.column_stack(U_cols)
                    V_mat = np.column_stack(V_cols)
                    W_mat = np.column_stack(W_cols)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U_mat * 2) / 2
                    V_rounded = np.round(V_mat * 2) / 2
                    W_rounded = np.round(W_mat * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded):
                        if U_rounded.shape[1] < best_rank:
                            best_rank = U_rounded.shape[1]
                            best_U = U_rounded
                            best_V = V_rounded
                            best_W = W_rounded
                        break
                    elif is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat):
                        if verify(U_mat, V_mat, W_mat):
                            if U_mat.shape[1] < best_rank:
                                best_rank = U_mat.shape[1]
                                best_U = U_mat
                                best_V = V_mat
                                best_W = W_mat
                            break
            
            # End rank loop
            
            # Early exit if we found a good solution
            if best_rank <= 24:
                break
        
        # End restart loop
        
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        
        return None, None, None, max_rank + 1

    # Run greedy search
    U, V, W, rank = greedy_search(max_rank=28, time_limit=130.0)
    
    if U is not None:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    R = n * m * p
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

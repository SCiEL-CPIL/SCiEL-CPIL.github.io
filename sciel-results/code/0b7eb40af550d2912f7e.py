import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build target tensor T indices
    # T[a, b, w] = 1 if a=i*m+j, b=j*p+k, w=k*n+i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                indices.append((a, b, w))
    
    # Convert to numpy arrays for fast indexing
    idx_a = np.array([x[0] for x in indices], dtype=np.int32)
    idx_b = np.array([x[1] for x in indices], dtype=np.int32)
    idx_w = np.array([x[2] for x in indices], dtype=np.int32)
    
    # Precompute the full tensor for verification
    # T has shape (dim_A, dim_B, dim_C)
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.float64)
    for a, b, w in indices:
        T[a, b, w] = 1.0
    
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    # Fallback solution (Rank 42) using Strassen
    def get_fallback():
        U = [[0] * 42 for _ in range(n * m)]
        V = [[0] * 42 for _ in range(m * p)]
        W = [[0] * 42 for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            U[a_idx[0]][start_col + 0] += 1
            U[a_idx[3]][start_col + 0] += 1
            U[a_idx[2]][start_col + 1] += 1
            U[a_idx[3]][start_col + 1] += 1
            U[a_idx[0]][start_col + 2] += 1
            U[a_idx[3]][start_col + 3] += 1
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[1]][start_col + 4] += 1
            U[a_idx[2]][start_col + 5] += 1
            U[a_idx[0]][start_col + 5] -= 1
            U[a_idx[1]][start_col + 6] += 1
            U[a_idx[3]][start_col + 6] -= 1
            
            V[b_idx[0]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] += 1
            V[b_idx[0]][start_col + 1] += 1
            V[b_idx[1]][start_col + 2] += 1
            V[b_idx[3]][start_col + 2] -= 1
            V[b_idx[2]][start_col + 3] += 1
            V[b_idx[0]][start_col + 3] -= 1
            V[b_idx[3]][start_col + 4] += 1
            V[b_idx[0]][start_col + 5] += 1
            V[b_idx[1]][start_col + 5] += 1
            V[b_idx[2]][start_col + 6] += 1
            V[b_idx[3]][start_col + 6] += 1
            
            W[w_idx[0]][start_col + 0] += 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[0]][start_col + 4] -= 1
            W[w_idx[0]][start_col + 6] += 1
            
            W[w_idx[1]][start_col + 2] += 1
            W[w_idx[1]][start_col + 4] += 1
            
            W[w_idx[2]][start_col + 1] += 1
            W[w_idx[2]][start_col + 3] += 1
            
            W[w_idx[3]][start_col + 0] += 1
            W[w_idx[3]][start_col + 1] -= 1
            W[w_idx[3]][start_col + 2] += 1
            W[w_idx[3]][start_col + 5] += 1

        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]
        
        # Block 1: k=0 (b cols 0-5), i=0 (w rows 0,3,6,9) -> w indices 0, 2, 1, 3
        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)
        
        # Block 2: k=1 (b cols 6-11), i=0 (w rows 1,4,7,10) -> w indices 4, 6, 5, 7
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)
        
        # Block 3: k=2 (b cols 12-17), i=0 (w rows 2,5,8,11) -> w indices 8, 10, 9, 11
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)
        
        return U, V, W
    
    U_fb, V_fb, W_fb = get_fallback()
    
    # Verify fallback
    U_fb_arr = np.array(U_fb, dtype=np.float64)
    V_fb_arr = np.array(V_fb, dtype=np.float64)
    W_fb_arr = np.array(W_fb, dtype=np.float64)
    
    def verify_solution(U, V, W):
        """Verify that U, V, W exactly decompose the tensor"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        
        # Compute hat_T at indices
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        vals = hat_T[idx_a, idx_b, idx_w]
        
        # Check that all target entries are 1
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        
        # Check that non-target entries are 0 (approximately)
        # Create a mask of non-target entries
        full_hat = hat_T
        # Zero out target entries
        full_hat[idx_a, idx_b, idx_w] = 0
        
        # Check if remaining entries are close to 0
        max_err = np.max(np.abs(full_hat))
        return max_err < 1e-9
    
    # Verify fallback
    if not verify_solution(U_fb_arr, V_fb_arr, W_fb_arr):
        # Fallback is broken, return zeros (shouldn't happen)
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = 42
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Strategy: Splitting method
    # Try to extract rank-1 terms and solve the residual
    # The key insight: if we can find a rank-1 decomposition of a sub-tensor,
    # the residual may have lower rank
    
    # First, let's try a direct approach: use the fact that the tensor is sparse
    # and try to find a low-rank decomposition via structured initialization
    
    # The tensor M_{2,4,6} has a special structure. Let's exploit it.
    # For each (i, j, k), we have T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try to find a decomposition with rank < 42
    # We'll use a combination of:
    # 1. Structured initialization based on the tensor's algebraic structure
    # 2. Local optimization to polish
    
    # Key observation: The tensor can be viewed as a sum of 2*4*6 = 48 rank-1 terms
    # (the schoolbook method). Strassen reduces this to 42 by sharing computations.
    # Can we do better?
    
    # Let's try to use the "splitting" idea:
    # Identify a rank-1 term, subtract it, and see if the residual has lower rank.
    
    # For the 2x4x6 case, let's think about the structure more carefully.
    # A[i,j] * B[j,k] contributes to C[i,k]
    
    # Let's try a different approach: use the known lower bound.
    # The tensor rank of M_{n,m,p} is at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But this is a weak lower bound. A better bound comes from the slice rank or other methods.
    
    # For now, let's try to find a rank < 42 solution using optimization.
    
    def residual_full(theta, r):
        U = theta[:dim_A * r].reshape(dim_A, r)
        V = theta[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W = theta[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        
        # We want hat_T to match T
        # T is 1 at target indices, 0 elsewhere
        diff = hat_T - T
        return diff.flatten()
    
    def try_rank_optimization(r, num_restarts=5, use_structured_init=False):
        """Try to find a rank-r decomposition using optimization"""
        for attempt in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize
            if use_structured_init and attempt == 0:
                # Structured initialization: start from the schoolbook decomposition
                # and try to compress
                U0 = np.zeros((dim_A, r))
                V0 = np.zeros((dim_B, r))
                W0 = np.zeros((dim_C, r))
                
                # Schoolbook: for each (i,j,k), we have a rank-1 term
                # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
                # But we only have r terms, not 48
                # So we need to share terms
                
                # Let's try a random sparse initialization
                np.random.seed(42 + attempt)
                for t in range(r):
                    # Pick a random (i,j,k)
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    a = i * m + j
                    b = j * p + k
                    w = k * n + i
                    
                    # Add a small random perturbation to avoid symmetry
                    U0[a, t] = 1.0 + np.random.randn() * 0.1
                    V0[b, t] = 1.0 + np.random.randn() * 0.1
                    W0[w, t] = 1.0 + np.random.randn() * 0.1
            else:
                # Random initialization
                np.random.seed(r * 100 + attempt)
                U0 = np.random.randn(dim_A, r)
                V0 = np.random.randn(dim_B, r)
                W0 = np.random.randn(dim_C, r)
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Use scipy.optimize.least_squares
            try:
                import scipy.optimize as so
                
                def res_func(t):
                    return residual_full(t, r)
                
                # Use BFGS for better convergence
                res = so.least_squares(res_func, theta0, method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14)
                
                if res.cost < 1e-10:
                    U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                    V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    # Round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    if verify_solution(U_int, V_int, W_int):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    else:
                        # Try with integer rounding
                        U_int2 = np.round(U_sol)
                        V_int2 = np.round(V_sol)
                        W_int2 = np.round(W_sol)
                        if verify_solution(U_int2, V_int2, W_int2):
                            return {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
            except Exception as e:
                pass
        
        return None
    
    # Try ranks from 24 to 41
    for r in range(24, 42):
        if time.time() > deadline:
            break
        
        # Try with structured initialization first
        res = try_rank_optimization(r, num_restarts=3, use_structured_init=True)
        if res:
            best_rank = r
            best_result = res
            break
        
        # Try with random initialization
        res = try_rank_optimization(r, num_restarts=5, use_structured_init=False)
        if res:
            best_rank = r
            best_result = res
            break
    
    return best_result

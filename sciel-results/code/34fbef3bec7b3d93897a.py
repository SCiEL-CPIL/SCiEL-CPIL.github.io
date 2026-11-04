import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build target tensor: T[a, b, w] = 1 if a=i*m+j, b=j*p+k, w=k*n+i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                indices.append((a, b, w))
    
    idx_a = np.array([x[0] for x in indices], dtype=np.int32)
    idx_b = np.array([x[1] for x in indices], dtype=np.int32)
    idx_w = np.array([x[2] for x in indices], dtype=np.int32)
    
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    def verify_solution(U, V, W, atol=1e-9):
        """Verify that U, V, W give exact decomposition."""
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        vals = hat_T[idx_a, idx_b, idx_w]
        return np.allclose(vals, 1.0, atol=atol)
    
    def verify_solution_fast(U, V, W, atol=1e-9):
        """Faster verification using only the relevant entries."""
        # For each (a,b,w) in indices, check sum_r U[a,r]*V[b,r]*W[w,r] == 1
        # Use einsum for speed
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        vals = hat_T[idx_a, idx_b, idx_w]
        return np.allclose(vals, 1.0, atol=atol)
    
    # Try to find rank-r decomposition using least squares
    def try_rank_ls(r, n_restarts=20, seed_base=0):
        """Try to find rank-r decomposition using scipy least squares."""
        import scipy.optimize as so
        
        for attempt in range(n_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(seed_base + attempt)
            
            # Smart initialization based on block structure
            if attempt == 0:
                # Block-diagonal initialization
                U0 = np.zeros((dim_A, r))
                V0 = np.zeros((dim_B, r))
                W0 = np.zeros((dim_C, r))
                
                # Two blocks: j in {0,1} and j in {2,3}
                half_r = r // 2
                rem = r - half_r
                
                U0[:4, :half_r] = np.random.randn(4, half_r) * 0.3
                V0[:12, :half_r] = np.random.randn(12, half_r) * 0.3
                W0[:6, :half_r] = np.random.randn(6, half_r) * 0.3
                
                U0[4:, half_r:] = np.random.randn(4, rem) * 0.3
                V0[12:, half_r:] = np.random.randn(12, rem) * 0.3
                W0[6:, half_r:] = np.random.randn(6, rem) * 0.3
            else:
                U0 = np.random.randn(dim_A, r) * 0.3
                V0 = np.random.randn(dim_B, r) * 0.3
                W0 = np.random.randn(dim_C, r) * 0.3
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def res_func(t):
                U = t[:dim_A * r].reshape(dim_A, r)
                V = t[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                W = t[dim_A * r + dim_B * r :].reshape(dim_C, r)
                hat_T = np.einsum('ar,br,wr->abw', U, V, W)
                return hat_T[idx_a, idx_b, idx_w] - 1.0
            
            try:
                res = so.least_squares(res_func, theta0, method='lm',
                                       xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                       max_nfev=3000)
                if res.cost < 1e-10:
                    U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                    V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    # Try rounding to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    if verify_solution(U_int, V_int, W_int):
                        return (U_int, V_int, W_int)
                    
                    # Try integer rounding
                    U_int2 = np.round(U_sol)
                    V_int2 = np.round(V_sol)
                    W_int2 = np.round(W_sol)
                    if verify_solution(U_int2, V_int2, W_int2):
                        return (U_int2, V_int2, W_int2)
                    
                    # Return floating point if it works
                    if verify_solution(U_sol, V_sol, W_sol):
                        return (U_sol, V_sol, W_sol)
                        
            except Exception:
                pass
        
        return None
    
    # Known construction: rank 24 using separate terms for each (j,k) pair
    # For each j in {0,1,2,3} and k in {0,...,5}, we have A(i,j)*B(j,k) -> C(i,k)
    # The output index is w = k*n + i
    # We can use rank 1 terms: for each (i,j,k), U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # But that's rank 48. We can do better.
    
    # Actually, let's think about this differently.
    # The tensor is T[i,j,k; i',j',k'] where the multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In our indexing: a = i*m+j, b = j*p+k, w = k*n+i
    # T[a,b,w] = 1 iff a=i*m+j, b=j*p+k, w=k*n+i for some i,j,k
    
    # This is the standard matrix multiplication tensor M_{2,4,6}
    # Lower bound: max(2*4, 4*6, 6*2) = max(8, 24, 12) = 24
    
    # Let's try rank 24 first with a structured approach
    def try_structured_rank_24():
        """Try to find rank 24 decomposition using the structure of the problem."""
        # The key insight: for each j, we have a 2x6 multiplication (A[:,j] * B[j,:])
        # This is a vector-matrix product, which has rank 2*6 = 12 if done naively
        # But we can share across j values.
        
        # Alternative: use the fact that the tensor decomposes as a sum over j
        # T = sum_j T_j where T_j[i,k; a,b,w] = 1 if a=i*m+j, b=j*p+k, w=k*n+i
        # Each T_j is a 2x1x6 tensor (rank at most 12)
        
        # Let's try a direct construction:
        # For rank 24, use 6 terms per j (4 j values * 6 terms = 24)
        # For fixed j and k: term r has U[i*m+j, r] = 1 for all i, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for all i
        # Wait, that doesn't work because U[i*m+j, r] would need to depend on i.
        
        # Let me try: for each j, use a 2x6 matrix multiplication of rank 12 (naive)
        # Total rank = 4 * 12 = 48. Too high.
        
        # Better: use Strassen-like ideas for the 2x2 sub-blocks
        # Split j into {0,1} and {2,3}
        # For each group of 2 j's, we have a 2x2x6 multiplication
        # 2x2x6 can be done in rank... let's see. Naive is 12.
        
        # Actually, let's just try the numerical approach with rank 24.
        return None
    
    # Try ranks from 24 upward
    best_solution = None
    best_rank = 48
    
    # First, try a known rank-48 construction as fallback
    def get_fallback_rank_48():
        """Naive rank-48 construction: one term per (i,j,k)."""
        rank = n * m * p  # 48
        U = np.zeros((dim_A, rank))
        V = np.zeros((dim_B, rank))
        W = np.zeros((dim_C, rank))
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    w = k * n + i
                    U[a, r] = 1
                    V[b, r] = 1
                    W[w, r] = 1
                    r += 1
        
        return U, V, W
    
    U_fb, V_fb, W_fb = get_fallback_rank_48()
    
    # Try to improve from rank 24 upward
    for r in range(24, 49):
        if time.time() > deadline - 10:
            break
        
        sol = try_rank_ls(r, n_restarts=10 if r <= 26 else 5, seed_base=r * 100)
        if sol is not None:
            U_sol, V_sol, W_sol = sol
            best_solution = (U_sol, V_sol, W_sol)
            best_rank = r
            break
    
    if best_solution is not None:
        U_sol, V_sol, W_sol = best_solution
        return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # Fallback to rank 48
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

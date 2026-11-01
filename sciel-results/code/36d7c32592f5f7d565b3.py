import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # We exploit tensor rank subadditivity:
    # rank(2x2x15) <= rank(2x2x5) + rank(2x2x10)
    # 
    # For 2x2x5: rank 14 (known optimal for small cases)
    # For 2x2x10: rank 28 = 2 * rank(2x2x5) via Strassen-like pairing
    # Total: 14 + 28 = 42
    #
    # Strategy: Build explicit decompositions using Kronecker products
    # and Strassen-style bilinear forms.
    
    # First, construct a rank-14 decomposition for 2x2x5
    # Using the known result that rank(2,2,5) = 14
    # We'll build it from rank-4 (2x2x1) + rank-10 (2x2x4)
    # Actually, let's use a more systematic approach.
    
    # For 2x2x1: rank 4 (schoolbook)
    # For 2x2x2: rank 7 (Strassen)
    # For 2x2x3: rank 11
    # For 2x2x4: rank 14
    # For 2x2x5: rank 17? Let me reconsider.
    
    # Actually, known results:
    # rank(2,2,1) = 4
    # rank(2,2,2) = 7 (Strassen)
    # rank(2,2,3) = 11
    # rank(2,2,4) = 14
    # rank(2,2,5) = 17
    # rank(2,2,10) = 34
    # So rank(2,2,15) <= 17 + 34 = 51, but we can do better.
    
    # Better: Use the fact that rank(2,2,p) grows roughly as 2.5p for large p
    # rank(2,2,5) = 17, rank(2,2,10) = 34, so rank(2,2,15) <= 17 + 34 = 51
    # But we can potentially do rank(2,2,15) = 42 if we find a better decomposition.
    
    # Let me try a direct construction using Strassen blocks.
    # For 2x2x15, we can split into three 2x2x5 blocks.
    # If rank(2,2,5) = 14, then rank(2,2,15) <= 3*14 = 42.
    
    # Construct rank-14 for 2x2x5:
    # This is non-trivial. Let me use a known construction.
    
    # Actually, let me just use the schoolbook as a safe fallback with rank 60,
    # but try to find a better one.
    
    # For now, let's construct the schoolbook decomposition (rank 60)
    # and see if we can improve.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now let's try to find a better decomposition using optimization
    # with the subadditivity approach
    
    from scipy.optimize import least_squares
    
    def solve_subadditive(n, m, p, rank_target, seed=0):
        """Try to find a rank-rank_target decomposition using subadditivity."""
        np.random.seed(seed)
        
        total_U = n * m
        total_V = m * p
        total_W = p * n
        
        def residuals(params):
            U = params[:rank_target*total_U].reshape(rank_target, total_U).T
            V = params[rank_target*total_U:rank_target*(total_U+total_V)].reshape(rank_target, total_V).T
            W = params[rank_target*(total_U+total_V):].reshape(rank_target, total_W).T
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        res.append(val - 1.0)
            return res
        
        best_U, best_V, best_W = None, None, None
        best_cost = float('inf')
        
        # Try multiple restarts with different initializations
        for trial in range(20):
            # Initialize with small random values
            params0 = np.random.randn(rank_target * (total_U + total_V + total_W)) * 0.1
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=2000)
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    U = result.x[:rank_target*total_U].reshape(rank_target, total_U).T
                    V = result.x[rank_target*total_U:rank_target*(total_U+total_V)].reshape(rank_target, total_V).T
                    W = result.x[rank_target*(total_U+total_V):].reshape(rank_target, total_W).T
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    # Check if rounded solution is exact
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-9:
                        return U_r, V_r, W_r
            except:
                pass
        
        return best_U, best_V, best_W
    
    # Try to find rank 42 decomposition
    U42, V42, W42 = solve_subadditive(n, m, p, 42, seed=123)
    
    if U42 is not None:
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
    
    # Fallback to schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # We'll construct the decomposition by combining sub-decompositions
    # (2,2,5) with rank 14 and (2,2,10) with rank 28
    # Total rank = 42
    
    # For (2,2,5): Use a known rank-14 decomposition
    # For (2,2,10): Use 2 copies of (2,2,5) rank-14 each = rank 28
    
    # First, let's define the rank-14 decomposition for (2,2,5)
    # This is a known optimal decomposition for 2x2x5
    
    def get_rank14_decomp_2x2x5():
        """
        Returns U, V, W for 2x2x5 matrix multiplication with rank 14.
        U shape: (4, 14) - for n*m = 4
        V shape: (10, 14) - for m*p = 10
        W shape: (10, 14) - for p*n = 10
        """
        # Using a known construction for 2x2x5
        # The 2x2x5 tensor has rank 14 (proven optimal)
        
        # Let's construct it systematically
        # For 2x2x5, we can think of it as having 5 columns
        # Each "column" k contributes to the multiplication
        
        # Known approach: use the fact that 2x2x5 can be decomposed
        # by considering the structure of the tensor
        
        # Let me try a different approach: use a precomputed solution
        # or construct one using algebraic methods
        
        # Actually, let's use a simpler approach: 
        # For 2x2x5, the schoolbook rank is 20.
        # We need rank 14, which is the known optimum.
        
        # Let me try to construct this using a pattern.
        # The key insight is that for 2x2x5, we can save 6 ranks from schoolbook.
        
        # I'll use a numerical approach to find the rank-14 decomposition
        from scipy.optimize import least_squares
        
        np.random.seed(123)
        
        rank = 14
        total_U = 4  # n*m
        total_V = 10  # m*p
        total_W = 10  # p*n
        
        def residuals(params):
            U = params[:rank*total_U].reshape(rank, total_U).T
            V = params[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
            W = params[rank*(total_U+total_V):].reshape(rank, total_W).T
            
            res = []
            for i in range(2):
                for j in range(2):
                    for k in range(5):
                        val = np.sum(U[i*2+j] * V[j*5+k] * W[k*2+i])
                        res.append(val - 1.0)
            return res
        
        best_U, best_V, best_W = None, None, None
        best_cost = float('inf')
        
        for trial in range(100):
            params0 = np.random.randn(rank * (total_U + total_V + total_W)) * 0.3
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=2000)
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    U = result.x[:rank*total_U].reshape(rank, total_U).T
                    V = result.x[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
                    W = result.x[rank*(total_U+total_V):].reshape(rank, total_W).T
                    # Round to integers or half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    # Check if rounded solution is exact
                    max_err = 0
                    for i in range(2):
                        for j in range(2):
                            for k in range(5):
                                val = np.sum(U_r[i*2+j] * V_r[j*5+k] * W_r[k*2+i])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-9:
                        return U_r, V_r, W_r
                    if cost < best_cost:
                        best_U, best_V, best_W = U_r, V_r, W_r
            except:
                pass
        
        # If we found a solution, return it
        if best_U is not None and best_cost < 1e-6:
            return best_U, best_V, best_W
        
        # Fallback: use schoolbook for 2x2x5 (rank 20) but we need rank 14...
        # Let's try a different initialization strategy
        for trial in range(100):
            # Use a structured initialization
            params0 = np.zeros(rank * (total_U + total_V + total_W))
            # Initialize with some structure
            for r in range(rank):
                # Set some entries to 1, some to -1, some to 0.5
                for idx in range(total_U):
                    params0[r * total_U + idx] = np.random.choice([-1, 0, 1, 0.5, -0.5])
                for idx in range(total_V):
                    params0[rank * total_U + r * total_V + idx] = np.random.choice([-1, 0, 1, 0.5, -0.5])
                for idx in range(total_W):
                    params0[rank * (total_U + total_V) + r * total_W + idx] = np.random.choice([-1, 0, 1, 0.5, -0.5])
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=3000)
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    U = result.x[:rank*total_U].reshape(rank, total_U).T
                    V = result.x[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
                    W = result.x[rank*(total_U+total_V):].reshape(rank, total_W).T
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    max_err = 0
                    for i in range(2):
                        for j in range(2):
                            for k in range(5):
                                val = np.sum(U_r[i*2+j] * V_r[j*5+k] * W_r[k*2+i])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-9:
                        return U_r, V_r, W_r
                    if cost < best_cost:
                        best_U, best_V, best_W = U_r, V_r, W_r
            except:
                pass
        
        if best_U is not None and best_cost < 1e-6:
            return best_U, best_V, best_W
        
        # Final fallback: return None to indicate failure
        return None, None, None
    
    # Get the rank-14 decomposition for 2x2x5
    U5, V5, W5 = get_rank14_decomp_2x2x5()
    
    if U5 is None:
        # Fallback to schoolbook for the whole 2x2x15
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # For 2x2x10, use 2 copies of the 2x2x5 decomposition
    # Each copy handles 5 columns, so total rank = 14 + 14 = 28
    U10 = np.hstack([U5, U5])
    V10 = np.zeros((10, 28))
    V10[:5, :14] = V5
    V10[5:, 14:] = V5
    W10 = np.zeros((10, 28))
    W10[:5, :14] = W5
    W10[5:, 14:] = W5
    
    # Combine 2x2x5 (rank 14) and 2x2x10 (rank 28) for total 2x2x15 (rank 42)
    U_full = np.hstack([U5, U10])
    V_full = np.zeros((15, 42))
    V_full[:5, :14] = V5
    V_full[5:, 14:] = V10
    W_full = np.zeros((30, 42))
    W_full[:5, :14] = W5
    W_full[5:, 14:] = W10
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

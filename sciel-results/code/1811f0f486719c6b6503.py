import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 3, 3, 8
    
    # Try to find a low-rank decomposition
    # Start with rank 21 (padding approach: 7+7+7)
    # But we need the explicit rank-7 decomposition for 3x3
    
    # For now, let's try a numerical approach
    # Initialize with schoolbook and try to reduce
    
    # Actually, let's try the padding approach with a specific rank-7 decomposition
    # I'll use a known one
    
    # Rank-7 decomposition for 3x3 matmul:
    # This is based on the work of Landsberg (1996)
    # One explicit decomposition:
    
    # Let me try to use a simple approach: 
    # For 3x3 x 3x8, split into 3x3x3 + 3x3x3 + 3x3x2
    # Use rank-7 for each 3x3x3, and rank-7 for 3x3x2 (via padding)
    # Total: 21
    
    # But I need the explicit rank-7 decomposition!
    
    # Let me try to find one numerically
    
    # The tensor T for 3x3x3 matmul:
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We want: T = sum_t U_t (3x3) (o) V_t (3x3) (o) W_t (3x3)
    # where (o) is the Kronecker product
    
    # In flattened form:
    # T_flat is 9x9x9
    # U_flat is 9x7, V_flat is 9x7, W_flat is 9x7
    
    # The condition: for all i,k:
    # 1 = sum_t W[k*3+i, t] * (sum_j U[i*3+j, t] * V[j*3+k, t])
    
    # This is a system of 9 equations with 7*9*3 = 189 unknowns
    # But the equations are nonlinear
    
    # Let me try to solve this using scipy.optimize
    
    from scipy.optimize import least_squares
    
    def residual_3x3(params):
        # params: 7*9 + 7*9 + 7*9 = 189 values
        # Reshape to U, V, W each 9x7
        U = params[0:63].reshape(9, 7)
        V = params[63:126].reshape(9, 7)
        W = params[126:189].reshape(9, 7)
        
        res = []
        for i in range(3):
            for k in range(3):
                val = 0
                for t in range(7):
                    inner = 0
                    for j in range(3):
                        inner += U[i*3+j, t] * V[j*3+k, t]
                    val += W[k*3+i, t] * inner
                res.append(val - 1)
        return np.array(res)
    
    # Initial guess: use schoolbook (rank 9) and zero out 2 terms
    # Or use random initialization
    np.random.seed(42)
    params0 = np.random.randn(189) * 0.1
    
    result = least_squares(residual_3x3, params0, method='lm', max_nfev=10000)
    
    if result.success or np.linalg.norm(result.fun) < 1e-6:
        U7 = result.x[0:63].reshape(9, 7)
        V7 = result.x[63:126].reshape(9, 7)
        W7 = result.x[126:189].reshape(9, 7)
        
        # Round to integers/half-integers
        U7 = np.round(U7 * 2) / 2
        V7 = np.round(V7 * 2) / 2
        W7 = np.round(W7 * 2) / 2
        
        # Verify
        max_err = 0
        for i in range(3):
            for k in range(3):
                val = 0
                for t in range(7):
                    inner = 0
                    for j in range(3):
                        inner += U7[i*3+j, t] * V7[j*3+k, t]
                    val += W7[k*3+i, t] * inner
                max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-9:
            # Great! We have a valid rank-7 decomposition for 3x3
            # Now extend to 3x3x8
            # ...
            pass
    
    # If the above doesn't work, fall back to schoolbook
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

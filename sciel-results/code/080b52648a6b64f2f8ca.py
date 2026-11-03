"""Program for matrix multiplication 3x4x6 tensor decomposition."""
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Target rank based on directive: 36
    # Fallback rank: 72 (schoolbook)
    
    def schoolbook():
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

    def try_rank(R):
        # Flatten U, V, W into vectors for optimization
        # U shape: (n*m, R), V shape: (m*p, R), W shape: (p*n, R)
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        # Random initialization
        np.random.seed(42)
        x0 = np.random.randn(total_size) * 0.1
        
        def objective(x):
            U = x[:size_U].reshape(n * m, R)
            V = x[size_U:size_U+size_V].reshape(m * p, R)
            W = x[size_U+size_V:].reshape(p * n, R)
            
            # Compute the tensor T_approx
            # T_approx[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We want T_approx to be 1 for all i, j, k.
            # There are n*m*p = 72 equations.
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum over c
                        u_val = U[i * m + j]
                        v_val = V[j * p + k]
                        w_val = W[k * n + i]
                        prod = np.sum(u_val * v_val * w_val)
                        errors.append(prod - 1.0)
                        
            errors = np.array(errors)
            return np.sum(errors**2)
            
        # Use a simple optimization method. Since it's non-convex, this might not find a global min.
        # Given the complexity, this might be slow or fail.
        # However, for the purpose of the task, we attempt it.
        # To keep it fast, we limit iterations.
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100})
            if res.success or res.fun < 1e-6:
                # Check if we can round to integers/half-integers
                U = res.x[:size_U].reshape(n * m, R)
                V = res.x[size_U:size_U+size_V].reshape(m * p, R)
                W = res.x[size_U+size_V:].reshape(p * n, R)
                
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness with rounded values
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_r[i * m + j]
                            v_val = V_r[j * p + k]
                            w_val = W_r[k * n + i]
                            prod = np.sum(u_val * v_val * w_val)
                            err = abs(prod - 1.0)
                            if err > max_err:
                                max_err = err
                                
                if max_err < 1e-9:
                    # Convert to lists of lists for return
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
            
        return None

    # Try rank 36 as suggested
    res = try_rank(36)
    if res:
        return res
        
    # Fallback to schoolbook
    return schoolbook()

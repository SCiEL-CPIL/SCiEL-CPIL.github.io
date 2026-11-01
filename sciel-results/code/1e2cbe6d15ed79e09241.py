"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def compute_tensor(U, V, W, R):
        """Compute T[i,j,k,l] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+l,r]"""
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i,j,k,l] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+l])
        return T
    
    # Target tensor: T[i,j,k,l] = 1 if j==k else 0
    T_target = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    T_target[i,j,k,l] = 1.0 if j == k else 0.0
    
    def verify_integer_solution(U, V, W):
        """Verify that integer-valued U, V, W exactly decompose the tensor."""
        T_check = compute_tensor(U.astype(float), V.astype(float), W.astype(float), U.shape[1])
        return np.max(np.abs(T_check - T_target)) < 1e-8
    
    def try_rank_gradient_descent(R, seed=42, max_iter=10000, lr=0.01):
        """Try to find a decomposition of rank R using gradient descent."""
        np.random.seed(seed)
        
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        for iteration in range(max_iter):
            T_current = compute_tensor(U, V, W, R)
            error = T_current - T_target
            loss = np.sum(error**2)
            
            if loss < 1e-12:
                break
            
            # Compute gradients using vectorized operations
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(m):
                        for l in range(p):
                            e = error[i,j,k,l]
                            dU[i*m+j] += 2 * e * V[j*p+k] * W[k*n+l]
                            dV[j*p+k] += 2 * e * U[i*m+j] * W[k*n+l]
                            dW[k*n+l] += 2 * e * U[i*m+j] * V[j*p+k]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        T_final = compute_tensor(U, V, W, R)
        max_err = np.max(np.abs(T_final - T_target))
        
        if max_err < 1e-6:
            return U, V, W, max_err
        return None, None, None, max_err
    
    def round_to_integers(U, V, W):
        """Try to round the factors to integers or half-integers and verify."""
        all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Try integer rounding with various scales
        for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            scaled_U = U * scale
            scaled_V = V * scale
            scaled_W = W * scale
            
            U_int = np.round(scaled_U)
            V_int = np.round(scaled_V)
            W_int = np.round(scaled_W)
            
            U_err = np.max(np.abs(U_int - scaled_U))
            V_err = np.max(np.abs(V_int - scaled_V))
            W_err = np.max(np.abs(W_int - scaled_W))
            
            max_err_scaled = max(U_err, V_err, W_err)
            if max_err_scaled < 1e-6:
                # Verify the integer solution
                if verify_integer_solution(U_int, V_int, W_int):
                    return U_int, V_int, W_int, "int"
        
        # Try half-integer rounding
        for scale in [1, 2, 4, 8, 16, 32, 64]:
            scaled_U = U * scale
            scaled_V = V * scale
            scaled_W = W * scale
            
            # Round to nearest half-integer: 2*x should be close to integer
            U_half = np.round(scaled_U * 2) / 2
            V_half = np.round(scaled_V * 2) / 2
            W_half = np.round(scaled_W * 2) / 2
            
            U_err = np.max(np.abs(U_half - scaled_U))
            V_err = np.max(np.abs(V_half - scaled_V))
            W_err = np.max(np.abs(W_half - scaled_W))
            
            max_err_half = max(U_err, V_err, W_err)
            if max_err_half < 1e-6:
                # Verify the half-integer solution
                U_float = U_half.astype(float)
                V_float = V_half.astype(float)
                W_float = W_half.astype(float)
                T_check = compute_tensor(U_float, V_float, W_float, U.shape[1])
                if np.max(np.abs(T_check - T_target)) < 1e-8:
                    return U_half, V_half, W_half, "half"
        
        return None, None, None, None
    
    # Try ranks from 7 upward (minimum possible is at least 7 for 2x2x3)
    for R in range(7, 13):
        # Try multiple seeds
        found = False
        for seed in range(20):
            U, V, W, err = try_rank_gradient_descent(R, seed=seed, max_iter=5000, lr=0.02)
            if U is not None:
                U_int, V_int, W_int, kind = round_to_integers(U, V, W)
                if U_int is not None:
                    if kind == "int":
                        U_list = [[int(x) for x in row] for row in U_int]
                        V_list = [[int(x) for x in row] for row in V_int]
                        W_list = [[int(x) for x in row] for row in W_int]
                    else:  # half
                        U_list = [[float(x) for x in row] for row in U_int]
                        V_list = [[float(x) for x in row] for row in V_int]
                        W_list = [[float(x) for x in row] for row in W_int]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook with rank = n*m*p = 12
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
# EVOLVE-BLOCK-END

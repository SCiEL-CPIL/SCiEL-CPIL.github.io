"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product
from fractions import Fraction
from sympy import Matrix, symbols, solve, init_printing

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor for (n,m) x (m,p) -> (n,p)
    # T[i,j,k,l] = delta_{jk} where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r] = delta_{jk}
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        T_check = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_check[i,j,k,l] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+l])
        
        T_target = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_target[i,j,k,l] = 1.0 if j == k else 0.0
        
        return np.max(np.abs(T_check - T_target)) < 1e-10
    
    def try_rank_sympy(R):
        """Try to find a decomposition of rank R using sympy exact arithmetic."""
        # Create symbolic variables
        U_vars = [symbols(f'U_{i}_{r}') for i in range(n*m) for r in range(R)]
        V_vars = [symbols(f'V_{i}_{r}') for i in range(m*p) for r in range(R)]
        W_vars = [symbols(f'W_{i}_{r}') for i in range(p*n) for r in range(R)]
        
        U_mat = Matrix(U_vars).reshape(n*m, R)
        V_mat = Matrix(V_vars).reshape(m*p, R)
        W_mat = Matrix(W_vars).reshape(p*n, R)
        
        # Build equations: for each (i,j,k,l), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+l,r] = delta_{jk}
        equations = []
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        expr = 0
                        for r in range(R):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + l
                            expr += U_mat[u_idx, r] * V_mat[v_idx, r] * W_mat[w_idx, r]
                        target = 1 if j == k else 0
                        equations.append(expr - target)
        
        # This system is too large to solve directly for R >= 8
        # Instead, use a heuristic: fix some variables and solve for others
        # Or use random initialization with gradient descent in exact arithmetic
        
        # Alternative approach: use the known rank-8 solution structure
        # For 2x2 * 2x3, the minimum rank is 8 (schoolbook)
        # Let's try to find a rank-8 solution
        
        if R == 8:
            # Try a structured approach
            # The schoolbook solution has rank 12 (n*m*p = 2*2*3 = 12)
            # But there might be a rank-8 solution
            
            # Let's try random initialization with gradient descent using exact arithmetic
            # This is complex, so let's try a simpler approach first
            
            pass
        
        # For now, return None to indicate failure
        return None
    
    def try_rank_numeric(R, max_iterations=10000):
        """Try to find a decomposition of rank R using numeric gradient descent."""
        np.random.seed(42)
        
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        def compute_tensor(U, V, W):
            T = np.zeros((n, m, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(m):
                        for l in range(p):
                            T[i,j,k,l] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+l])
            return T
        
        T_target = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T_target[i,j,k,l] = 1.0 if j == k else 0.0
        
        # Adam optimizer
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        lr = 0.05
        for t in range(1, max_iterations + 1):
            T_current = compute_tensor(U, V, W)
            error = T_current - T_target
            loss = np.sum(error**2)
            
            if loss < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(m):
                        for l in range(p):
                            e = error[i,j,k,l]
                            for r in range(R):
                                dU[i*m+j, r] += 2 * e * V[j*p+k, r] * W[k*n+l, r]
                                dV[j*p+k, r] += 2 * e * U[i*m+j, r] * W[k*n+l, r]
                                dW[k*n+l, r] += 2 * e * U[i*m+j, r] * V[j*p+k, r]
            
            # Adam update
            m_U = beta1 * m_U + (1 - beta1) * dU
            v_U = beta2 * v_U + (1 - beta2) * dU**2
            m_hat_U = m_U / (1 - beta1**t)
            v_hat_U = v_U / (1 - beta2**t)
            U -= lr * m_hat_U / (np.sqrt(v_hat_U) + epsilon)
            
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * dV**2
            m_hat_V = m_V / (1 - beta1**t)
            v_hat_V = v_V / (1 - beta2**t)
            V -= lr * m_hat_V / (np.sqrt(v_hat_V) + epsilon)
            
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * dW**2
            m_hat_W = m_W / (1 - beta1**t)
            v_hat_W = v_W / (1 - beta2**t)
            W -= lr * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
        
        T_final = compute_tensor(U, V, W)
        max_err = np.max(np.abs(T_final - T_target))
        
        if max_err < 1e-8:
            # Try to convert to exact integers/half-integers
            all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Try different scales
            for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                scaled_U = U * scale
                scaled_V = V * scale
                scaled_W = W * scale
                
                U_int = np.round(scaled_U)
                V_int = np.round(scaled_V)
                W_int = np.round(scaled_W)
                
                U_err = np.max(np.abs(U_int - scaled_U))
                V_err = np.max(np.abs(V_int - scaled_V))
                W_err = np.max(np.abs(W_int - scaled_W))
                
                if max(U_err, V_err, W_err) < 1e-6:
                    # Verify exact solution
                    U_final = U_int.astype(float)
                    V_final = V_int.astype(float)
                    W_final = W_int.astype(float)
                    
                    if verify_decomposition(U_final, V_final, W_final):
                        U_list = [[int(x) for x in row] for row in U_final]
                        V_list = [[int(x) for x in row] for row in V_final]
                        W_list = [[int(x) for x in row] for row in W_final]
                        return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try half-integers
            for scale in [1, 2, 4, 8, 16, 32, 64]:
                scaled_U = U * scale * 2
                scaled_V = V * scale * 2
                scaled_W = W * scale * 2
                
                U_half = np.round(scaled_U) / 2
                V_half = np.round(scaled_V) / 2
                W_half = np.round(scaled_W) / 2
                
                U_err = np.max(np.abs(U_half - U * scale))
                V_err = np.max(np.abs(V_half - V * scale))
                W_err = np.max(np.abs(W_half - W * scale))
                
                if max(U_err, V_err, W_err) < 1e-6:
                    U_final = U_half
                    V_final = V_half
                    W_final = W_half
                    
                    if verify_decomposition(U_final, V_final, W_final):
                        U_list = [[float(x) for x in row] for row in U_final]
                        V_list = [[float(x) for x in row] for row in V_final]
                        W_list = [[float(x) for x in row] for row in W_final]
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 8 upward
    for R in range(8, 13):
        result = try_rank_numeric(R)
        if result is not None:
            return result
    
    # Fallback: return schoolbook (rank 12)
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

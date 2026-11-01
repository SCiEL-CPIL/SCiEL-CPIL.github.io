"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = 10  # Fixed bond dimension (rank)
    
    # Target tensor: T[i,j,k,l] = delta_{jk}
    T_target = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    T_target[i,j,k,l] = 1.0 if j == k else 0.0
    
    def compute_tensor(U, V, W):
        """Compute T[i,j,k,l] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+l,r]"""
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        T[i,j,k,l] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+l])
        return T
    
    def verify_solution(U, V, W, tol=1e-10):
        """Verify that the factor matrices produce the target tensor."""
        T_check = compute_tensor(U, V, W)
        max_err = np.max(np.abs(T_check - T_target))
        return max_err < tol, max_err
    
    # Try multiple random initializations with gradient descent
    best_result = None
    best_err = float('inf')
    
    for trial in range(50):
        np.random.seed(1000 + trial)
        
        # Initialize with random values
        U = np.random.randn(n * m, R) * 1.0
        V = np.random.randn(m * p, R) * 1.0
        W = np.random.randn(p * n, R) * 1.0
        
        # Adam-like optimizer with momentum
        U_mom = np.zeros_like(U)
        V_mom = np.zeros_like(V)
        W_mom = np.zeros_like(W)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        t = 0
        lr = 0.01
        
        for iteration in range(3000):
            t += 1
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
            U_mom = beta1 * U_mom + (1 - beta1) * dU
            V_mom = beta1 * V_mom + (1 - beta1) * dV
            W_mom = beta1 * W_mom + (1 - beta1) * dW
            
            U_hat = U_mom / (1 - beta1**t)
            V_hat = V_mom / (1 - beta1**t)
            W_hat = W_mom / (1 - beta1**t)
            
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            U -= lr_t * U_hat / (np.sqrt(1 - beta2**t) + epsilon)
            V -= lr_t * V_hat / (np.sqrt(1 - beta2**t) + epsilon)
            W -= lr_t * W_hat / (np.sqrt(1 - beta2**t) + epsilon)
        
        T_final = compute_tensor(U, V, W)
        max_err = np.max(np.abs(T_final - T_target))
        
        if max_err < best_err:
            best_err = max_err
            best_result = (U.copy(), V.copy(), W.copy())
        
        if max_err < 1e-8:
            break
    
    if best_result is None:
        # Fallback to schoolbook
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
    
    U, V, W = best_result
    
    # Now try to convert to exact integer or half-integer values
    # Strategy: find a scaling factor s such that s*U, s*V, s*W are all close to integers
    # or s*U, s*V, s*W are all close to half-integers
    
    def try_scale_to_integers(U, V, W, scales):
        """Try to find a scale that makes all entries integers."""
        for scale in scales:
            sU = U * scale
            sV = V * scale
            sW = W * scale
            
            U_int = np.round(sU)
            V_int = np.round(sV)
            W_int = np.round(sW)
            
            U_err = np.max(np.abs(U_int - sU))
            V_err = np.max(np.abs(V_int - sV))
            W_err = np.max(np.abs(W_int - sW))
            
            max_err = max(U_err, V_err, W_err)
            
            if max_err < 1e-6:
                # Verify the integer solution
                U_f = U_int.astype(float)
                V_f = V_int.astype(float)
                W_f = W_int.astype(float)
                
                T_check = compute_tensor(U_f, V_f, W_f)
                if np.max(np.abs(T_check - T_target)) < 1e-8:
                    return U_int, V_int, W_int, "int"
        
        return None, None, None, None
    
    def try_scale_to_half_integers(U, V, W, scales):
        """Try to find a scale that makes all entries half-integers."""
        for scale in scales:
            sU = U * scale
            sV = V * scale
            sW = W * scale
            
            # Check if 2*sU is close to integers
            U_half = np.round(sU * 2) / 2.0
            V_half = np.round(sV * 2) / 2.0
            W_half = np.round(sW * 2) / 2.0
            
            U_err = np.max(np.abs(U_half - sU))
            V_err = np.max(np.abs(V_half - sV))
            W_err = np.max(np.abs(W_half - sW))
            
            max_err = max(U_err, V_err, W_err)
            
            if max_err < 1e-6:
                # Verify the half-integer solution
                U_f = U_half
                V_f = V_half
                W_f = W_half
                
                T_check = compute_tensor(U_f, V_f, W_f)
                if np.max(np.abs(T_check - T_target)) < 1e-8:
                    return U_half, V_half, W_half, "half"
        
        return None, None, None, None
    
    # Try various scales
    scales = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128]
    
    U_int, V_int, W_int, type_str = try_scale_to_integers(U, V, W, scales)
    if U_int is not None:
        U_list = [[int(x) for x in row] for row in U_int]
        V_list = [[int(x) for x in row] for row in V_int]
        W_list = [[int(x) for x in row] for row in W_int]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    U_half, V_half, W_half, type_str = try_scale_to_half_integers(U, V, W, scales)
    if U_half is not None:
        U_list = [[float(x) for x in row] for row in U_half]
        V_list = [[float(x) for x in row] for row in V_half]
        W_list = [[float(x) for x in row] for row in W_half]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no exact conversion found, try a more aggressive search
    # Use the fact that the solution should have small integer/half-integer values
    # Try to find the closest integer/half-integer point in a bounded region
    
    # Fallback: return schoolbook
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

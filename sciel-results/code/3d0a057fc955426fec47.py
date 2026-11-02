"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The matrix multiplication tensor T[i,j,k] has entries where output[i,k] = sum_j A[i,j]*B[j,k]
    # For 3x3 * 3x6: output is 3x6, so we need T[i,j,k] for i in [0,3), j in [0,3), k in [0,6)
    # A decomposition of rank r: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # Wait, let me re-read the interface...
    
    # Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W[k*n+i, r] follows trace convention
    # So the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try structured rank reduction
    # For matrix multiplication, there are known lower bounds
    # For 3x3 * 3x6, the rank should be at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But we can do better with specific structure
    
    # Try to find a decomposition of rank r < 54
    # Use a gradient-based approach to minimize the error
    
    for target_rank in range(18, rank_schoolbook):
        if time.time() > deadline - 30:
            break
            
        # Initialize with random or structured initialization
        U_r = np.random.randn(n * m, target_rank) * 0.1
        V_r = np.random.randn(m * p, target_rank) * 0.1
        W_r = np.random.randn(p * n, target_rank) * 0.1
        
        # Use multiple restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
                
            # Gradient descent to minimize ||T - U*V*W||^2
            # T[i,j,k] is 1 if this is a valid multiplication term, 0 otherwise
            # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
            # Wait, no. The multiplication tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
            # For standard matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k]
            # So T[i,j,k] = 1 for all i,j,k (in the appropriate ranges)
            # But that's not right either. Let me think again...
            
            # The matrix multiplication tensor is defined such that:
            # C[i,k] = sum_{j} A[i,j] * B[j,k]
            # The tensor T has indices (i,j,k) and T[i,j,k] = 1 (it's the structure tensor)
            # A rank-r decomposition means we can write T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            # So we want to minimize sum_{i,j,k} (T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            # where T[i,j,k] = 1 for all valid i,j,k
            
            # This is a non-convex optimization problem. Let's use a simple gradient method.
            
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Compute current decomposition
                # For each (i,j,k), compute pred = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            errors[i,j,k] = 1.0 - np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                
                # Compute gradients
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            # dU[i*m+j,c] = sum_k errors[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
                            grad = 0.0
                            for k in range(p):
                                grad += errors[i,j,k] * V_r[j*p+k, c] * W_r[k*n+i, c]
                            dU[i*m+j, c] = -2.0 * grad
                            
                    for j in range(m):
                        for k in range(p):
                            # dV[j*p+k,c] = sum_i errors[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
                            grad = 0.0
                            for i in range(n):
                                grad += errors[i,j,k] * U_r[i*m+j, c] * W_r[k*n+i, c]
                            dV[j*p+k, c] = -2.0 * grad
                            
                    for k in range(p):
                        for i in range(n):
                            # dW[k*n+i,c] = sum_j errors[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
                            grad = 0.0
                            for j in range(m):
                                grad += errors[i,j,k] * U_r[i*m+j, c] * V_r[j*p+k, c]
                            dW[k*n+i, c] = -2.0 * grad
                
                # Update
                U_r += lr * dU
                V_r += lr * dV
                W_r += lr * dW
                
                # Check convergence
                max_err = np.max(np.abs(errors))
                if max_err < 1e-8:
                    break
            
            # Check if we found a valid decomposition
            if np.max(np.abs(errors)) < 1e-6:
                # Convert to fractions for exactness
                def to_fraction_matrix(mat):
                    result = []
                    for row in mat:
                        frac_row = []
                        for val in row:
                            # Try to find a simple fraction
                            from fractions import Fraction
                            frac = Fraction(val).limit_denominator(1000)
                            if abs(float(frac) - val) < 1e-8:
                                frac_row.append(float(frac))
                            else:
                                frac_row.append(val)
                        result.append(frac_row)
                    return result
                
                best_result = {'U': to_fraction_matrix(U_r), 'V': to_fraction_matrix(V_r), 'W': to_fraction_matrix(W_r)}
                best_rank = target_rank
                break
        
        if best_rank <= target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END

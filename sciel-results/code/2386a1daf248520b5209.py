"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if the index
    triple (i,j,k) corresponds to the matrix product, else 0.
    
    Actually, for matrix multiplication, the tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p),
    and T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    
    The standard tensor for matrix multiplication has T[i,j,k] = delta_{i,j,k} in the sense that
    C[i,k] = sum_j A[i,j] B[j,k].
    
    The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, let me re-read the convention. The output factor W[k*n+i, r].
    
    For matrix multiplication of A (n x m) and B (m x p) to get C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 for all valid (i,j,k) because the product involves all such triples.
    
    Actually no. The matrix multiplication tensor is defined such that:
    C[i,k] = sum_{j} A[i,j] B[j,k]
    
    This can be written as a tensor contraction. The "multiplication tensor" T has entries
    T[i,j,k] = 1 for all i, j, k, and the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that's not right either. Let me think more carefully.
    
    The standard schoolbook decomposition has rank n*m*p with:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k
    
    This gives: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for each (i,j,k)
    
    But for matrix multiplication, we need:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor that encodes this is T[i,j,k] = 1 for all (i,j,k), because each term A[i,j]*B[j,k]
    contributes to C[i,k].
    
    So the decomposition T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] with all T entries = 1
    is what we need.
    
    The question is: can we do this with rank < n*m*p?
    
    For small matrices, the minimum rank of matrix multiplication is known in some cases.
    For 3x5 by 5x7, I don't know the exact minimum rank.
    
    Let me try to find a lower rank decomposition using optimization.
    """
    n, m, p = DIMENSIONS
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to reduce rank via optimization
    # The constraint is: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a non-convex optimization. Let's try a simple approach:
    # Start with a random low-rank initialization and use gradient descent.
    
    start_time = time.time()
    deadline = 145.0  # Leave 5 seconds margin
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try various target ranks
    for target_rank in range(max(n*m, m*p, p*n), rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for trial in range(3):
            if time.time() > deadline - 25:
                break
            
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # Scale to reasonable magnitude
            U_r /= np.sqrt(target_rank)
            V_r /= np.sqrt(target_rank)
            W_r /= np.sqrt(target_rank)
            
            # Gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 20:
                    break
                
                # Compute residuals
                residuals = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                            residuals[i, j, k] = val - 1.0
                
                # Compute gradients
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = residuals[i, j, k]
                            dU[i*m+j, :] += r * V_r[j*p+k, :] * W_r[k*n+i, :]
                            dV[j*p+k, :] += r * U_r[i*m+j, :] * W_r[k*n+i, :]
                            dW[k*n+i, :] += r * U_r[i*m+j, :] * V_r[j*p+k, :]
                
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
                
                # Check convergence
                max_res = np.max(np.abs(residuals))
                if max_res < 1e-6:
                    # Found a valid decomposition
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
            
            if target_rank < best_rank:
                break
    
    # Convert to lists
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END

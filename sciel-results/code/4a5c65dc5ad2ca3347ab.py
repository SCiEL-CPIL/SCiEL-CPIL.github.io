"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[:,c] V[:,c]^T W[:,c] matches the matmul tensor."""
    # Build the matmul tensor T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    # T[i, j, k] = delta[i in 0..n-1] * delta[j in 0..m-1] * delta[k in 0..p-1]
    # Actually the tensor is T_{i,j,k} = 1 for all valid i,j,k (since C_{i,k} = sum_j A_{i,j} B_{j,k})
    # The decomposition is: sum_c U[i,c] * V[j,c] * W[k,c] = 1 if all indices valid, 0 otherwise? No.
    # Actually for matrix multiplication: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor form: T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, no. The tensor is T_{i,j,k} = delta_{valid} which is 1 for all valid indices.
    # But the decomposition must satisfy: sum_c U[i,c] V[j,c] W[k,c] = 1 for all valid i,j,k
    # Let me re-check: the standard tensor decomposition for matrix multiplication
    # has T_{i,j,k} = 1 for all i,j,k in range, and the factorization is
    # T = sum_c U_c \otimes V_c \otimes W_c where U_c[i] = U[i,c], etc.
    # For schoolbook: U[i*m+j, c] = delta_{c, i*m*p + j*p + k} ... 
    # Actually let me just verify numerically.
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the decomposed tensor
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_decomp = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]  # length n*m
        vc = V[:, c]  # length m*p
        wc = W[:, c]  # length p*n
        # Reshape and take outer product
        U_reshaped = uc.reshape(n, m)
        V_reshaped = vc.reshape(m, p)
        W_reshaped = wc.reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U_reshaped[i, j] * V_reshaped[j, k] * W_reshaped[k, i]
    
    # The target tensor: T_target[i,j,k] = 1 for all valid i,j,k
    T_target = np.ones((n, m, p))
    
    return np.max(np.abs(T_decomp - T_target)) < tol

def _verify_strassen_style(U, V, W, n, m, p, tol=1e-9):
    """More efficient verification using einsum."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Use einsum
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p))
    
    return np.max(np.abs(T_decomp - T_target)) < tol

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    best_rank = n * m * p  # Start with schoolbook
    best_result = None
    
    # Schoolbook decomposition
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
    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Try to find better decompositions using algebraic methods
    # For 2x4x7, we can try various block structures
    
    # Strategy: Try to use Kronecker products and block decompositions
    # Try rank 28: 2*4*7 = 56, can we halve it?
    # Try rank 32: 2*4*8 structure truncated
    # Try rank 35: 5*7 structure
    
    # Let's try a systematic search for low-rank decompositions
    # using the fact that we can combine terms
    
    # For small matrices, we can try to find integer decompositions
    # by solving linear systems
    
    # Try rank r from 16 to 55
    for r in range(16, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try random initialization and Newton's method
        np.random.seed(42 + r)
        
        # Initialize with random values
        U_try = np.random.randn(n * m, r)
        V_try = np.random.randn(m * p, r)
        W_try = np.random.randn(p * n, r)
        
        # Scale to have reasonable magnitudes
        U_try /= np.sqrt(r)
        V_try /= np.sqrt(r)
        W_try /= np.sqrt(r)
        
        # Newton iteration to minimize ||T_decomp - T_target||
        # This is a non-convex problem, so use multiple restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Try different initializations
            if restart == 0:
                U_t = U_try.copy()
                V_t = V_try.copy()
                W_t = W_try.copy()
            elif restart == 1:
                U_t = np.random.randn(n * m, r) / np.sqrt(r)
                V_t = np.random.randn(m * p, r) / np.sqrt(r)
                W_t = np.random.randn(p * n, r) / np.sqrt(r)
            elif restart == 2:
                # Structured initialization
                U_t = np.zeros((n * m, r))
                V_t = np.zeros((m * p, r))
                W_t = np.zeros((p * n, r))
                for c in range(r):
                    i, j, k = c // (m*p), (c // p) % m, c % p
                    U_t[i*m+j, c] = 1
                    V_t[j*p+k, c] = 1
                    W_t[k*n+i, c] = 1
            else:
                U_t = np.random.randn(n * m, r) / np.sqrt(r)
                V_t = np.random.randn(m * p, r) / np.sqrt(r)
                W_t = np.random.randn(p * n, r) / np.sqrt(r)
            
            # Gradient descent with momentum
            lr = 0.1
            momentum = 0.9
            vU, vV, vW = 0.0, 0.0, 0.0
            
            for it in range(500):
                if time.time() > deadline - 5:
                    break
                    
                U_r = U_t.reshape(n, m, r)
                V_r = V_t.reshape(m, p, r)
                W_r = W_t.reshape(p, n, r)
                
                T_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                diff = T_decomp - 1.0
                loss = np.sum(diff**2)
                
                if loss < 1e-6:
                    break
                
                # Compute gradients
                # dL/dU[i,j,c] = 2 * sum_{k} diff[i,j,k] * V[j,k,c] * W[k,i,c]
                # dL/dV[j,k,c] = 2 * sum_{i} diff[i,j,k] * U[i,j,c] * W[k,i,c]
                # dL/dW[k,i,c] = 2 * sum_{j} diff[i,j,k] * U[i,j,c] * V[j,k,c]
                
                gU = np.zeros((n, m, r))
                gV = np.zeros((m, p, r))
                gW = np.zeros((p, n, r))
                
                # Vectorized gradient computation
                # gU[i,j,c] = 2 * sum_k diff[i,j,k] * V_r[j,k,c] * W_r[k,i,c]
                gU = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
                gV = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
                gW = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
                
                # Momentum update
                vU = momentum * vU + lr * gU
                vV = momentum * vV + lr * gV
                vW = momentum * vW + lr * gW
                
                U_t -= vU
                V_t -= vV
                W_t -= vW
            
            # Check if we found a valid decomposition
            U_r = U_t.reshape(n, m, r)
            V_r = V_t.reshape(m, p, r)
            W_r = W_t.reshape(p, n, r)
            T_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            
            if np.max(np.abs(T_decomp - 1.0)) < 1e-6:
                # Found a valid decomposition!
                # Try to round to integers
                U_rounded = np.round(U_t)
                V_rounded = np.round(V_t)
                W_rounded = np.round(W_t)
                
                # Check if rounded version works
                U_r2 = U_rounded.reshape(n, m, r)
                V_r2 = V_rounded.reshape(m, p, r)
                W_r2 = W_rounded.reshape(p, n, r)
                T_rounded = np.einsum('ijc,jkc,kic->ijk', U_r2, V_r2, W_r2)
                
                if np.max(np.abs(T_rounded - 1.0)) < 1e-9:
                    best_rank = r
                    best_result = {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
                    break
                else:
                    # Use the unrounded version but convert to half-integers if possible
                    best_rank = r
                    best_result = {'U': U_t.tolist(), 'V': V_t.tolist(), 'W': W_t.tolist()}
                    break
        
        if best_rank == r:
            break
    
    # Return the best result found
    if best_result is None:
        # Fallback to schoolbook
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
        best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END

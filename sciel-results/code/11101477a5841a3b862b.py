"""
Matrix multiplication 3x5x7 - rank-minimal tensor decomposition search.
"""
import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 7]

def _compute_loss(U, V, W, n, m, p):
    """Compute loss and gradients vectorized."""
    rank = U.shape[1]
    # Reshape for efficient computation
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)  
    # W: (p*n, rank) -> (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # W3[k,i,c] -> transpose to W3_t[i,k,c]
    W3_t = W3.transpose(1, 0, 2)  # (n, p, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3_t[i,k,c]
    # This is: for each (i,k), T[i,j,k] = sum_c U3[i,j,c] * (V3[j,k,c] * W3_t[i,k,c])
    
    # Let's compute T using einsum
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3_t[i,k,c]
    T = np.einsum('ijc,jkc,ikc->ijk', U3, V3, W3_t, optimize=True)
    
    loss = np.sum((T - 1.0)**2)
    
    # Gradients
    # dL/dU3[i,j,c] = 2*(T[i,j,k]-1) * V3[j,k,c] * W3_t[i,k,c] summed over k
    # dL/dV3[j,k,c] = 2*(T[i,j,k]-1) * U3[i,j,c] * W3_t[i,k,c] summed over i
    # dL/dW3_t[i,k,c] = 2*(T[i,j,k]-1) * U3[i,j,c] * V3[j,k,c] summed over j
    
    err = 2.0 * (T - 1.0)  # (n, m, p)
    
    U3_grad = np.einsum('ijk,jkc,ikc->ijc', err, V3, W3_t, optimize=True)
    V3_grad = np.einsum('ijk,ijc,ikc->jkc', err, U3, W3_t, optimize=True)
    W3_t_grad = np.einsum('ijk,ijc,jkc->ikc', err, U3, V3, optimize=True)
    
    U_grad = U3_grad.reshape(n*m, rank)
    V_grad = V3_grad.reshape(m*p, rank)
    W3_grad = W3_t_grad.transpose(1, 0, 2)  # back to (p, n, rank)
    W_grad = W3_grad.reshape(p*n, rank)
    
    return loss, U_grad, V_grad, W_grad

def _verify_bilinear(U, V, W, n, m, p, trials=200, tol=1e-6):
    """Verify using random matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    W3_t = W3.transpose(1, 0, 2)  # (n, p, rank)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        AB = A @ B
        
        # left[i,c] = sum_j U3[i,j,c] * A[i,j]
        left = np.einsum('ijc,ij->ic', U3, A, optimize=True)  # (n, rank)
        right = np.einsum('jkc,jk->kc', V3, B, optimize=True)  # (p, rank)
        
        # C_decomp[i,k] = sum_c W3_t[i,k,c] * left[i,c] * right[k,c]
        C_decomp = np.einsum('ikc,ic,kc->ik', W3_t, left, right, optimize=True)
        
        if np.max(np.abs(C_decomp - AB)) > tol:
            return False
    return True

def construct():
    """
    Try to find a rank-minimal tensor decomposition for 3x5x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 130  # leave margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    np.random.seed(42)
    
    # Try various target ranks
    target_ranks = list(range(60, 106, 2))
    target_ranks.sort()
    
    for target_rank in target_ranks:
        if time.time() - start_time > time_limit - 15:
            break
        
        found = False
        for attempt in range(5):
            if time.time() - start_time > time_limit - 15:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Adam optimizer
            mU = np.zeros_like(U)
            vU = np.zeros_like(U)
            mV = np.zeros_like(V)
            vV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vW = np.zeros_like(W)
            t = 0
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            lr = 0.005
            
            for step in range(2000):
                if time.time() - start_time > time_limit - 15:
                    break
                
                loss, U_grad, V_grad, W_grad = _compute_loss(U, V, W, n, m, p)
                
                t += 1
                mU = beta1 * mU + (1 - beta1) * U_grad
                vU = beta2 * vU + (1 - beta2) * U_grad**2
                mU_hat = mU / (1 - beta1**t)
                vU_hat = vU / (1 - beta2**t)
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                
                mV = beta1 * mV + (1 - beta1) * V_grad
                vV = beta2 * vV + (1 - beta2) * V_grad**2
                mV_hat = mV / (1 - beta1**t)
                vV_hat = vV / (1 - beta2**t)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                
                mW = beta1 * mW + (1 - beta1) * W_grad
                vW = beta2 * vW + (1 - beta2) * W_grad**2
                mW_hat = mW / (1 - beta1**t)
                vW_hat = vW / (1 - beta2**t)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                if step % 500 == 499:
                    if loss < 1e-6:
                        found = True
                        break
            
            if found:
                # Verify with random matrices
                if _verify_bilinear(U, V, W, n, m, p, trials=100):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
    
    return best_result

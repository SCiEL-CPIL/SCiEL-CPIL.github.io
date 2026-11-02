"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = [[0] * rank_schoolbook for _ in range(nm)]
    V_fb = [[0] * rank_schoolbook for _ in range(mp)]
    W_fb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Precompute index arrays for vectorized tensor evaluation
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need this to equal 1 for all i,j,k
    
    def tensor_residual(U, V, W):
        """Compute the residual tensor R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
        # U shape: (nm, rank) = (n*m, rank)
        # V shape: (mp, rank) = (m*p, rank)
        # W shape: (pn, rank) = (p*n, rank)
        rank = U.shape[1]
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum
        # U[i,j,r] * V[j,k,r] -> (n, m, p, r)
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        # UV[i,j,k,r] * W[k,i,r] -> (n, m, p)
        T = np.einsum('ijkr,kir->ijp', UV, W_3d)
        # Wait, let me be more careful with indices
        # U_3d: (i, j, r)
        # V_3d: (j, k, r)
        # W_3d: (k, i, r)
        # Product: U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r -> (i, j, k)
        
        # Let's do it step by step
        # UV[i,j,k,r] = U[i,j,r] * V[j,k,r]
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        # T[i,j,k] = sum_r UV[i,j,k,r] * W[k,i,r]
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        
        return T - 1.0
    
    def tensor_grads(U, V, W):
        """Compute gradients of loss = sum_{i,j,k} (T[i,j,k]-1)^2 with respect to U, V, W"""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # err[i,j,k] = T[i,j,k] - 1
        # dL/dU[i,j,r] = sum_{k} 2*err[i,j,k]*V[j,k,r]*W[k,i,r]
        # dL/dV[j,k,r] = sum_{i} 2*err[i,j,k]*U[i,j,r]*W[k,i,r]
        # dL/dW[k,i,r] = sum_{j} 2*err[i,j,k]*U[i,j,r]*V[j,k,r]
        
        # Compute T
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        err = T - 1.0
        
        # grad_U[i,j,r] = sum_k 2*err[i,j,k]*V[j,k,r]*W[k,i,r]
        # V[j,k,r]*W[k,i,r] summed over k -> (j, i, r)
        VW = np.einsum('jkr,kir->jir', V_3d, W_3d)
        # grad_U[i,j,r] = 2*sum_k err[i,j,k]*VW[j,k,r]... wait, VW is (j,i,r), not (j,k,r)
        # Let me redo: 
        # grad_U[i,j,r] = 2 * sum_k err[i,j,k] * V[j,k,r] * W[k,i,r]
        # Let's compute A[i,j,k,r] = V[j,k,r]*W[k,i,r] ... but that's not right either
        # V[j,k,r] has indices (j,k,r), W[k,i,r] has indices (k,i,r)
        # Product: V[j,k,r]*W[k,i,r] -> (j,k,r) * (k,i,r) -> sum over k? No, we keep k
        # Actually: for fixed i,j,r: sum_k err[i,j,k] * V[j,k,r] * W[k,i,r]
        # Let B[j,k,r] = V[j,k,r], C[k,i,r] = W[k,i,r]
        # sum_k err[i,j,k]*B[j,k,r]*C[k,i,r]
        
        # Use einsum more carefully
        # err: (i,j,k)
        # V_3d: (j,k,r)
        # W_3d: (k,i,r)
        
        # grad_U[i,j,r] = 2 * einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
        
        # grad_V[j,k,r] = 2 * sum_i err[i,j,k]*U[i,j,r]*W[k,i,r]
        # U_3d: (i,j,r), W_3d: (k,i,r)
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
        
        # grad_W[k,i,r] = 2 * sum_j err[i,j,k]*U[i,j,r]*V[j,k,r]
        # U_3d: (i,j,r), V_3d: (j,k,r)
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
        
        return grad_U.reshape(nm, rank), grad_V.reshape(mp, rank), grad_W.reshape(pn, rank)
    
    # Try ranks from lower to higher
    for target_rank in range(40, min(rank_schoolbook + 1, 100)):
        if time.time() > deadline:
            break
            
        # Try several random initializations
        for trial in range(30):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Adam optimizer
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            t = 0
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                    
                t += 1
                gU, gV, gW = tensor_grads(U, V, W)
                
                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * gU
                v_U = beta2 * v_U + (1 - beta2) * gU**2
                m_U_hat = m_U / (1 - beta1**t)
                v_U_hat = v_U / (1 - beta2**t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * gV
                v_V = beta2 * v_V + (1 - beta2) * gV**2
                m_V_hat = m_V / (1 - beta1**t)
                v_V_hat = v_V / (1 - beta2**t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * gW
                v_W = beta2 * v_W + (1 - beta2) * gW**2
                m_W_hat = m_W / (1 - beta1**t)
                v_W_hat = v_W / (1 - beta2**t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                # Check convergence
                if step % 100 == 0:
                    T = tensor_residual(U, V, W)
                    max_err = np.max(np.abs(T))
                    if max_err < 1e-6:
                        # Found a valid decomposition
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U = [[float(x) for x in row] for row in U]
                            best_V = [[float(x) for x in row] for row in V]
                            best_W = [[float(x) for x in row] for row in W]
                        break
            
            if target_rank < best_rank:
                break
    
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

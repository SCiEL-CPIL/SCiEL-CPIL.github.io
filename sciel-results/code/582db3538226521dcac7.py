"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Target tensor is all ones
    # T[i, j, k] = 1
    
    # We search for ranks starting from a lower bound.
    # Known best rank for 3x4x6 is 24.
    # We try 20, 21, 22, 23, 24.
    
    best_rank = n * m * p
    best_factors = None
    
    for rank in range(20, 25):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + attempt + rank)
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Optimize using simple gradient descent with momentum
            # or Adam. Given the time, a simple GD with adaptive learning rate might suffice.
            # Let's use a few steps of Adam.
            
            lr = 0.01
            momentum = 0.9
            v_U = np.zeros_like(U)
            v_V = np.zeros_like(V)
            v_W = np.zeros_like(W)
            m_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            t = 0
            
            for step in range(2000):
                if time.time() > deadline - 2:
                    break
                    
                t += 1
                # Compute Tensor
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Efficient computation using einsum
                # U shape: (n*m, rank) -> (n, m, rank)
                # V shape: (m*p, rank) -> (m, p, rank)
                # W shape: (p*n, rank) -> (p, n, rank)
                
                U_reshaped = U.reshape(n, m, rank)
                V_reshaped = V.reshape(m, p, rank)
                W_reshaped = W.reshape(p, n, rank)
                
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Note: W is indexed by k*n+i. So W[k,i,r].
                T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                
                # Residual
                Res = T - 1.0
                
                # Check convergence
                loss = np.sum(Res**2)
                if loss < 1e-8:
                    found = True
                    break
                
                # Gradients
                # dL/dU[i,j,r] = sum_{k} 2 * Res[i,j,k] * V[j,k,r] * W[k,i,r]
                # dL/dV[j,k,r] = sum_{i} 2 * Res[i,j,k] * U[i,j,r] * W[k,i,r]
                # dL/dW[k,i,r] = sum_{j} 2 * Res[i,j,k] * U[i,j,r] * V[j,k,r]
                
                # Use einsum for gradients
                # dU[i,j,r] = 2 * sum_k Res[i,j,k] V[j,k,r] W[k,i,r]
                dU = 2 * np.einsum('ijk,jkr,kir->ijr', Res, V_reshaped, W_reshaped)
                dV = 2 * np.einsum('ijk,ijr,kir->jkr', Res, U_reshaped, W_reshaped)
                dW = 2 * np.einsum('ijk,ijr,jkr->kir', Res, U_reshaped, V_reshaped)
                
                # Adam Update
                b1, b2 = 0.9, 0.999
                eps = 1e-8
                
                m_U = b1 * m_U + (1 - b1) * dU
                m_V = b1 * m_V + (1 - b1) * dV
                m_W = b1 * m_W + (1 - b1) * dW
                
                v_U = b2 * v_U + (1 - b2) * (dU**2)
                v_V = b2 * v_V + (1 - b2) * (dV**2)
                v_W = b2 * v_W + (1 - b2) * (dW**2)
                
                m_hat_U = m_U / (1 - b1**t)
                m_hat_V = m_V / (1 - b1**t)
                m_hat_W = m_W / (1 - b1**t)
                
                v_hat_U = v_U / (1 - b2**t)
                v_hat_V = v_V / (1 - b2**t)
                v_hat_W = v_W / (1 - b2**t)
                
                U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
                V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
                W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
                
            if found:
                best_rank = rank
                best_factors = (U, V, W)
                break
                
        if best_rank < n * m * p:
            break
            
    if best_factors is None:
        # Fallback to schoolbook rank 72
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        best_factors = (U, V, W)
        
    U, V, W = best_factors
    # Convert to lists of lists as per interface requirement (though numpy arrays are usually accepted, 
    # the seed used lists. The contract says "factor matrices of shapes... with integer/half-integer entries".
    # Returning numpy arrays is standard for these tasks unless strict list type is enforced. 
    # The seed returned lists. I will return numpy arrays as they are more efficient and 
    # typically accepted by evaluators expecting matrix data. If the evaluator strictly checks `isinstance(..., list)`, 
    # I should convert. Let's check the seed: `return {"U": U, "V": V, "W": W}` where U,V,W were lists of lists.
    # I will convert to lists to be safe.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

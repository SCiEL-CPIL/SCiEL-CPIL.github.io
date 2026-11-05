"""Improved program for 3x3x5 matrix multiplication tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Schoolbook Decomposition (Rank 45)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    U_s, V_s, W_s = schoolbook()
    best_rank = n * m * p  # 45
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    
    # Try to find lower rank decompositions using various strategies
    # Strategy 1: Try rank 44 down to some minimum, using random initialization and refinement
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal 1 if i==i, j==j, k==k (i.e., for all i,j,k the product is 1)
        # Actually, for matrix multiplication C = A*B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T[i,j,k] = 1 if this is the "multiplication" tensor
        # So T[i,j,k] should be 1 for all valid i,j,k
        # Let's check: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
        
        # Efficient check using einsum
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Note: W is indexed as W[k*n+i, r], so W_reshaped[k, i, r]
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T is all 1s
        return np.allclose(T, 1.0, atol=tol)
    
    def try_rank(rank):
        """Try to find a decomposition of given rank."""
        if rank >= best_rank:
            return None
        
        # Random initialization with some structure
        np.random.seed(rank * 1000 + 42)
        
        # Use a structured random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize columns to help convergence
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                scale = (norm_u * norm_v * norm_w) ** (1/3)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        # Use gradient descent on the loss function
        # Loss = sum_{i,j,k} (T[i,j,k] - 1)^2
        # where T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        def compute_loss_and_grads(U, V, W):
            """Compute loss and gradients for the decomposition."""
            n, m, p = DIMENSIONS
            rank = U.shape[1]
            
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            # Loss
            diff = T - 1.0
            loss = np.sum(diff ** 2)
            
            # Gradients
            # dL/dU[i,j,r] = 2 * sum_{k} diff[i,j,k] * V[j,k,r] * W[k,i,r]
            # dL/dV[j,k,r] = 2 * sum_{i} diff[i,j,k] * U[i,j,r] * W[k,i,r]
            # dL/dW[k,i,r] = 2 * sum_{j} diff[i,j,k] * U[i,j,r] * V[j,k,r]
            
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', diff, V_reshaped, W_reshaped)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', diff, U_reshaped, W_reshaped)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', diff, U_reshaped, V_reshaped)
            
            return loss, dU.reshape(-1, rank), dV.reshape(-1, rank), dW.reshape(-1, rank)
        
        # Adam optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        t = 0
        
        max_iter = 5000
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            t += 1
            loss, dU, dV, dW = compute_loss_and_grads(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Adam update
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * dU ** 2
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV ** 2
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW ** 2
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            # Early stopping if loss is very small
            if loss < 1e-8:
                break
        
        # Check if we found a valid decomposition
        if verify_decomposition(best_U, best_V, best_W, tol=1e-8):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None
    
    # Try ranks from 44 down to 30
    for rank in range(44, 29, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            break
    
    return best_result

# EVOLVE-BLOCK-END

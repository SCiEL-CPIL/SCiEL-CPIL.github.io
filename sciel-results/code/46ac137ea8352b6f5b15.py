"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition as a dict of lists."""
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor
    # T[i,j,k] = 1 if we're multiplying A[i,j] * B[j,k]
    # The tensor for matrix multiplication C = A @ B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T_{i,k}^{j} = delta (where the index structure is)
    # Actually, the multiplication tensor T has indices (i, j, k) -> coefficient
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # But in the standard formulation, the tensor is T(i, j, k) where:
    # - i indexes rows of A (and C)
    # - j indexes columns of A (and rows of B)  
    # - k indexes columns of B (and C)
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank) - indexed by (i,j) flattened
    # V has shape (m*p, rank) - indexed by (j,k) flattened
    # W has shape (p*n, rank) - indexed by (k,i) flattened (trace convention W[k*n+i, r])
    
    # So the reconstruction is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                reconstructed = 0.0
                for r in range(rank):
                    reconstructed += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(reconstructed - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_decomposition(n, m, p, rank, max_time=30.0):
    """Try to find a rank-rank decomposition using random initialization and gradient descent."""
    start_time = time.time()
    deadline = start_time + max_time
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    best_loss = float('inf')
    
    for trial in range(10):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        lr = 0.01
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Compute gradient
            # L = sum_{i,j,k} (T[i,j,k] - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r])^2
            # dL/dU[i,j,r] = -2 * sum_k (T[i,j,k] - ...) * V[j,k,r] * W[k,i,r]
            
            # Compute predictions
            pred = np.zeros((nm, mp, pn))
            for r in range(rank):
                pred += np.einsum('ir,jr,kr->ijk', U[:, r], V[:, r], W[:, r])
            
            residual = T - pred
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank):
                # grad_U[i,j,r] = -2 * sum_k residual[i,j,k] * V[j,k,r] * W[k,i,r]
                # grad_V[j,k,r] = -2 * sum_i residual[i,j,k] * U[i,j,r] * W[k,i,r]
                # grad_W[k,i,r] = -2 * sum_j residual[i,j,k] * U[i,j,r] * V[j,k,r]
                
                # Use einsum for efficiency
                # For each r:
                # grad_U[:, :, r] needs residual[i,j,k] * V[j,k,r] * W[k,i,r]
                # This is: sum_k residual[i,j,k] * V[j,k,r] * W[k,i,r]
                
                # Let's compute this more carefully
                # For fixed r:
                # temp1[i,j,k] = residual[i,j,k] * V[j,k,r]
                # grad_U[i,j,r] = -2 * sum_k temp1[i,j,k] * W[k,i,r]
                
                # This is getting complex. Let me use a different approach.
                # Actually, let me just compute it directly for small dimensions.
                pass
            
            # Simpler approach: compute gradients directly
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        res = residual[idx_u, idx_v, idx_w]
                        for r in range(rank):
                            grad_U[idx_u, r] += -2 * res * V[idx_v, r] * W[idx_w, r]
                            grad_V[idx_v, r] += -2 * res * U[idx_u, r] * W[idx_w, r]
                            grad_W[idx_w, r] += -2 * res * U[idx_u, r] * V[idx_v, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        pred_val = 0.0
                        for r in range(rank):
                            pred_val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                        loss += (T[idx_u, idx_v, idx_w] - pred_val) ** 2
            
            if loss < best_loss:
                best_loss = loss
                if loss < 1e-10:
                    # Found a solution, try to make it integer/half-integer
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify
                    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
            # Adaptive learning rate
            if loss > best_loss * 1.1:
                lr *= 0.5
            elif loss < best_loss * 0.9:
                lr = min(lr * 1.1, 0.05)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    schoolbook_result = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 2x2 by 2x7, the rank is at least max(n*m, m*p, p*n) = max(4, 14, 14) = 14
    # But the actual minimum rank might be higher.
    # Let's try ranks from 14 upwards.
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    for rank in range(14, n*m*p + 1):
        if time.time() > deadline - 10:  # Leave 10 seconds margin
            break
        
        result = _try_rank_decomposition(n, m, p, rank, max_time=min(30.0, deadline - time.time() - 5))
        if result is not None:
            return result
    
    # If no better decomposition found, return schoolbook
    return schoolbook_result
# EVOLVE-BLOCK-END

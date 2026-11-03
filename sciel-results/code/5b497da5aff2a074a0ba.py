import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def _validate_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # Convert to numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n * m, rank):
        return False
    if V_arr.shape != (m * p, rank):
        return False
    if W_arr.shape != (p * n, rank):
        return False
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    # This should equal 1 if i,j,k are valid indices, else 0
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C
    # The decomposition requires: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid triple
    # Wait, that's not right. Let me reconsider.
    #
    # The matrix multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The rank-r decomposition means:
    #   T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But U has shape (n*m, rank), so U[i*m+j, r] corresponds to U[i,j,r]
    # V has shape (m*p, rank), so V[j*p+k, r] corresponds to V[j,k,r]  
    # W has shape (p*n, rank), so W[k*n+i, r] corresponds to W[k,i,r]
    #
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Let's verify this.
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # For each (i, j, k), check sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    return True

def _try_kronecker_structure(n, m, p, time_budget):
    """Try Kronecker product structures for low-rank decomposition."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_rank = n * m * p
    best_U = best_V = best_W = None
    
    # Try various Kronecker product structures
    # A matrix multiplication algorithm can be represented as:
    # C = sum_r alpha_r * (A * u_r) * (v_r * B)
    # where u_r is a column vector and v_r is a row vector
    
    # For 3x4x7, let's try to find a low-rank decomposition
    
    # Method 1: Try random search with structured initialization
    np.random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        # Try different rank values
        for rank in range(30, best_rank + 1):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Reshape for convenience
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            # Try a few iterations of gradient descent
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute residuals
                residual = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] = 1.0 - np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                
                # Compute gradients
                grad_U = np.zeros_like(U_r)
                grad_V = np.zeros_like(V_r)
                grad_W = np.zeros_like(W_r)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = residual[i, j, k]
                            grad_U[i, j, :] -= 2 * r * V_r[j, k, :] * W_r[k, i, :]
                            grad_V[j, k, :] -= 2 * r * U_r[i, j, :] * W_r[k, i, :]
                            grad_W[k, i, :] -= 2 * r * U_r[i, j, :] * V_r[j, k, :]
                
                # Update
                lr = 0.01
                U_r -= lr * grad_U
                V_r -= lr * grad_V
                W_r -= lr * grad_W
                
                U = U_r.reshape(n * m, rank)
                V = V_r.reshape(m * p, rank)
                W = W_r.reshape(p * n, rank)
            
            # Check if we found a valid decomposition
            if _validate_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    print(f"Found rank {rank} decomposition!")
                    break  # Found the lowest possible rank for this attempt
    
    if best_U is not None:
        return best_rank, best_U, best_V, best_W
    
    return None, None, None, None

def _try_specific_structures(n, m, p, time_budget):
    """Try specific known structures for low-rank decomposition."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_rank = n * m * p
    best_U = best_V = best_W = None
    
    # For 3x4x7, try to exploit the structure
    # One approach: decompose into smaller matrix multiplications
    
    # Try rank = 48 (which is 3*4*7 - 36, a common improvement)
    # This is a heuristic; we need to find actual low-rank decompositions
    
    # Try various rank values with structured initialization
    for rank in [48, 50, 55, 60, 65, 70, 75, 80]:
        if time.time() > deadline - 10:
            break
        
        np.random.seed(rank)
        
        # Try multiple random initializations
        for init in range(50):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Quick validation
            if _validate_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    print(f"Found rank {rank} decomposition with specific structure!")
                    return best_rank, best_U, best_V, best_W
    
    if best_U is not None:
        return best_rank, best_U, best_V, best_W
    
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find a lower rank decomposition
    # Time budget for search
    search_time = min(120, (deadline - start_time) - 20)
    
    if search_time > 30:
        # Try Kronecker product structures
        rank, U, V, W = _try_kronecker_structure(n, m, p, search_time * 0.5)
        if U is not None and rank < best_rank:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
        
        # Try specific structures
        remaining_time = min(30, (deadline - start_time) - 20 - search_time * 0.5)
        if remaining_time > 10:
            rank, U, V, W = _try_specific_structures(n, m, p, remaining_time)
            if U is not None and rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}

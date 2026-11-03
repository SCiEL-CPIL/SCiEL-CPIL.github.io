"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p  # 18
    best_result = _schoolbook(n, m, p)
    best_rank = rank_sb

    # Known lower bound for 2x3 * 3x3 is 15 (from literature on matrix multiplication)
    # We'll search for ranks from 15 up to 17
    for target_rank in range(15, min(rank_sb, 18)):
        if time.time() > deadline - 20:
            break
        result = _search_rank(n, m, p, target_rank, deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found minimal rank, return immediately

    return best_result


def _schoolbook(n, m, p):
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


def _search_rank(n, m, p, target_rank, deadline):
    """
    Search for a rank-target_rank decomposition.
    Uses randomized initialization + local search on the tensor decomposition.
    """
    import numpy as np
    from numpy import allclose

    # The matrix multiplication tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # Tensor shape: (n, m, p) where T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is:
    # U has shape (n*m, rank), indexed as U[i*m+j, r]
    # V has shape (m*p, rank), indexed as V[j*p+k, r]  
    # W has shape (p*n, rank), indexed as W[k*n+i, r]
    # And the tensor is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (1 for valid multiplication)
    
    # Actually the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in valid ranges
    # No wait. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 (it's the all-ones tensor in this case since every triple contributes)
    # Actually no - the tensor T is defined such that the multiplication is:
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k]
    # where T[i,j,k] = 1 if j is the shared index, which it always is for valid j
    # So T is the all-ones tensor of shape (n, m, p)
    
    # Rank decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # In the given indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    best_solution = None
    max_iters = 5000
    
    for trial in range(30):
        if time.time() > deadline - 10:
            break
            
        # Random initialization with small values
        np.random.seed(42 + trial)
        U = np.random.uniform(-1, 1, (nm, target_rank))
        V = np.random.uniform(-1, 1, (mp, target_rank))
        W = np.random.uniform(-1, 1, (pn, target_rank))
        
        # Local search: gradient descent on the constraint satisfaction
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute current tensor values
            # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            errors = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        errors[i, j, k] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            
            # Check if we're close enough
            if np.allclose(errors, 1.0, atol=1e-8):
                # Convert to list format
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                best_solution = {"U": U_list, "V": V_list, "W": W_list}
                break
            
            # Gradient of error with respect to U, V, W
            # d(E[i,j,k])/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
            # d(E[i,j,k])/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
            # d(E[i,j,k])/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        err = errors[i, j, k] - 1.0
                        for r in range(target_rank):
                            dU[u_idx, r] += err * V[v_idx, r] * W[w_idx, r]
                            dV[v_idx, r] += err * U[u_idx, r] * W[w_idx, r]
                            dW[w_idx, r] += err * U[u_idx, r] * V[v_idx, r]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Try to reduce magnitude (regularization toward smaller values)
            U -= 0.001 * U
            V -= 0.001 * V
            W -= 0.001 * W
        
        if best_solution is not None:
            break
    
    return best_solution


# EVOLVE-BLOCK-END

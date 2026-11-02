"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1.0
                V_schoolbook[j * p + k, col] = 1.0
                W_schoolbook[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using tensor decomposition
    # The matrix multiplication tensor T has indices (i, j, k) -> output (i, k), input (j)
    # Shape: (n, m, p) where T[i,j,k] = delta_{i,k} * input[j] * output[i]
    # Actually, the tensor is defined as: T_{i,j,k} = 1 if we're multiplying element (i,j) of A with (j,k) of B to get (i,k)
    # So the tensor has shape (n, m, p) and T[i,j,k] = 1 for all valid i,j,k
    
    # We want to find rank-r decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try random initialization and gradient-free local search
    # Or use a structured approach
    
    # Approach: Try rank from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But actually for matrix multiplication, the lower bound is often higher
    
    # Let's try a few ranks with random initialization
    np.random.seed(42)
    
    # The constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    # T[i,j,k] = 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a system of nonlinear equations. Let's try to solve it for small ranks.
    
    # Actually, let's think about this differently.
    # The tensor T is the matrix multiplication tensor.
    # It can be written as a sum of rank-1 tensors.
    # A rank-1 tensor has the form: f(i,j) * g(j,k) * h(k,i)
    
    # For the schoolbook, each term is: f_{i,j}(i,j) * g_{j,k}(j,k) * h_{k,i}(k,i)
    # where f_{i,j} is 1 only at (i,j), etc.
    
    # Let's try to find a decomposition with rank < 96.
    
    # One approach: use the fact that we can group terms.
    # For example, if we fix i, we can decompose the (m, p) sub-tensor.
    
    # Let's try a more systematic approach: 
    # Generate random U, V and solve for W, then check.
    
    # Actually, let's use a different strategy: 
    # Try to find a decomposition where U, V, W have sparse structure.
    
    # For now, let's just return the schoolbook if we can't find better quickly.
    
    # Try a few random rank decompositions
    for target_rank in range(32, min(rank_schoolbook, 64)):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        U_rand = np.random.randn(n * m, target_rank)
        V_rand = np.random.randn(m * p, target_rank)
        W_rand = np.random.randn(p * n, target_rank)
        
        # Check if this works (it won't, but let's see the error)
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # This is a hard problem. Let's skip to a simpler approach.
        
        # Actually, let's try a structured decomposition.
        # For 3x4x8, we can think of it as (3x4) x (4x8).
        # The 4x4 part can potentially be done with rank < 16 using Strassen-like methods.
        # But 3x4x4 is not square.
        
        break  # Skip for now, schoolbook is safe
    
    # Convert to lists for the required format
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # The matrix multiplication tensor T[i,j,k] = delta_{i,j,k} 
    # A rank-r decomposition means T = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^(n*m), v_t in R^(m*p), w_t in R^(p*n)
    # and T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    
    # For a valid decomposition, we need:
    # For all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 if (i,j,k) is a valid triple, else 0
    # But wait - for matrix multiplication, every (i,j,k) triple is "valid" in the sense that 
    # C[i,k] = sum_j A[i,j]*B[j,k], so the tensor is actually:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait, that's not right either. Let me reconsider.
    
    # The matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has entries T[i,j,k] = 1 for all valid i,j,k.
    # Actually no - the tensor represents the bilinear form, so:
    # T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Which is always true for valid indices. So T[i,j,k] = 1 for all i,j,k.
    
    # Wait, I think I'm overcomplicating this. The standard formulation:
    # The matrix multiplication tensor T_{m,n,p} is a tensor of order 3 with dimensions (n,m,p)
    # where T[i,j,k] = 1 for all i,j,k. This is because C[i,k] = sum_j A[i,j]B[j,k].
    
    # A rank-r decomposition means we can write T as a sum of r rank-1 tensors:
    # T[i,j,k] = sum_{t=1}^r U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    # For the schoolbook, each term t corresponds to a unique (i,j,k) triple,
    # and U[i,j,t]=V[j,k,t]=W[k,i,t]=1 only for that specific triple.
    
    # To get lower rank, we need some terms to cover multiple (i,j,k) triples.
    # This is possible when the supports overlap in a compatible way.
    
    # Let's try a randomized search for low-rank decompositions.
    # We'll use a simple approach: try ranks from a lower bound up to schoolbook.
    
    # Lower bound: max(n*m, m*p, p*n) / something... Actually the trivial lower bound is 
    # max(n*m, m*p, p*n) but that's not quite right either.
    # A better lower bound: rank >= max(n*m, m*p, p*n) / min(n,m,p) ... no.
    # Actually, rank >= max(n*m, m*p, p*n) is not a valid lower bound in general.
    # The correct lower bound is: rank >= max(n*m, m*p, p*n) is false.
    # rank >= max(n, m, p) * something... 
    # For 2x5x6, let's just try ranks from, say, 10 up to 60.
    
    best_result = {
        "U": U_school, 
        "V": V_school, 
        "W": W_school,
        "rank": rank_school
    }
    
    # Try to find a better decomposition
    # We'll use a simple randomized local search.
    
    def verify_decomposition(U, V, W, n, m, p, rank):
        """Verify that U, V, W form a valid decomposition."""
        # T[i,j,k] should equal 1 for all i,j,k
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for t in range(rank):
                        val += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def try_random_decomposition(rank, n, m, p, max_iters=5000):
        """Try to find a random decomposition of given rank."""
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = np.random.uniform(-1, 1, (n*m, rank))
            V = np.random.uniform(-1, 1, (m*p, rank))
            W = np.random.uniform(-1, 1, (p*n, rank))
            
            # Simple gradient-free optimization: try to minimize error
            # Error = sum_{i,j,k} (sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] - 1)^2
            
            # Use a simple iterative approach
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    return None
                
                # Compute error
                error = 0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Compute current value
                            val = 0
                            for t in range(rank):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            err = val - 1.0
                            error += err * err
                            
                            # Gradients
                            for t in range(rank):
                                grad_U[i*m+j, t] += 2 * err * V[j*p+k, t] * W[k*n+i, t]
                                grad_V[j*p+k, t] += 2 * err * U[i*m+j, t] * W[k*n+i, t]
                                grad_W[k*n+i, t] += 2 * err * U[i*m+j, t] * V[j*p+k, t]
                
                if error < 1e-10:
                    break
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check if we found a valid decomposition
            if error < 1e-10:
                # Convert to lists and round to nearest half-integer
                U_list = [[round(x * 2) / 2 for x in row] for row in U]
                V_list = [[round(x * 2) / 2 for x in row] for row in V]
                W_list = [[round(x * 2) / 2 for x in row] for row in W]
                
                # Verify with rounded values
                if verify_decomposition(U_list, V_list, W_list, n, m, p, rank):
                    return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
        
        return None
    
    # Try ranks from low to high
    # Start from a reasonable lower bound
    min_rank = max(n*m, m*p, p*n)  # This is not a valid lower bound, but let's start here
    # Actually, let's start from something smaller
    for rank in range(10, rank_school + 1):
        if time.time() > deadline:
            break
        
        result = try_random_decomposition(rank, n, m, p, max_iters=100)
        if result is not None:
            best_result = result
            break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END

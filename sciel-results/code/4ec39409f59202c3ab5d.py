"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook decomposition
    rank0 = nm * mp // p  # This is n*m*p = 2*4*7 = 56
    # Actually schoolbook rank is n*m*p = 56
    rank0 = n * m * p
    
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook is correct
    # Tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For schoolbook, each term (i,j,k) has exactly one column c where all three are 1.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Try to reduce rank by finding linear dependencies
    # We can formulate this as: find rank r and matrices U,V,W such that
    # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] sums to 1 for all i,j,k
    
    # A simple approach: try random low-rank decompositions and check if they work
    # Or use the fact that for 2x4x7, there might be known lower bounds
    
    # The rank of matrix multiplication tensor M_{n,m,p} is at most n*m*p (schoolbook)
    # For 2x2x2, rank is 7 (Strassen)
    # For 2x4x4, rank is at most 14 (two 2x2x2 blocks)
    # For 2x4x7, we can write 7 = 2+2+2+1, so rank <= 3*7 + 2*4*1 = 21 + 8 = 29?
    # Or 7 = 4+3, rank(2x4x4) + rank(2x4x3)
    # rank(2x4x3) <= 2*4*3 = 24, but maybe lower
    
    # Let's try a simple random search for lower rank
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    deadline = time.time() + 100  # Leave safety margin
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound for rank of M_{2,4,7}: 
    # By the "slice rank" bound or other bounds, but let's just try empirically
    # Try ranks from 28 to 55
    
    np.random.seed(42)
    
    for r in range(28, rank0):
        if time.time() > deadline:
            break
        
        # Try random initialization
        for trial in range(5):
            if time.time() > deadline:
                break
            
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Check if this decomposition works
            # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
            
            # This is a hard constraint to satisfy with random matrices
            # So this approach won't work directly
            
            break  # Skip this approach
    
    # Better approach: use the known Strassen-like constructions
    # For 2x2x2, rank 7 is known
    # Let's construct a rank 28 decomposition by treating 4x7 as 2x2 blocks of 2x2 and 2x1
    
    # Actually, let's try a different approach:
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,2), j in [0,4), k in [0,7)
    # We want to express this as sum of rank-1 tensors
    
    # A rank-1 tensor has the form u_i * v_j * w_k
    
    # Let's try to find a rank 28 decomposition by hand or via known results
    
    # For now, return the schoolbook decomposition
    # In a real scenario, we would use known algebraic geometry results
    # The rank of M_{2,4,7} is known to be 28? Let's assume we can achieve 28
    
    # Actually, let's just return schoolbook for now, since finding a lower rank
    # decomposition is non-trivial and requires domain-specific knowledge
    
    # Convert to lists for the return format
    U_list = U0.tolist()
    V_list = V0.tolist()
    W_list = W0.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

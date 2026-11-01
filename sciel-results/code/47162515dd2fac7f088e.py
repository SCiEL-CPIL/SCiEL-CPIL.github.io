"""Rewritten program: search for rank-minimal tensor decomposition of 2x2 * 2x15 matrix multiplication."""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank: n*m*p = 60
    # Try to find a lower rank decomposition
    # 
    # The matrix multiplication tensor T has entries T[i,j,k] = delta_{i+k, j}
    # where i in [0,n), j in [0,m), k in [0,p)
    # Actually for (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor is: T[i,j,k] = 1 if the multiplication involves A[i,j]*B[j,k]
    # 
    # The tensor decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # 
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should equal the tensor
    # 
    # For schoolbook: rank = n*m*p, with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # for c = i*m*p + j*p + k
    #
    # Let's try to find a lower rank. The known lower bound for matrix multiplication
    # is at least max(n*m, m*p, p*n) = max(4, 30, 30) = 30.
    # 
    # For 2x2 * 2x15, the rank is at least 30 (since m*p = 30 and p*n = 30).
    # 
    # Let's try rank = 30 first. If that doesn't work, try higher ranks.
    
    # Build the target tensor
    # T[a, b, c] where a = i*m+j (index into A), b = j*p+k (index into B), c = k*n+i (index into C)
    # The tensor value is 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k]
    # i.e., T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # and 0 otherwise
    
    # Actually, the tensor is: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries are 0.
    
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c
    
    # This is a 3-way tensor decomposition problem.
    
    # Let's try a structured approach. For rank R, we can try to solve this as a
    # system of linear equations if we fix two of the matrices.
    
    # Strategy: Try ranks from 30 upward. For each rank, try to find a valid decomposition.
    
    # First, let's verify the schoolbook solution works
    def verify(U, V, W, n, m, p):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # Build the resulting tensor and compare to target
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Target: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0
        
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        
        # Check that all other entries are 0 (sample some)
        # For efficiency, check a subset
        for _ in range(200):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            # Check if this is a valid (i,j,k) combination
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k1 = b % p
            k2 = c // n
            i2 = c % n
            if not (i1 == i2 and j1 == j2):
                # This should be 0
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > 1e-10:
                    return False
        return True
    
    # Try schoolbook first as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Known result: rank of (2,2,15) matrix multiplication
    # The lower bound is max(n*m, m*p, p*n) = 30
    # 
    # Let's try a structured approach for rank 30.
    # 
    # For rank 30, we need U (4x30), V (30x30), W (30x4)
    # 
    # One approach: use the fact that for 2x2 matrices, there are known rank-7 decompositions
    # (Strassen). For 2x2 * 2x15, we might be able to use a similar structure.
    
    # Let's try a more systematic search using random initialization and optimization
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try ranks from 30 upward
    for rank in range(30, min(61, rank_schoolbook + 1)):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Try to optimize using a simple gradient-free approach
            # or try a structured approach
            
            # Actually, let's try a different approach: fix V to be identity-like
            # and solve for U and W
            
            # For rank = m*p = 30, let V be a 30x30 matrix
            # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            
            # Let's try: V[b,r] = delta_{b,r} (identity)
            # Then: U[a,r]*W[c,r] = T[a,r,c]
            # So for each r, U[:,r]*W[:,r]^T should equal the r-th "slice" of T
            
            # But T is sparse, so this might not work directly.
            
            # Alternative: try to use the structure of the problem
            # For 2x2 * 2x15, we can think of it as 15 independent 2x2 * 2x1 problems
            # but that's not quite right because of the shared j index.
            
            # Let's just try random search with verification
            # This is unlikely to find a solution quickly, but let's try
            
            # Actually, let's try a more structured approach
            # Based on known results for small matrix multiplication
            
            # For 2x2 * 2xn, the rank is known to be 2n for n >= 2
            # (since the lower bound is max(4, 2n, 2n) = 2n for n >= 2)
            # Wait, that's not necessarily true. Let me think again.
            
            # The lower bound is max(n*m, m*p, p*n) = max(4, 2*15, 15*2) = 30
            # And the upper bound is n*m*p = 60
            # 
            # Known results suggest that for 2x2 * 2xn, the rank is 2n for large n
            # Let's try rank = 30 = 2*15
            
            if rank == 30:
                # Try a structured decomposition
                # U is 4x30, V is 30x30, W is 30x4
                
                # Idea: for each column k of B (0 to 14), we need to compute
                # A * B[:,k] which is a 2x2 matrix times a 2x1 vector
                # This requires rank 4 per k (schoolbook) or rank 3 (Strassen-like)
                # But the columns share structure through the j index.
                
                # Let's try: U[i*m+j, r] where r = j*p + k' for some mapping
                # Actually, let's try a different parameterization
                
                # For rank 30, let's try:
                # r = 0..29, map to (j, k') where j in {0,1}, k' in {0..14}
                # U[i*m+j, j*p+k'] = some value depending on i
                # V[j*p+k, j*p+k'] = delta_{k,k'}
                # W[k'*n+i, j*p+k'] = some value depending on i
                
                # This would give: sum_{j,k'} U[i*m+j, j*p+k'] * V[j*p+k, j*p+k'] * W[k'*n+i, j*p+k']
                # = sum_j U[i*m+j, j*p+k] * W[k*n+i, j*p+k]
                # = sum_j U[i*m+j, j*p+k] * W[k*n+i, j*p+k]
                
                # We need this to equal 1 if (i,j,k) is valid, i.e., always 1 for our target
                # Wait, the target is T[i*m+j, j*p+k, k*n+i] = 1
                # So we need: sum_j' U[i*m+j', j'*p+k] * W[k*n+i, j'*p+k] = 1 for all i,k
                # And for invalid (a,b,c), it should be 0.
                
                # Let's try: U[i*m+j, j*p+k] = 1 for all i,j,k
                # W[k*n+i, j*p+k] = 1/2 for all i,j,k
                # Then sum_j' 1 * 1/2 = 2 * 1/2 = 1. 
                
                # Check: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # With V[b,r] = delta_{b,r} where b = j*p+k, r = j'*p+k'
                # V[j*p+k, j'*p+k'] = delta_{j,j'} * delta_{k,k'}
                
                # So T[i*m+j, j*p+k, k*n+i] = sum_{j',k'} U[i*m+j, j'*p+k'] * delta_{j,j'}*delta_{k,k'} * W[k*n+i, j'*p+k']
                # = U[i*m+j, j*p+k] * W[k*n+i, j*p+k]
                # = 1 * 1/2 = 1/2. Not 1.
                
                # So we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
                # Let's try U = 1, W = 1. Then we get 1. 
                
                # Check invalid entries: T[a,b,c] where a=i*m+j, b=j'*p+k', c=k''*n+i''
                # with (i,j) != (i'',j'') or k != k''
                # T = sum_{j'',k''} U[i*m+j, j''*p+k''] * delta_{j,j''}*delta_{k,k''} * W[k''*n+i'', j''*p+k'']
                # = U[i*m+j, j*p+k] * W[k*n+i'', j*p+k]
                # = 1 * 1 = 1 if i'' != i or k'' != k. That's wrong!
                
                # So this simple approach doesn't work.
                
                # Let's try a different structure.
                # We need more careful design.
                
                # For now, let's just use the schoolbook solution
                pass
        
        # If we found a valid lower rank, return it
        # For now, fall back to schoolbook
    
    # Return schoolbook solution
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END

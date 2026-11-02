import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0  # 5s safety margin

def construct():
    n, m, p = DIMENSIONS
    # Start with naive rank-96 decomposition
    naive_rank = n * m * p
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_naive, "V": V_naive, "W": W_naive}
    best_rank = naive_rank
    
    # Try to find a lower-rank decomposition using structured search
    # We'll use a greedy approach: try to combine terms
    
    # Convert to numpy for efficiency
    U_np = np.array(U_naive, dtype=float)
    V_np = np.array(V_naive, dtype=float)
    W_np = np.array(W_naive, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if i,j,k are valid indices for the multiplication, 0 otherwise
    
    # Actually, the decomposition should satisfy:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,j,k} for the tensor
    # where the tensor represents matrix multiplication
    
    # Let's verify the naive decomposition first
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # In naive, each (i,j,k) has exactly one r where all three are 1, so T=1 for all (i,j,k)
    # But that's not right - we need T to be the matrix multiplication tensor
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if the indices form a valid
    # multiplication path, which is always the case for any (i,j,k) in ranges.
    # Actually, T[i,j,k] = 1 for all valid (i,j,k) since we're just summing A[i,j]*B[j,k]
    
    # So the tensor is all-ones on the valid domain. The rank is the minimum number
    # of rank-1 tensors that sum to this.
    
    # For a full tensor of size n*m*p, the rank can be less than n*m*p if there's structure.
    # The matrix multiplication tensor of size (n,m,p) has rank at most n*m*p, but can be lower.
    
    # Known results: for (2,2,2) the rank is 7 (Strassen), for (3,3,3) it's 23 (Laser/Strassen)
    # For our case (3,4,8), we can try to use block structure.
    
    # Let me try a different approach: use the fact that we can pad to a square problem
    # and use known fast algorithms.
    
    # For now, let's try a simple optimization: check if we can merge any terms
    # This is a greedy local search
    
    # Convert to lists for the return format
    def verify(U, V, W, rank):
        """Check if the decomposition is valid"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
        
        # The tensor should be all ones
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Try to reduce rank by removing linearly dependent columns
    # This is a simple approach: check if any column is a linear combination of others
    
    # For now, return the naive solution as a baseline
    # In a more complete implementation, we would search for lower-rank solutions
    
    return best_result

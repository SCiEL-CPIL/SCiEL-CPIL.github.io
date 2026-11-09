"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build schoolbook decomposition
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
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T has entries T[i,j,k] = delta_{j,k} for (i,j) in U, (j,k) in V, (k,i) in W
    # Actually T[i,j,k] = 1 if the column corresponds to (i,j,k)
    
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if j==k else 0... 
    # Wait, the tensor for matrix multiplication is T_{i,j,k} where (i,j) indexes first matrix, (j,k) second
    # T_{i,j,k} = delta_{j,k}... no.
    
    # Matrix multiplication: C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]... 
    # Actually the bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    # As a 3-tensor: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No.
    
    # The standard embedding: U has rows indexed by (i,j), V by (j,k), W by (k,i)
    # For each term in the sum: A[i,j]*B[j,k] contributes to C[i,k]
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Wait, that's not right either. Let me re-read the contract.
    
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # The decomposition should satisfy: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,k}... 
    # No, for matrix multiplication it should be:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # If we write A[i,j] and B[j,k] as inputs, and C[i,k] as output,
    # then the tensor T[i,j,k] satisfies: T[i,j,k] = 1 if we use A[i,j] and B[j,k] to compute C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
    
    # Wait, that means the tensor is all ones? That can't be right for rank minimization.
    
    # Let me reconsider. The matrix multiplication tensor T_{n,m,p} has:
    # T[i,j,k] = 1 if j==k... no.
    
    # Standard definition: T_{n,m,p} in R^{n*m} x R^{m*p} x R^{n*p}
    # T[(i,j),(j',k),(i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} * (1 if j==k else 0)... 
    # No.
    
    # Correct: T[(i,j),(j',k),(i',k')] = delta_{i,i'} * delta_{k,k'} * delta_{j,j'} * delta_{j,k}
    # So T[(i,j),(j',k),(i',k')] = 1 if i=i', k=k', j=j', and j=k, else 0
    # This means T[(i,j),(j,k),(i,k)] = 1, and 0 otherwise.
    
    # So the tensor has 1s at positions where the middle index matches: (i,j),(j,k),(i,k)
    # And 0 elsewhere.
    
    # For the decomposition: sum_c U[(i,j),c] * V[(j',k),c] * W[(i',k'),c] should equal T[(i,j),(j',k),(i',k')]
    # = delta_{i,i'} * delta_{k,k'} * delta_{j,j'} * delta_{j,k}
    
    # In schoolbook: for each (i,j,k), we have a column c with U[(i,j),c]=1, V[(j,k),c]=1, W[(i,k),c]=1
    # Then sum_c U[(i,j),c] * V[(j',k),c] * W[(i',k'),c] 
    # = 1 if there exists c such that U[(i,j),c]=1, V[(j',k),c]=1, W[(i',k'),c]=1
    # In schoolbook, column (i,j,k) has U[(i,j)]=1, V[(j,k)]=1, W[(i,k)]=1
    # So the sum is 1 iff i=i', j=j', k=k' (and j=k from the V index)... 
    # Wait, V index is (j',k), and in schoolbook V[(j,k),c]=1 for column (i,j,k)
    # So V[(j',k),c]=1 iff j'=j and k=k (the k in V index matches the k in column)
    # Similarly W[(i',k'),c]=1 iff i'=i and k'=k
    # And U[(i,j),c]=1 iff i=i and j=j
    # So the product is 1 iff i=i', j=j', k=k', and also j=k (from V index constraint)
    # But we need delta_{j,k} which is 1 iff j=k, so this works!
    
    # So the tensor T[(i,j),(j',k),(i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} * delta_{j,k}
    
    # Now to reduce rank, we need to find linear combinations of columns that still satisfy this.
    
    # Let's try a different approach: use known results or try specific constructions.
    # For 2x3x10, let's try to find if we can do better than 60.
    
    # One approach: use the fact that we can have half-integer entries.
    # Try to find a decomposition of lower rank by solving a system.
    
    # For now, return schoolbook as fallback, but try optimization if time permits.
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by merging columns where possible
    # This is complex; for now, return schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

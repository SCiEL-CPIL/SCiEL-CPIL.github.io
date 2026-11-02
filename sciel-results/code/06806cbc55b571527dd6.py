import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Schoolbook rank as baseline
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # The multiplication tensor T[i,j,k] = delta[i,k] (for C[i,k] = sum_j A[i,j]*B[j,k])
    # We need: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is W[k*n+i, r], so W index is (k,i)
    
    # Actually, re-reading: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to the tensor
    # The tensor for matrix mult: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So the target tensor has T[i,j,k] = 1 for all valid i,j,k (since each product contributes)
    
    # Wait, no. The matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the bilinear form has terms A[i,j]*B[j,k] for each (i,j,k)
    # The tensor T[i,j,k] = 1 for all i,j,k (each triple contributes exactly one product)
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a tensor decomposition problem.
    
    # For a systematic approach, let's try to use known constructions.
    # For 5x5x5, one approach is to use the fact that we can sometimes merge terms.
    
    # Let me try a different approach: use a structured decomposition.
    # We can write the multiplication as a sum of rank-1 tensors where each term
    # corresponds to a specific pattern.
    
    # Actually, let me just return the schoolbook decomposition for now,
    # but try to improve it with a simple optimization.
    
    # Simple improvement: check if we can use fewer terms by exploiting linearity.
    # For fixed j, the terms A[i,j]*B[j,k] for all i,k form a rank-1 matrix in (i,k) space.
    # So for each j, we have n*p = 25 terms that are all "aligned" in a certain way.
    
    # The key insight: for each j, the matrix A[:,j] * B[j,:] is rank 1 in (i,k) space.
    # So we can group by j and use n*p terms per j, total m*n*p = 125.
    
    # To reduce rank, we need to find linear dependencies or use more clever combinations.
    
    # Let me try a brute-force approach with a time limit:
    # Start with schoolbook and try to reduce by finding redundant terms.
    
    # Actually, for 5x5x5, the best known rank is 100 (from Strassen-like constructions).
    # Let me try to construct a rank-100 decomposition.
    
    # One approach: use the 4x4x4 rank-49 algorithm and extend to 5x5x5.
    # But this is complex. Let me try a simpler constructive approach.
    
    # Alternative: use a grid-based construction where we partition the indices.
    
    # For now, let me return the schoolbook decomposition which is guaranteed correct.
    rank = rank_schoolbook
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

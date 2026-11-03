"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = 84
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using random initialization
    # and iterative refinement. This is a heuristic search.
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    deadline = time.time() + 100  # Safety margin
    
    # Strategy: Try to reduce rank by looking for dependencies
    # We use a simple approach: start with schoolbook, then try to
    # remove columns that are linear combinations of others.
    
    # First, let's verify the schoolbook decomposition works
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For each column c in schoolbook: U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # This means the tensor entry T[i,j,k] gets contribution 1 from the column
    # where (i,j,k) matches, so T[i,j,k] = 1 for all (i,j,k).
    
    # The matrix multiplication tensor has T[i,j,k] = 1 if we're computing (A@B)[i,k]
    # where A is n x m and B is m x p. So (A@B)[i,k] = sum_j A[i,j]*B[j,k].
    # The tensor is T[i,j,k] = 1 (it's a 3-tensor with all entries 1? No.)
    
    # Actually, the matrix multiplication tensor is defined as:
    # M[i,j,k] = 1 for all i,j,k, and the multiplication is:
    # C[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
    
    # Wait, let me re-read the task. The tensor decomposition of matrix multiplication
    # means we want to represent the bilinear map (A,B) -> A@B as a sum of rank-1
    # tensors. The standard form is:
    # (A@B)[i,k] = sum_c sum_j U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
    
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta_{j,j'} for each (i,k)
    # Actually, the condition is:
    # sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c] = delta_{j,j'}
    
    # This is the standard matrix multiplication tensor.
    
    # Let's try a simple search: random rank-r decompositions and check if they work.
    # But this is very hard to do randomly.
    
    # Alternative: use the known result that for 3x4x7, the rank might be lower than 84.
    # Known bounds: rank(3x4x7) is between... let's just try to find a better one.
    
    # For now, return the schoolbook decomposition.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

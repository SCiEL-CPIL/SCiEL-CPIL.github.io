"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    
    # Build the matrix multiplication tensor T[i,j,k] where:
    # T[i,j,k] = 1 if k corresponds to output position (i,j) in C = A@B
    # For C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: C[i,k] = sum_r U[i*m+j_r, r] * V[j_r*p+k, r]
    # The tensor M[i,k,j] = 1 if this term contributes to C[i,k] from A[i,j]*B[j,k]
    
    # Flatten: index a = i*m + j (input A), index b = j*p + k (input B), index c = k*n + i (output C)
    # M[a,b,c] = 1 if there exists i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # This is the matrix multiplication tensor
    
    # We want to find rank-r decomposition:
    # M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Strategy: Use the fact that matrix multiplication has structure.
    # For 3x3x7, we can try to find a low-rank decomposition.
    
    # First, let's build the tensor as a matrix and find its rank.
    # Reshape M as a matrix of shape (n*m, m*p*n) or similar.
    
    # Actually, let's think about this differently.
    # The matrix multiplication tensor for A(n x m) * B(m x p) = C(n x p)
    # has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where the output is C[i,k] = sum_j A[i,j] B[j,k]
    
    # The tensor M has shape (n, m, p) with M[i,j,k] = 1
    # But we need to be careful about the indexing convention.
    
    # Output factor W[k*n+i, r] means W is indexed by (k, i) -> output position
    # So W[c, r] where c = k*n + i
    
    # Let's construct the tensor T[i,j,k] = 1 for all valid i,j,k
    # Then we need to decompose it.
    
    # The trivial decomposition has rank n*m*p = 63
    # But we can do better using the structure of matrix multiplication.
    
    # For 3x3x3, the minimum rank is known to be 21 (Landsberg's bound or similar)
    # For 3x3x7, let's try to find a good decomposition.
    
    # Approach: Use the fact that we can share some terms.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as a sum of rank-1 terms.
    
    # Let's try to construct a decomposition with rank < 63.
    # One approach: use the fact that for fixed j, the terms A[i,j]*B[j,k] 
    # form an outer product structure.
    
    # Actually, let's just use the trivial decomposition but see if we can compress.
    # The key insight: if two terms have the same (i,j) or (j,k) pattern, we might merge them.
    
    # Let's try a different approach: use QR decomposition on the flattened tensor.
    
    # Build the tensor as a 3D array
    T = np.ones((n, m, p))
    
    # Flatten to a matrix: reshape to (n*m, p) for each j? No, that's not right.
    
    # The tensor M[i,j,k] = 1 for all i,j,k
    # We want M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's try to find a decomposition by solving a system of equations.
    # But that's complex. Let's use a simpler approach.
    
    # For now, let's use the trivial decomposition with rank = n*m*p
    # But we can potentially reduce it.
    
    # Let's try rank = n*p + m*p + n*m - some savings
    # Actually, let's just use the trivial one for now and see if we can improve.
    
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

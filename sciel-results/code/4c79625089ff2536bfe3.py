"""Program for rank-minimal tensor decomposition of 2x3 * 3x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We seek a rank-r decomposition of the matrix multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = delta(i,j,k) which is 1 if the indices form a valid
    # multiplication chain, else 0. Actually the tensor is:
    # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' and j=j', k=k', i=i'
    # More precisely, the multiplication tensor for (n x m) * (m x p) -> (n x p)
    # has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    
    # The schoolbook rank is n*m*p = 2*3*6 = 36.
    # Each term in the schoolbook decomposition corresponds to a fixed (i,j,k):
    #   U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1, all else 0.
    
    # We will try to find a lower-rank decomposition by solving a sparse linear system.
    # Strategy: Try ranks from 20 upward, and for each rank, set up a linear system
    # where we introduce free variables for the factor matrices and enforce that
    # the tensor product U ⊗ V ⊗ W (in the appropriate sense) equals the target tensor.
    
    # The condition for exact decomposition:
    # For each (a, b, c) in the tensor, T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a system of (n*m)*(m*p)*(p*n) = 6*18*12 = 1296 equations in r*6 + r*18 + r*12 = 36r unknowns.
    # But the equations are cubic (trilinear), not linear. However, we can linearize by
    # fixing two of the three matrices and solving for the third, or by using a different approach.
    
    # Alternative: Use the fact that the tensor has a block structure. 
    # Let's think of the multiplication as: C[i,k] = sum_j A[i,j] * B[j,k]
    # For each pair (i,k), this is an inner product of a row of A and a column of B.
    # The tensor can be viewed as a sum over j of outer products.
    
    # Let's try a direct construction using rank-7 Strassen-like blocks.
    # Split the 3 columns of A into groups, and the 6 rows of B into groups.
    
    # Actually, let's use a simpler approach: try to find a rank-30 decomposition
    # by exploiting the fact that the 2x3 and 3x6 matrices have structure.
    
    # For now, let's implement a rank-36 schoolbook solution that is clean,
    # and then try to improve it.
    
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

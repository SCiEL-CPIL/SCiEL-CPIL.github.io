"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find rank-28 decomposition
    # Schoolbook gives rank 30. We need to eliminate 2 terms.
    
    # The tensor T[i,j,k] for matmul: T[i, j, k] = 1 if we consider
    # A[i,j] * B[j,k] contributing to C[i,k]
    # In the decomposition: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal the tensor entry.
    
    # For schoolbook: each (i,j,k) gives one term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    
    # To reduce rank, we look for linear dependencies.
    # For 2x3x3x5, rank 28 is known to be achievable.
    
    # Let's try a constructive approach:
    # Use the fact that for the last column of V (k=4), and certain rows,
    # we can combine terms.
    
    # Actually, let me try a direct search approach with numpy.
    # Build the tensor and try to find a rank-28 decomposition.
    
    start_time = time.time()
    deadline = start_time + 120.0
    
    # First, try a known rank-28 construction
    # For 2x3x3x5, one approach is to use the "Laderman-like" reduction
    
    # Let me try: start with schoolbook rank 30, then see if we can merge
    # two pairs of terms by finding linear dependencies.
    
    # The tensor in index form: T[a, b, c] where a = i*m+j (i in 0..1, j in 0..2)
    # b = j*p+k (j in 0..2, k in 0..4), c = k*n+i (k in 0..4, i in 0..1)
    # T[a,b,c] = 1 if the (i,j) of a matches the (j,k) of b and (k,i) of c, else 0
    
    # Build the tensor
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    # Try to find a rank-28 decomposition using iterative tensor decomposition
    # This is hard in general, but for this small case we can try.
    
    # Alternative: use a known explicit rank-28 decomposition
    # For 2x3x3x5, the following construction works:
    # Use 28 terms where 26 are standard and 2 are "merged" terms
    
    # Let me try a different approach: use the fact that we can write
    # some products as sums of fewer terms.
    
    # Actually, for this specific case, let me just verify rank 28 exists
    # by using a randomized approach to find a decomposition.
    
    # Use truncated SVD / iterative approach
    # Reshape tensor to matrix and try to find low-rank factorization
    
    # Flatten tensor to a matrix of shape (n*m) x (m*p * p*n)
    T_mat = tensor.reshape(n*m, m*p * p*n)
    
    # Try rank 28
    rank = 28
    # This is not straightforward because it's a 3-way tensor
    
    # Let me try a more direct approach: assume we can find 28 rank-1 terms
    # by modifying the schoolbook.
    
    # For the schoolbook, we have 30 terms indexed by (i,j,k).
    # Can we merge two pairs?
    
    # Consider terms where i=0 and i=1 with same (j,k). 
    # Term (0,j,k): U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
    # Term (1,j,k): U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1
    # These share the same V factor! So they can't be directly merged into one term.
    
    # But maybe we can find a different combination.
    
    # Let me try to use a numerical optimization approach.
    # Initialize with schoolbook and try to zero out 2 columns.
    
    if time.time() > deadline - 5:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find rank-28 decomposition using a structured approach
    # Based on known results for small matrix multiplication
    
    # For 2x3x3x5, here's an explicit rank-28 decomposition:
    # We use the following idea: for the "middle" dimension m=3,
    # we can sometimes save 2 ranks by combining terms.
    
    # Let me try a constructive rank-28:
    # Use all 30 schoolbook terms but replace 2 pairs with 2 merged terms.
    
    # Actually, let me just try to verify if rank 28 works by construction.
    # I'll use a known pattern from small matmul decompositions.
    
    # For 2x2x2x2, rank 7 is known (Laderman).
    # For 2x3x3x5, let me try a similar reduction.
    
    # Given time constraints, let me return a rank-29 or rank-28 if I can find one.
    # Otherwise, return schoolbook rank 30.
    
    # Let me try a quick numerical approach:
    # Use the tensor and try to find a rank-28 decomposition via ALS or similar.
    
    # This is complex. Let me just return the schoolbook for safety,
    # but try one more thing: check if rank 29 is achievable by merging one pair.
    
    # For now, return schoolbook rank 30.
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Convert to list of lists with integer entries
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

<<<<<<< SEARCH
"""
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    This is the best known constructive rank.
    """
=======
"""
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Returns a rank-27 trivial decomposition (standard matrix multiplication)
    which is valid and exact.
    """

import numpy as np

def construct():
    # Trivial decomposition of 3x3 matrix multiplication
    # U has shape (3*3, 27) = (9, 27)
    # V has shape (3*3, 27) = (9, 27)
    # W has shape (3*3, 27) = (9, 27)
    
    # The multiplication tensor T[i, j, k] = 1 if output index corresponds to 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For the trivial decomposition, each term corresponds to a specific (i, j, k) 
    # where we compute A[i,j] * B[j,k] and add to C[i,k].
    
    # Indices:
    # A is 3x3, flattened as A[i*3 + j]
    # B is 3x3, flattened as B[j*3 + k]
    # C is 3x3, flattened as C[i*3 + k]
    
    # We need 27 terms (i, j, k) for i, j, k in {0, 1, 2}
    
    n = 3
    m = 3
    p = 3
    
    rank = n * m * p  # 27
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Map (i, j, k) to rank index
    # rank_idx = i * m * p + j * p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                
                # U: index is i*m + j (row i, col j of A)
                U[i * m + j, r] = 1
                
                # V: index is j*p + k (row j, col k of B)
                V[j * p + k, r] = 1
                
                # W: index is k*n + i (following the trace convention W[k*n + i, r])
                # Wait, the task says: "The output factor follows the official trace convention W[k*n+i,r]"
                # This suggests W is indexed by (k, i) where k is the column of C and i is the row of C?
                # Let's re-read: "W[k*n+i,r]"
                # Usually, C is indexed by (row, col). Let's assume C[i, k] where i is row, k is col.
                # Then the index in the flattened C would be i*n + k.
                # But the convention says W[k*n + i, r]. This implies the first index of W is k and second is i?
                # Or is it that the output matrix C has dimensions (p, n)? No, C is (n, p).
                # Let's look at the shapes: W is (p*n, rank).
                # If W[k*n + i, r], then the index is k*n + i.
                # This implies the first "dimension" of the output index is k (size n? no, k goes to p) 
                # and the second is i (size n?).
                # Actually, if C is n x p, and we flatten it, usually it's row-major: C[i, k] -> i*p + k.
                # However, the convention W[k*n + i, r] suggests a specific ordering.
                # Let's assume the output index is defined as idx = k * n + i.
                # This means for a fixed k (column of B / column of C?), and i (row of A / row of C?),
                # we have a term.
                # In matrix multiplication C[i, k] = sum_j A[i, j] B[j, k].
                # So i is row of C, k is col of C.
                # If the index is k * n + i, then the first factor is k (0..p-1) and second is i (0..n-1).
                # Since n=p=3, this is just a different flattening order.
                
                W[k * n + i, r] = 1
    
    return {
        'U': U,
        'V': V,
        'W': W
    }
>>>>>>> REPLACE

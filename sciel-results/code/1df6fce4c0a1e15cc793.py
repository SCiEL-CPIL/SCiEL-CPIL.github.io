"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for
    multiplying a 2x2 matrix by a 2x15 matrix.
    
    We use Strassen's algorithm on the 2x2x2 block structure where possible,
    and combine with standard multiplication for the remaining dimensions.
    
    The matrix multiplication tensor has dimensions (n, m, p) = (2, 2, 15).
    The tensor T_{i,j,k} = 1 if (i,j,k) corresponds to a valid multiplication term.
    Specifically, T[i, j, k] = 1 means the (i,k) entry of the output involves
    the product of A[i,j] and B[j,k].
    
    For 2x2 * 2x15:
    - A is 2x2, B is 2x15, C is 2x15
    - C[i,k] = sum_j A[i,j] * B[j,k]
    - Total terms: 2*2*15 = 60 (schoolbook)
    
    We can apply Strassen to the 2x2 part (treating the 15 columns as vectors):
    Let A = [[a11, a12], [a21, a22]]
    Let B_j = [B[j,0], B[j,1], ..., B[j,14]] for j in {0,1}
    
    Strassen for 2x2 * 2x1:
    M1 = (a11+a22)*(b11+b22)
    M2 = (a21+a22)*b11
    M3 = a11*(b21-b11)
    M4 = a22*(b12-b22)
    M5 = (a11+a12)*b22
    M6 = a21*(b12-b22)
    M7 = a12*(b11+b21)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 2x2 * 2x15, we can treat each column independently but share the
    Strassen structure across columns. Actually, since B is 2x15, we can
    apply Strassen per column, but that gives 7*15 = 105 terms, which is worse.
    
    Better approach: Use the fact that we're multiplying 2x2 by 2x15.
    We can decompose B into 15 separate 2x1 matrices, but that's not helpful.
    
    Alternative: Use the rank of the matrix multiplication tensor.
    For m x n * n x p, the rank is at most m*n*p (schoolbook).
    For 2x2 * 2x15, schoolbook rank = 60.
    
    Using Strassen on the 2x2 part: we can view this as computing
    A * B where A is 2x2 and B is 2x15.
    
    Let's define the tensor more carefully.
    T is a 3-tensor with indices (i, j, k) where i in {0,1}, j in {0,1}, k in {0,...,14}.
    T[i,j,k] = 1 for all valid (i,j,k).
    
    We want to write T as a sum of rank-1 tensors:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, the problem states the output factor follows W[k*n+i, r].
    So W has indices (k, i) flattened to k*n+i.
    
    Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    So W is of shape (p*n, rank) = (15*2, rank) = (30, rank).
    U is of shape (n*m, rank) = (2*2, rank) = (4, rank).
    V is of shape (m*p, rank) = (2*15, rank) = (30, rank).
    
    The tensor decomposition is:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the schoolbook solution, each term (i,j,k) is its own rank-1 component:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    Now, can we do better than 60?
    
    For 2x2 * 2x1, the minimal rank is 7 (Strassen).
    For 2x2 * 2x15, we can potentially use a similar approach.
    
    Idea: Group the 15 columns into groups and apply Strassen-like tricks.
    But 15 is not a power of 2, so this is tricky.
    
    Alternative: Use the fact that the tensor for 2x2 * 2xp has rank at most
    7*p (applying Strassen per column) = 105, which is worse than 60.
    
    So schoolbook (rank 60) is actually better than per-column Strassen.
    
    Can we do better than 60? The theoretical lower bound for 2x2 * 2xp is
    at least 2*2*p = 4p = 60? No, that's not a lower bound.
    
    Actually, for matrix multiplication of m x n * n x p, the rank is at least
    max(m*n, n*p, m*p) in some cases, but for 2x2 * 2x15, the lower bound
    is not clear.
    
    Let me think about this differently. The tensor T has 60 non-zero entries.
    Each rank-1 component can cover at most one entry if we use the standard
    form. But with Strassen, we can cover multiple entries with one component
    by using linear combinations.
    
    For 2x2 * 2x1, Strassen uses 7 components to cover 4 entries (each output
    entry is a linear combination of the 7 products).
    
    For 2x2 * 2x15, we have 30 output entries (2*15). Each output entry C[i,k]
    is a sum of 2 products: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    
    So we have 30 output entries, each requiring 2 products = 60 products total.
    
    Can we share products between different output entries? In Strassen for 2x2*2x1,
    the 7 products are shared between the 4 output entries.
    
    For 2x2 * 2x15, we can apply the same Strassen structure per column, giving
    7*15 = 105 products, which is worse.
    
    But maybe we can share across columns? If B[0,k] and B[1,k] are related for
    different k, we might share. But in general, they are independent.
    
    So it seems 60 is the best we can do with simple methods.
    
    However, there might be a more clever decomposition. Let me try to use
    numerical optimization to find a lower rank decomposition.
    
    Actually, for this specific case, I believe the rank is exactly 60, since
    the 15 columns are independent and each column requires the 2x2 * 2x1
    multiplication which has rank 7, but 7*15 = 105 > 60.
    
    Wait, that's not right. The rank of the tensor for 2x2 * 2x15 is not
    simply 15 times the rank for 2x2 * 2x1, because the tensor is not a
    direct sum in that way.
    
    Let me just implement the schoolbook solution with rank 60, and then
    try to optimize using numerical methods to see if we can find a lower rank.
    
    Given the time constraints, I'll implement the schoolbook solution first,
    and then add a simple optimization attempt if time permits.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

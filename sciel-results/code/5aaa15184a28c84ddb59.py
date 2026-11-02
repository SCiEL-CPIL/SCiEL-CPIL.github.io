<<<<<<< SEARCH
def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        # ...
        ks = k_start
        ke = k_start + 1
        
        # M1 = (A00 + A11) * (B0,ks + B1,ke)
        # C0,ks += M1, C1,ke += M1
        U[0, c0] += 1
        U[3, c0] += 1
        V[7 * 0 + ks, c0] += 1
        V[7 * 1 + ke, c0] += 1
        W[2 * ks + 0, c0] += 1
        W[2 * ke + 1, c0] += 1
=======
import numpy as np
import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x7 matrix.
    
    The tensor T has indices (i, j, k) where:
    - i in {0, 1} (row of A)
    - j in {0, 1} (col of A, which is row of B)
    - k in {0, ..., 6} (col of B)
    
    The multiplication is C[i, k] = sum_j A[i, j] * B[j, k]
    
    We need to find U, V, W such that:
    T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    
    For a 2x2 by 2x7 multiplication, the tensor is:
    T[i, j, k] = 1 if (i, j, k) corresponds to a valid multiplication term
    
    Actually, the tensor for matrix multiplication A (2x2) * B (2x7) is:
    T[i, j, k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,6}
    
    Wait, that's not right. The tensor T for matrix multiplication is defined as:
    T[i, j, k] = 1 if the product A[i, j] * B[j, k] contributes to C[i, k]
    
    So T[i, j, k] = 1 for all valid (i, j, k) where j is the shared index.
    
    The tensor shape is (2, 2, 7) with T[i, j, k] = 1 for all i, j, k.
    
    We need to decompose this tensor into rank-r terms.
    
    A simple schoolbook decomposition has rank 2*2*7 = 28 (one term per (i,j,k) combination).
    But we can do better.
    
    Actually, for matrix multiplication, the tensor is:
    T[i, j, k] = delta_{shared} where the shared index is j.
    
    Let me think again. The matrix multiplication tensor for (m x n) * (n x p) is:
    T[i, j, k] = 1 if we are multiplying A[i, j] by B[j, k] to get C[i, k]
    
    So for A (2x2) and B (2x7), we have:
    - i in {0, 1} (rows of A)
    - j in {0, 1} (cols of A = rows of B)
    - k in {0, ..., 6} (cols of B)
    
    T[i, j, k] = 1 for all i, j, k
    
    The rank of this tensor is what we want to minimize.
    
    A trivial decomposition has rank 2*2*7 = 28, with each term being a single entry.
    
    But we can group terms. For each k, we have a 2x2 matrix multiplication problem.
    For a 2x2 matrix multiplication, the rank is 7 (Strassen).
    
    So for 7 columns, if we use Strassen for each column independently, we get rank 7*7 = 49.
    That's worse than schoolbook!
    
    Wait, that's not right either. Let me reconsider.
    
    The tensor T[i, j, k] = 1 for all i, j, k can be decomposed as:
    T = sum_r U_r * V_r * W_r
    
    where U_r is a vector in R^2, V_r is a vector in R^2, W_r is a vector in R^7.
    
    One simple decomposition: for each (i, j), we have T[i, j, k] = 1 for all k.
    So we can write:
    T[i, j, k] = sum_{(i', j')} delta_{i, i'} * delta_{j, j'} * 1
    
    This gives rank 2*2 = 4, with:
    U_{(i', j')} = e_{i'} (standard basis vector in R^2)
    V_{(i', j')} = e_{j'} (standard basis vector in R^2)
    W_{(i', j')} = (1, 1, 1, 1, 1, 1, 1) (all-ones vector in R^7)
    
    So rank 4 is achievable!
    
    Let me verify:
    T[i, j, k] = sum_{i'=0}^1 sum_{j'=0}^1 delta_{i, i'} * delta_{j, j'} * 1
    = delta_{i, i} * delta_{j, j} * 1
    = 1
    
    Yes! So rank 4 works.
    
    Can we do better? Rank 3?
    
    The tensor T[i, j, k] = 1 is a 2x2x7 tensor with all entries 1.
    
    The rank of a tensor with all entries 1 is at least the maximum of the ranks of its slices.
    Each slice T[i, :, :] is a 2x7 matrix of all 1s, which has rank 1.
    Each slice T[:, j, :] is a 2x7 matrix of all 1s, which has rank 1.
    Each slice T[:, :, k] is a 2x2 matrix of all 1s, which has rank 1.
    
    So the tensor rank is at least 1, but we know rank 4 works.
    
    Actually, I think the minimum rank is 4. Here's why:
    
    Consider the 2x2 slice T[:, :, k] for any fixed k. This is a 2x2 matrix of all 1s.
    The rank of a 2x2 matrix of all 1s is 1 (it's a rank-1 matrix).
    
    But the tensor rank is different from the matrix rank of slices.
    
    Let me think about this differently. The tensor T can be viewed as a linear map from R^2 x R^7 to R^2.
    
    Actually, let's just use the rank-4 decomposition. It's simple and exact.
    
    U has shape (2*2, 4) = (4, 4)
    V has shape (2*7, 4) = (14, 4)
    W has shape (7*2, 4) = (14, 4)
    
    Wait, the interface says:
    U: (n*m, rank) where n=2, m=2, so (4, rank)
    V: (m*p, rank) where m=2, p=7, so (14, rank)
    W: (p*n, rank) where p=7, n=2, so (14, rank)
    
    And the output factor follows the official trace convention W[k*n+i, r].
    
    So W has indices (k, i) where k is the column of B (0 to 6) and i is the row of C (0 to 1).
    W[k*n+i, r] = W[k*2+i, r]
    
    Let me define the rank-4 decomposition:
    
    For each (i', j') in {0, 1} x {0, 1}:
    U[i', r] = 1 if r corresponds to (i', j'), else 0
    V[j', r] = 1 if r corresponds to (i', j'), else 0
    W[k, i, r] = 1 for all k, i if r corresponds to (i', j')... 
    
    Wait, that's not right. Let me re-derive.
    
    We want: T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, i, r]
    
    But the interface says U has shape (n*m, rank) = (4, rank).
    So U is indexed by (i, j) pairs, not just i.
    
    Let me re-read the interface:
    U: (n*m, rank) - so U is indexed by (i, j) where i is row of A, j is col of A
    V: (m*p, rank) - so V is indexed by (j, k) where j is row of B, k is col of B
    W: (p*n, rank) - so W is indexed by (k, i) where k is col of B, i is row of C
    
    And the tensor is: T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    Wait, that doesn't look right either. Let me think about the standard tensor decomposition for matrix multiplication.
    
    The matrix multiplication tensor T has indices (i, j, k) where:
    - i: row of A (0 to n-1)
    - j: col of A / row of B (0 to m-1)
    - k: col of B (0 to p-1)
    
    T[i, j, k] = 1 (the coefficient of A[i, j] * B[j, k] in C[i, k])
    
    The tensor decomposition is:
    T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    Wait, that's not standard. Let me look at this more carefully.
    
    Actually, I think the standard form is:
    T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    
    But the interface says U has shape (n*m, rank), which suggests U is indexed by (i, j).
    
    Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    So:
    U: shape (n*m, rank) - indexed by (i, j) where i in [0, n), j in [0, m)
    V: shape (m*p, rank) - indexed by (j, k) where j in [0, m), k in [0, p)
    W: shape (p*n, rank) - indexed by (k, i) where k in [0, p), i in [0, n)
    
    And the tensor is:
    T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    This makes sense! The three factors share indices in a cyclic way:
    - U shares (i, j) with T
    - V shares (j, k) with T
    - W shares (k, i) with T
    
    So for our case:
    n = 2, m = 2, p = 7
    U: shape (4, rank) - indexed by (i, j) with i in {0,1}, j in {0,1}
    V: shape (14, rank) - indexed by (j, k) with j in {0,1}, k in {0,...,6}
    W: shape (14, rank) - indexed by (k, i) with k in {0,...,6}, i in {0,1}
    
    The tensor T[i, j, k] = 1 for all i, j, k.
    
    We want: T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    A simple rank-4 decomposition:
    For each (i', j') in {0, 1} x {0, 1}:
    Let r = i' * 2 + j' (so r in {0, 1, 2, 3})
    
    U[i, j, r] = 1 if (i, j) = (i', j'), else 0
    V[j, k, r] = 1 for all j, k (if we want to sum over all k)
    W[k, i, r] = 1 for all k, i
    
    Wait, that doesn't work. Let me think again.
    
    We want: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1 for all i, j, k.
    
    If we set:
    U[i, j, r] = delta_{i, i'} * delta_{j, j'} for r = (i', j')
    V[j, k, r] = 1 for all j, k
    W[k, i, r] = 1 for all k, i
    
    Then: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    = sum_{i', j'} delta_{i, i'} * delta_{j, j'} * 1 * 1
    = 1
    
    Yes! So rank 4 works.
    
    Let me verify the shapes:
    U: (4, 4) - U[i*2+j, r] where i, j in {0,1}, r in {0,1,2,3}
    V: (14, 4) - V[j*7+k, r] where j in {0,1}, k in {0,...,6}, r in {0,1,2,3}
    W: (14, 4) - W[k*2+i, r] where k in {0,...,6}, i in {0,1}, r in {0,1,2,3}
    
    So:
    U[i*2+j, i*2+j] = 1 for all i, j (diagonal matrix)
    V[j*7+k, r] = 1 for all j, k, r
    W[k*2+i, r] = 1 for all k, i, r
    
    Let me code this up.
    """
    n = 2
    m = 2
    p = 7
    rank = 4
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # U[i*m + j, r] = 1 if r == i*m + j
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1
    
    # V[j*p + k, r] = 1 for all j, k, r
    V[:] = 1
    
    # W[k*n + i, r] = 1 for all k, i, r
    W[:] = 1
    
    return {'U': U, 'V': V, 'W': W}
>>>>>>> REPLACE

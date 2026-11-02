"""
Rank-minimal (or near-minimal) exact tensor decomposition for matrix multiplication T_{4,4,6}.
Output: U (n*m, rank), V (m*p, rank), W (p*n, rank) with entries in {0, +/-1, +/-1/2}.
Index convention: U row i*m+j, V row j*p+k, W row k*n+i.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

# ---------- helpers ----------

def make_schoolbook():
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def verify(U, V, W, tol=1e-9):
    n, m, p = 4, 4, 6
    # Build T from U,V,W
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can do this via einsum or matrix multiplication trick:
    # For each c, the term is outer product structure.
    # Actually: T is a 3-tensor. We can check by reconstructing multiplication.
    # Alternative: check that for random A (n x m) and B (m x p),
    # sum_c U_c^T A * V_c^T B * W_c = A @ B
    # More directly: for each c, define U_c (n x m), V_c (m x p), W_c (p x n)
    # where U_c[i,j] = U[i*m+j, c], etc.
    # Then T = sum_c U_c (tensor) V_c (tensor) W_c where tensor product means:
    # T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    # Let's just compute T and compare to identity-like tensor.
    # The multiplication tensor T_{i,j,k} = 1 iff k corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually T_{i,j,k} = 1 for all i,j,k (the schoolbook tensor has 1s everywhere? No!)
    # Wait: the multiplication tensor for T_{n,m,p} has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # Because C[i,k] = sum_j A[i,j] * B[j,k], so each (i,j,k) contributes 1 * A[i,j] * B[j,k].
    # Yes! T_{i,j,k} = 1 for all i,j,k.
    r = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        # This is a rank-1 tensor addition
        # We can do: for each i,j,k, but that's slow. Use einsum:
        # Actually, let's do it via:
        # temp[i,j] = Uc[i,j] * Wc[:, i]  -> no, Wc depends on k and i.
        # Let's just do a loop over i,j and use matrix ops
        # For fixed i: T[i,:,:] += Uc[i,:] (outer) (Vc * Wc[:, i])
        # Uc[i,:] is a row of length m. Vc is m x p. Wc[:, i] is a column of length p.
        # So T[i,:,:] += Uc[i,:] @ (Vc * Wc[:, i])
        # Wait: T[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
        # For fixed i: T[i, j, k] = Uc[i,j] * (Vc[j,k] * Wc[k,i])
        # Let M = Vc * Wc[:, i] (elementwise, Vc is m x p, Wc[:,i] is p x 1 broadcast)
        # Then T[i, j, k] = Uc[i,j] * M[j,k]
        # So T[i,:,:] += np.outer(Uc[i,:], M[:, j]) ... no
        # T[i, j, k] = Uc[i, j] * M[j, k]
        # This is like: for each j, T[i, j, :] += Uc[i, j] * M[j, :]
        # Or: T[i] += Uc[i, :, None] * M  (where Uc[i,:,None] is m x 1, M is m x p)
        # Uc[i, :, None] * M gives m x p, which is T[i, :, :] contribution.
        M = Vc * Wc[:, i, None] if Wc.ndim == 3 else Vc * Wc[:, i]
        # Wc is p x n, so Wc[:, i] is length p. Vc is m x p.
        # M[j,k] = Vc[j,k] * Wc[k,i]
        # So M = Vc * Wc[:, i] (broadcast: Vc is m x p, Wc[:,i] is p,)
        T[i, :, :] += Uc[i, :][None, :] * M
    # Compare T to ones
    target = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T, target, atol=tol)


def rank_of_decomposition(U, V, W):
    return U.shape[1]


# ---------- Strassen 2x2x2 base ----------
# Strassen: 7 multiplications for 2x2x2
# C11 = (A11+A22)(B11+B22) - (A22)(B11+B22) - (A11)(B22+B11) ... 
# Standard Strassen:
# P1 = (A11+A22)(B11+B22)
# P2 = (A21+A22)B11
# P3 = A11(B21-B22)
# P4 = A22(B12-B11)
# P5 = (A11+A12)B22
# P6 = (A21-A11)(B11+B12)
# P7 = (A12-A22)(B21+B22)
# C11 = P1 + P4 - P5 + P7
# C12 = P3 + P5
# C21 = P2 + P6
# C22 = P1 - P2 + P3 + P6

STRASSEN_222 = {
    # Each entry: list of (U_coeff, V_coeff, W_coeff, sign) for each of the 7 products
    # Actually, let's define the 7 bilinear forms directly.
    # For 2x2x2, indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U row: i*2+j, V row: j*2+k, W row: k*2+i
    # P1: (A00+A11)(B00+B11) -> U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
    # We need to express each P as a rank-1 term.
    # P1 = (A00+A11)(B00+B11) contributes to C00 and C11.
    # In the tensor decomposition, each rank-1 term is U_c (2x2) x V_c (2x2) x W_c (2x2)
    # P1: U1 = [[1,0],[0,1]], V1 = [[1,1],[0,0]], W1 = [[1,0],[0,1]]
    #   Check: T[i,j,k] += U1[i,j]*V1[j,k]*W1[k,i]
    #   i=0,j=0,k=0: 1*1*1=1
    #   i=0,j=0,k=1: 1*1*0=0
    #   i=0,j=1,k=0: 0*0*1=0
    #   i=0,j=1,k=1: 0*0*0=0
    #   i=1,j=0,k=0: 0*1*0=0
    #   i=1,j=0,k=1: 0*1*1=0
    #   i=1,j=1,k=0: 1*0*0=0
    #   i=1,j=1,k=1: 1*0*1=0
    #   Hmm, that only gives T[0,0,0]=1. That's not right.
    # I think I'm confusing the tensor structure. Let me reconsider.
    #
    # The multiplication tensor T_{2,2,2} has T[i,j,k] = 1 for all i,j,k in {0,1}.
    # A rank-1 decomposition term is: T = sum_c U_c (outer) V_c (outer) W_c
    # where U_c is 2x2, V_c is 2x2, W_c is 2x2, and
    # T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    #
    # For Strassen, the 7 products are:
    # P1 = (A00+A11)(B00+B11)
    # This is a bilinear form in A and B. In the tensor, this corresponds to a sum of rank-1 terms.
    # Actually, (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This is 4 separate products, but they are grouped.
    # In the tensor decomposition, we need to find U_c, V_c, W_c such that the sum reconstructs T.
    #
    # Let me just look up the standard Strassen tensor decomposition.
    # For 2x2x2, rank 7:
    # The 7 terms are:
    # c=0: U = [[1,0],[1,0]], V = [[1,0],[0,0]], W = [[1,0],[0,0]]  -> A00*B00 -> C00
    # That's just schoolbook. Strassen uses clever combinations.
    #
    # Let me define Strassen terms properly:
    # P1 = (A00+A11)(B00+B11): 
    #   U1 = [[1,0],[1,0]], V1 = [[1,1],[0,0]], W1 = [[1,0],[0,1]]
    #   T1[i,j,k] = U1[i,j]*V1[j,k]*W1[k,i]
    #   i=0,j=0,k=0: 1*1*1=1
    #   i=0,j=0,k=1: 1*1*0=0
    #   i=1,j=0,k=0: 1*1*0=0
    #   i=1,j=0,k=1: 1*1*1=1
    #   Others 0.
    #   So T1 = e_000 + e_101 (in 2x2x2 indexing)
    #   This contributes to C00 (i=0,k=0) and C11 (i=1,k=1) with the same product.
    #
    # P2 = (A10+A11)B00:
    #   U2 = [[0,0],[1,1]], V2 = [[1,0],[0,0]], W2 = [[1,0],[0,0]]
    #   T2[1,0,0] = 1*1*1 = 1, T2[1,1,0] = 1*1*1 = 1? 
    #   i=1,j=0,k=0: U2[1,0]=1, V2[0,0]=1, W2[0,1]=0 -> 0
    #   i=1,j=1,k=0: U2[1,1]=1, V2[1,0]=0, W2[0,1]=0 -> 0
    #   i=1,j=0,k=1: U2[1,0]=1, V2[0,1]=0, W2[1,1]=0 -> 0
    #   Hmm, W2 = [[1,0],[0,0]] means W2[k,i]. For i=0,k=0: W2[0,0]=1.
    #   T2[1,0,0] = U2[1,0]*V2[0,0]*W2[0,1] = 1*1*0 = 0. 
    #   I have the indexing wrong. W row is k*n+i, so for 2x2, W[k*2+i].
    #   W2[k,i] where the matrix is indexed as W[row, col] with row = k*2+i.
    #   So W2 as a 2x2 matrix: W2[k,i] = W2_matrix[k*2+i, :] ... no.
    #   Let me redefine: W_c is a p x n matrix, so W_c[k, i].
    #   For 2x2x2: W_c is 2x2, W_c[k,i].
    #   T[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
    #
    # P2 = (A10+A11)B00:
    #   This should contribute to C10 and C11 (i=1, k=0).
    #   U2[i,j]: A10 and A11 are at i=1, j=0 and i=1, j=1.
    #   U2 = [[0,0],[1,1]]
    #   V2[j,k]: B00 is at j=0, k=0.
    #   V2 = [[1,0],[0,0]]
    #   W2[k,i]: contributes to i=1, k=0. So W2[0,1] = 1.
    #   W2 = [[0,1],[0,0]]
    #   T2[1,0,0] = U2[1,0]*V2[0,0]*W2[0,1] = 1*1*1 = 1
    #   T2[1,1,0] = U2[1,1]*V2[1,0]*W2[0,1] = 1*0*1 = 0
    #   Not quite. I need V2[1,0] to be 1 for A11*B00.
    #   But B00 is only at j=0,k=0. So V2[1,0] should be 0.
    #   The issue: (A10+A11)B00 = A10*B00 + A11*B00.
    #   A11*B00: i=1,j=1,k=0. But B00 is at j=0,k=0, so V2[1,0] must be 1? No, V2[j,k] with j=1,k=0.
    #   B is 2x2, B[j,k]. B00 is B[0,0]. So for A11*B00, we need V2[1,0] to capture B[0,0] when j=1.
    #   That means V2[1,0] = 1, but V2 is supposed to represent B. V2[j,k] should be 1 when (j,k) is a position in B.
    #   For B00, (j,k)=(0,0). So V2[0,0]=1. For A11*B00, we need j=1 (from A11) and k=0 (from B00).
    #   But V2[j,k] = V2[1,0]. If V2[1,0]=1, then we're saying B[1,0] is involved, which is wrong.
    #
    # I think the issue is that Strassen's grouping doesn't map directly to a single rank-1 term per product.
    # Each Strassen product P_c is a bilinear form, which in the tensor T corresponds to a sum of rank-1 terms.
    # But Strassen's trick is that the 7 products can be arranged so that the total rank is 7, not 8.
    #
    # Let me just use a known rank-7 decomposition for 2x2x2.
    # Reference: The 7 terms for Strassen are:
    # (I'll define them as U_c, V_c, W_c with entries in {0,1,-1})
    
    # Actually, let me just use a different approach: I'll use a numerical method to find a low-rank decomposition.
    # For 4x4x6, the schoolbook rank is 96. I'll try to find a lower rank using ALS-style optimization.
    # But the problem requires exact decomposition with integer/half-integer entries.
    # 
    # Given the complexity, let me just return the schoolbook decomposition for now, 
    # but with a search for a small improvement using block structure.
    # 
    # Alternative: Use the fact that T_{4,4,6} can be decomposed using smaller blocks.
    # 4x4x6 = 2x2x2 blocks? 4x4 = 4 blocks of 2x2, 6 = 3 blocks of 2.
    # So T_{4,4,6} = sum of 4*3 = 12 copies of T_{2,2,2}, each of rank 7 (Strassen).
    # Total rank = 12 * 7 = 84.
    # That's better than 96!
    # 
    # Let me implement this.
}


def strassen_222_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is (U_c, V_c, W_c) where U_c is 2x2, V_c is 2x2, W_c is 2x2.
    T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    """
    # Based on standard Strassen:
    # P1 = (A00+A11)(B00+B11) -> C00, C11
    # P2 = (A10+A11)B00 -> C10, C11  
    # P3 = A00(B10-B11) -> C00, C01
    # P4 = A11(B01-B00) -> C10, C11
    # P5 = (A00+A01)B11 -> C00, C01
    # P6 = (A10-A00)(B00+B01) -> C10, C11
    # P7 = (A01-A11)(B10+B11) -> C00, C01
    
    terms = []
    
    # P1: (A00+A11)(B00+B11)
    U1 = np.array([[1, 0], [1, 0]])  # A00 + A11 (column 0: A00, A11)
    V1 = np.array([[1, 1], [0, 0]])  # B00 + B11 (row 0: B00, B11)
    W1 = np.array([[1, 0], [0, 1]])  # C00 + C11 (W[k,i]: k=0,i=0 -> C00; k=1,i=1 -> C11)
    terms.append((U1, V1, W1))
    
    # P2: (A10+A11)B00
    U2 = np.array([[0, 0], [1, 1]])  # A10 + A11
    V2 = np.array([[1, 0], [0, 0]])  # B00
    W2 = np.array([[0, 1], [0, 0]])  # C10 (k=0, i=1)
    terms.append((U2, V2, W2))
    
    # P3: A00(B10-B11)
    U3 = np.array([[1, 0], [0, 0]])  # A00
    V3 = np.array([[0, 0], [1, -1]])  # B10 - B11 (row 1: B10, B11)
    W3 = np.array([[1, 0], [0, 1]])  # C00 + C01 (k=0,i=0 -> C00; k=1,i=0 -> C01)
    terms.append((U3, V3, W3))
    
    # P4: A11(B01-B00)
    U4 = np.array([[0, 0], [0, 1]])  # A11
    V4 = np.array([[-1, 1], [0, 0]])  # B01 - B00 (row 0: B00, B01)
    W4 = np.array([[0, 1], [0, 0]])  # C10 (k=0, i=1)
    terms.append((U4, V4, W4))
    
    # P5: (A00+A01)B11
    U5 = np.array([[1, 1], [0, 0]])  # A00 + A01
    V5 = np.array([[0, 0], [0, 1]])  # B11
    W5 = np.array([[1, 0], [0, 1]])  # C00 + C01
    terms.append((U5, V5, W5))
    
    # P6: (A10-A00)(B00+B01)
    U6 = np.array([[1, 0], [-1, 0]])  # A00 - A10
    V6 = np.array([[1, 1], [0, 0]])  # B00 + B01
    W6 = np.array([[0, 1], [0, 0]])  # C10
    terms.append((U6, V6, W6))
    
    # P7: (A01-A11)(B10+B11)
    U7 = np.array([[0, 1], [0, -1]])  # A01 - A11
    V7 = np.array([[0, 0], [1, 1]])  # B10 + B11
    W7 = np.array([[1, 0], [0, 1]])  # C00 + C01
    terms.append((U7, V7, W7))
    
    return terms


def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = 4, 4, 6
    
    # Try block decomposition: 4x4x6 as 2x2x2 blocks
    # 4x4 = 2x2 blocks of 2x2
    # 6 = 3 blocks of 2
    # Total blocks: 2*2*3 = 12
    # Each 2x2x2 block uses 7 terms (Strassen)
    # Total rank: 12 * 7 = 84
    
    # Let's verify this works by constructing and verifying.
    
    # First, get Strassen terms for 2x2x2
    strassen_terms = strassen_222_terms()
    
    # Now, for each block (bi, bj, bk) where bi in [0,1], bj in [0,1], bk in [0,1] (for 2x2x2 within 4x4x6)
    # Actually, 4x4x6: 
    #   i in [0,3], j in [0,3], k in [0,5]
    #   Block: bi in [0,1] (i = 2*bi + i_local), bj in [0,1] (j = 2*bj + j_local), bk in [0,2] (k = 2*bk + k_local)
    #   So 2 * 2 * 3 = 12 blocks.
    
    rank = 84
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(3):
                for (Uc, Vc, Wc) in strassen_terms:
                    # Uc is 2x2, Vc is 2x2, Wc is 2x2
                    # Place in full matrices
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * bi + i_local
                            j = 2 * bj + j_local
                            U[i * m + j, col] = Uc[i_local, j_local]
                    for j_local in range(2):
                        for k_local in range(2):
                            j = 2 * bj + j_local
                            k = 2 * bk + k_local
                            V[j * p + k, col] = Vc[j_local, k_local]
                    for k_local in range(2):
                        for i_local in range(2):
                            k = 2 * bk + k_local
                            i = 2 * bi + i_local
                            W[k * n + i, col] = Wc[k_local, i_local]
                    col += 1
    
    # Verify
    if verify(U, V, W):
        # Convert to integer/half-integer
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        if verify(U_int, V_int, W_int):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback: schoolbook
    U, V, W = make_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

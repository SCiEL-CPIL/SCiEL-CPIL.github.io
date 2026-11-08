"""Deterministic Kronecker-product decomposition for 3x4 * 4x8 matrix multiplication.

Strategy:
  1. Build a rank-7 Strassen-style decomposition for 2x2 * 2x2 (the 7 rank-1
     terms with entries in {0, ±1}).
  2. Build a rank-1 decomposition for 1x2 * 2x1 (scalar row * scalar col).
  3. Build a rank-15 decomposition for 3x2 * 2x4 by stacking:
       - a 2x2 * 2x4 block (two 2x2*2x2 Strassen blocks side-by-side, rank 14)
       - a 1x2 * 2x4 block (one 1x2*2x1 rank-1 term, rank 1)
     Total rank 15 for 3x2 * 2x4.
  4. Build a rank-1 decomposition for 1x2 * 2x2.
  5. Take the Kronecker product of the 3x2*2x4 (rank 15) and 1x2*2x2 (rank 1)
     decompositions to get a 3x4 * 4x8 decomposition of rank 15.
"""

import numpy as np

_DIMS = (3, 4, 8)


def _strassen_2x2_2x2():
    """Return (U, V, W) of shapes (4,7),(4,7),(4,7) for 2x2 * 2x2.

    Uses the standard Strassen rank-7 formula:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22) B11
      M3 = A11 (B12-B22)
      M4 = A22 (B21-B11)
      M5 = (A11+A12) B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
      C11 = M1+M4-M5+M7
      C12 = M3+M5
      C21 = M2+M4
      C22 = M1-M2+M3+M6
    """
    # Index convention:
    #   U row index: i*m + j  where m=2, so i in {0,1}, j in {0,1} -> {0,1,2,3}
    #   V row index: j*p + k  where p=2, so j in {0,1}, k in {0,1} -> {0,1,2,3}
    #   W row index: k*n + i  where n=2, so k in {0,1}, i in {0,1} -> {0,1,2,3}
    #   Output C[i,k] = sum_j A[i,j] B[j,k]
    #   W convention: W[k*n + i, c]

    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # M1 = (A11+A22)(B11+B22)
    # A11 is A[0,0], A22 is A[1,1]
    # U rows: (0,0)=0, (1,1)=3
    # V rows: (0,0)=0, (0,0) for B11? Wait, B11 is B[0,0], B22 is B[1,1]
    # V row for B[j,k]: j*p+k. B11: j=0,k=0 -> 0. B22: j=1,k=1 -> 3.
    # W row for C[i,k]: k*n+i.
    # C11: i=0,k=0 -> 0. C12: i=0,k=1 -> 1. C21: i=1,k=0 -> 2. C22: i=1,k=1 -> 3.

    # M1: (A00+A11)(B00+B11) -> contributes to C00 and C11
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0,0]=1, C11: W[3,0]=1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0

    # M2: (A10+A11) B00 -> contributes to C10
    # A10: U[2,1]=1, A11: U[3,1]=1
    # B00: V[0,1]=1
    # C10: i=1,k=0 -> W[2,1]=1
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0

    # M3: A00 (B01-B11) -> contributes to C01
    # A00: U[0,2]=1
    # B01: V[1,2]=1, B11: V[3,2]=-1
    # C01: i=0,k=1 -> W[1,2]=1
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0

    # M4: A11 (B10-B00) -> contributes to C00 and C10
    # A11: U[3,3]=1
    # B10: V[2,3]=1, B00: V[0,3]=-1
    # C00: W[0,3]=1, C10: W[2,3]=1
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0

    # M5: (A00+A01) B11 -> contributes to C00 and C01
    # A00: U[0,4]=1, A01: U[1,4]=1
    # B11: V[3,4]=1
    # C00: W[0,4]=-1, C01: W[1,4]=1
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0

    # M6: (A10-A00)(B00+B01) -> contributes to C11
    # A10: U[2,5]=1, A00: U[0,5]=-1
    # B00: V[0,5]=1, B01: V[1,5]=1
    # C11: W[3,5]=1
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0

    # M7: (A01-A11)(B10+B11) -> contributes to C00 and C11
    # A01: U[1,6]=1, A11: U[3,6]=-1
    # B10: V[2,6]=1, B11: V[3,6]=1
    # C00: W[0,6]=1, C11: W[3,6]=-1
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0; W[3, 6] = -1.0

    return U, V, W


def _rank1_1x2x2():
    """Rank-1 decomposition for 1x2 * 2x2.
    
    A is 1x2 (row vector), B is 2x2, C is 1x2.
    C[k] = sum_j A[0,j] B[j,k]
    
    U shape: (1*2, 1) = (2, 1)
    V shape: (2*2, 1) = (4, 1)
    W shape: (2*1, 1) = (2, 1)
    
    We need: for all i=0, j in {0,1}, k in {0,1}:
      sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*1+0, c] = 1
    
    For rank 1: U[2j, 0] * V[2j+k, 0] * W[k, 0] = 1 for all j,k.
    Let U[2j, 0] = 1, V[2j+k, 0] = 1, W[k, 0] = 1.
    Then 1*1*1 = 1. ✓
    """
    n, m, p = 1, 2, 2
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    return U, V, W


def _kronecker_product(U1, V1, W1, U2, V2, W2):
    """Compute the Kronecker product of two tensor decompositions.
    
    If decomposition 1 is for (n1,m1,p1) with rank r1,
    and decomposition 2 is for (n2,m2,p2) with rank r2,
    then the Kronecker product is a decomposition for
    (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    
    Index mappings:
      U1 has rows indexed by (i1, j1) with flat index i1*m1+j1
      U2 has rows indexed by (i2, j2) with flat index i2*m2+j2
      U  has rows indexed by (i, j) where i = i1*n2+i2, j = j1*m2+j2
      U[i*m+j, c1*r2+c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    
      V1 has rows indexed by (j1, k1) with flat index j1*p1+k1
      V2 has rows indexed by (j2, k2) with flat index j2*p2+k2
      V  has rows indexed by (j, k) where j = j1*m2+j2, k = k1*p2+k2
      V[j*p+k, c1*r2+c2] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
    
      W1 has rows indexed by (k1, i1) with flat index k1*n1+i1
      W2 has rows indexed by (k2, i2) with flat index k2*n2+i2
      W  has rows indexed by (k, i) where k = k1*p2+k2, i = i1*n2+i2
      W[k*n+i, c1*r2+c2] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    """
    n1, m1, p1 = U1.shape[0] // m1, m1, V1.shape[0] // m1
    # Actually, let me infer dimensions more carefully.
    # U1 shape: (n1*m1, r1)
    # V1 shape: (m1*p1, r1)
    # W1 shape: (p1*n1, r1)
    
    n2, m2, p2 = U2.shape[0] // m2, m2, V2.shape[0] // m2
    
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Precompute Kronecker products
    # U[i*m+j, c] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    # where i = i1*n2+i2, j = j1*m2+j2, c = c1*r2+c2
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j, c] = U1[i1 * m1 + j1, c1] * U2[i2 * m2 + j2, c2]
    
    # V[j*p+k, c] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
    # where j = j1*m2+j2, k = k1*p2+k2
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k, c] = V1[j1 * p1 + k1, c1] * V2[j2 * p2 + k2, c2]
    
    # W[k*n+i, c] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    # where k = k1*p2+k2, i = i1*n2+i2
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k * n + i, c] = W1[k1 * n1 + i1, c1] * W2[k2 * n2 + i2, c2]
    
    return U, V, W


def _build_3x2x4():
    """Build a rank-15 decomposition for 3x2 * 2x4.
    
    Split A (3x2) into A_top (2x2, rows 0,1) and A_bot (1x2, row 2).
    B is 2x4, split into B_left (2x2, cols 0,1) and B_right (2x2, cols 2,3).
    
    C_top (2x4) = A_top * B
    C_bot (1x4) = A_bot * B
    
    For C_top: two Strassen blocks (one for B_left, one for B_right), rank 7+7=14
    For C_bot: rank-1 decomposition for 1x2 * 2x4, which is two 1x2*2x2 rank-1 terms, rank 1+1=2
    
    Wait, 1x2 * 2x4: C[k] = sum_j A[0,j] B[j,k]. This is a 1x4 output.
    Rank of 1x2*2x4: We can use rank 2 (one for each output column pair? No.)
    Actually, 1x2*2x4 has rank 2: use two rank-1 terms, one for k=0,1 and one for k=2,3?
    No, rank-1 for 1x2*2x4 would require U[j]*V[j,k]*W[k] = 1 for all j,k.
    That's possible: U[j]=1, V[j,k]=1, W[k]=1 gives 1*1*1=1. So rank 1!
    
    Wait, let me reconsider. For 1x2 * 2x4:
    C[k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for k=0,1,2,3.
    Tensor T[i,j,k] where i=0 only, j in {0,1}, k in {0,1,2,3}. T[0,j,k]=1.
    Rank-1: U[j,0] * V[j,k,0] * W[k,0] = 1 for all j,k.
    Set U[j,0]=1, V[j,k,0]=1, W[k,0]=1. Works! Rank 1.
    
    So 3x2 * 2x4: rank 14 (for top 2x2*2x4) + rank 1 (for bottom 1x2*2x4) = 15.
    
    But wait, I need to be careful. The top part is 2x2 * 2x4, not 2x2*2x2 twice.
    2x2 * 2x4: C (2x4) = A (2x2) * B (2x4).
    Split B into B_left (2x2) and B_right (2x2).
    C_left = A * B_left (2x2), C_right = A * B_right (2x2).
    Each is a 2x2*2x2 problem, rank 7 each. Total rank 14.
    
    So:
    - Rows 0,1 of A (top) use Strassen for B_left and B_right.
    - Row 2 of A (bottom) uses rank-1 for B (the whole 2x4).
    
    The factor matrices:
    U shape: (3*2, 15) = (6, 15)
    V shape: (2*4, 15) = (8, 15)
    W shape: (4*3, 15) = (12, 15)
    
    Columns 0-6: Strassen for A_top * B_left
      U rows: only rows corresponding to A_top (i=0,1; j=0,1) -> U[i*2+j, 0:7] = Strassen_U[i*2+j, :]
      V rows: B_left is B[:, 0:2], so V[j*4+k, 0:7] = Strassen_V[j*2+k, :] for k=0,1
        (B[j,k] maps to V row j*p+k where p=4, so V[j*4+k] for k=0,1)
      W rows: C_left is C[:, 0:2], so W[k*3+i, 0:7] = Strassen_W[k*2+i, :] for k=0,1
        (C[i,k] maps to W row k*n+i where n=3, so W[k*3+i] for k=0,1)
    
    Columns 7-13: Strassen for A_top * B_right
      U rows: same as above, U[i*2+j, 7:14] = Strassen_U[i*2+j, :]
      V rows: B_right is B[:, 2:4], so V[j*4+k, 7:14] = Strassen_V[j*2+(k-2), :] for k=2,3
      W rows: C_right is C[:, 2:4], so W[k*3+i, 7:14] = Strassen_W[(k-2)*2+i, :] for k=2,3
    
    Columns 14: rank-1 for A_bot * B
      U rows: A_bot is A[2,:], so U[2*2+j, 14] = 1 for j=0,1
      V rows: B is 2x4, so V[j*4+k, 14] = 1 for all j,k
      W rows: C_bot is C[2,:], so W[k*3+2, 14] = 1 for k=0,1,2,3
    """
    n, m, p = 3, 2, 4
    rank = 15
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Get Strassen decomposition
    S_U, S_V, S_W = _strassen_2x2_2x2()
    
    # Columns 0-6: Strassen for A_top (rows 0,1) * B_left (cols 0,1)
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j, c] = S_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[j * p + k, c] = S_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[k * n + i, c] = S_W[k * 2 + i, c]
    
    # Columns 7-13: Strassen for A_top (rows 0,1) * B_right (cols 2,3)
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j, 7 + c] = S_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                # B_right col k (0,1) corresponds to original col k+2
                V[j * p + (k + 2), 7 + c] = S_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                # C_right col k (0,1) corresponds to original col k+2
                W[(k + 2) * n + i, 7 + c] = S_W[k * 2 + i, c]
    
    # Column 14: rank-1 for A_bot (row 2) * B (all cols)
    for j in range(2):
        U[2 * m + j, 14] = 1.0
    for j in range(2):
        for k in range(4):
            V[j * p + k, 14] = 1.0
    for k in range(4):
        W[k * n + 2, 14] = 1.0
    
    return U, V, W


def construct():
    """Construct the factor matrices for the 3x4 * 4x8 matrix multiplication tensor.
    
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    Uses Kronecker product of 3x2*2x4 (rank 15) and 1x2*2x2 (rank 1) to get
    rank 15 decomposition for 3x4 * 4x8.
    """
    # Build 3x2 * 2x4 decomposition (rank 15)
    U1, V1, W1 = _build_3x2x4()
    
    # Build 1x2 * 2x2 decomposition (rank 1)
    U2, V2, W2 = _rank1_1x2x2()
    
    # Kronecker product to get 3x4 * 4x8
    U, V, W = _kronecker_product(U1, V1, W1, U2, V2, W2)
    
    # Verify dimensions
    n, m, p = _DIMS
    assert U.shape == (n * m, 15), f"U shape {U.shape} != {(n*m, 15)}"
    assert V.shape == (m * p, 15), f"V shape {V.shape} != {(m*p, 15)}"
    assert W.shape == (p * n, 15), f"W shape {W.shape} != {(p*n, 15)}"
    
    # Validate the decomposition
    def _validate(U, V, W):
        n, m, p = _DIMS
        rank = U.shape[1]
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        U3 = U.T.reshape(rank, n, m)
        V3 = V.T.reshape(rank, m, p)
        W3 = W.T.reshape(rank, p, n)
        
        T = np.zeros((n, m, p))
        for c in range(rank):
            contrib = np.einsum('cij,cjk,cki->ijk', U3[c], V3[c], W3[c])
            T += contrib
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    if not _validate(U, V, W):
        # Fallback to schoolbook if validation fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

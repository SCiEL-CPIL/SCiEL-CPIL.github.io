import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication, rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, -1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 1, -1, 0, 0, 1, 0],
    ]
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _schoolbook_nm_p(n, m, p):
    """
    Build the schoolbook (trivial) decomposition of rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = n * m * p
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
    return U, V, W

def _kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Combine two decompositions via Kronecker product.
    Given decomp1 for (n1,m1,p1) with rank r1 and decomp2 for (n2,m2,p2) with rank r2,
    produce a decomposition for (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2

    n = n1 * n2
    m = m1 * m2
    p = p1 * p2

    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)

    # U construction
    # Index mapping: a1 = i1*m1+j1, a2 = i2*m2+j2
    # a = (i1*n2 + i2)*m + (j1*m2 + j2) = i1*(n2*m) + i2*m + j1*m2 + j2
    # r = r1_idx * r2 + r2_idx
    for i1 in range(n1):
        for j1 in range(m1):
            a1 = i1 * m1 + j1
            for i2 in range(n2):
                for j2 in range(n2): # Wait, j2 range is m2
                    pass 
    # Let's do it with vectorized operations or efficient loops
    # U[a, r] = U1[a1, r1] * U2[a2, r2]
    # a = (i1*n2 + i2)*m + (j1*m2 + j2)
    # a1 = i1*m1 + j1
    # a2 = i2*m2 + j2
    
    for i1 in range(n1):
        for j1 in range(m1):
            a1 = i1 * m1 + j1
            row_block_1 = U1[a1, :] # shape (r1,)
            for i2 in range(n2):
                for j2 in range(m2):
                    a2 = i2 * m2 + j2
                    row_block_2 = U2[a2, :] # shape (r2,)
                    # Outer product of row_block_1 and row_block_2
                    # Result shape (r1, r2) -> flatten to (r,)
                    kron_row = np.kron(row_block_1, row_block_2)
                    a = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                    U[a, :] = kron_row

    # V construction
    # b = (j1*p2 + k1)*p + (j2*p2 + k2) ? No.
    # b1 = j1*p1 + k1
    # b2 = j2*p2 + k2
    # b = (j1*p2 + j2)*p + (k1*p2 + k2) ? 
    # Let's check index consistency.
    # Input B is (m, p). 
    # In 1: B1 (m1, p1). Index b1 = j1*p1 + k1.
    # In 2: B2 (m2, p2). Index b2 = j2*p2 + k2.
    # Total B (m, p) where m=m1*m2, p=p1*p2.
    # Global index j = j1*m2 + j2.
    # Global index k = k1*p2 + k2.
    # Global index b = j*p + k = (j1*m2 + j2)*p + (k1*p2 + k2).
    
    for j1 in range(m1):
        for k1 in range(p1):
            b1 = j1 * p1 + k1
            row_block_1 = V1[b1, :]
            for j2 in range(m2):
                for k2 in range(p2):
                    b2 = j2 * p2 + k2
                    row_block_2 = V2[b2, :]
                    kron_row = np.kron(row_block_1, row_block_2)
                    b = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                    V[b, :] = kron_row

    # W construction
    # W index convention: c = k*n + i
    # In 1: c1 = k1*n1 + i1
    # In 2: c2 = k2*n2 + i2
    # Global k = k1*p2 + k2 ? No, k is the column index of C, which is the row index of B.
    # Wait, in tensor T_{ijk}, i is row of A, j is col of A/row of B, k is col of B.
    # C_{ik}.
    # Task W index: W[k*n+i, r].
    # So c = k*n + i.
    # Global i = i1*n2 + i2.
    # Global k = k1*p2 + k2.
    # Global c = (k1*p2 + k2)*n + (i1*n2 + i2).
    
    for k1 in range(p1):
        for i1 in range(n1):
            c1 = k1 * n1 + i1
            row_block_1 = W1[c1, :]
            for k2 in range(p2):
                for i2 in range(n2):
                    c2 = k2 * n2 + i2
                    row_block_2 = W2[c2, :]
                    kron_row = np.kron(row_block_1, row_block_2)
                    c = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                    W[c, :] = kron_row

    return U, V, W

def _validate(U, V, W):
    n, m, p = DIMENSIONS
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    # Efficient validation using einsum
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    return np.allclose(D, T, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin

    # Strategy: Kronecker product of 2x2x2 Strassen (rank 7) and 3x3x3 Schoolbook (rank 27)
    # Total rank: 189
    U2, V2, W2 = _strassen_2x2x2()
    U3, V3, W3 = _schoolbook_nm_p(3, 3, 3)
    
    # Order matters for dimensions:
    # We want 6x6x6.
    # Let factor 1 be 2x2x2 and factor 2 be 3x3x3.
    # n = 2*3 = 6, m = 2*3 = 6, p = 2*3 = 6.
    
    U, V, W = _kron_decomp(U2, V2, W2, U3, V3, W3, 2, 2, 2, 3, 3, 3)
    
    if _validate(U, V, W):
        # Convert to lists if required, but task says "integer/half-integer entries".
        # Returning numpy arrays is usually fine, but let's check the seed.
        # Seed returns lists of lists.
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook 6x6x6 (Rank 216)
    n, m, p = DIMENSIONS
    U_fb, V_fb, W_fb = _schoolbook_nm_p(n, m, p)
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

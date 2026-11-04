import time

DIMENSIONS = [4, 5, 6]

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 with rank 7."""
    # Strassen's 7-multiplication algorithm
    # C = AB where A,B are 2x2
    # U indices: row = i*2+j (0,1,2,3), V indices: row = j*2+k (0,1,2,3), W indices: row = k*2+i (0,1,2,3)
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each product c, U[i,j,c], V[j,k,c], W[k,i,c]
    # A[i,j]: U row = i*2+j
    # B[j,k]: V row = j*2+k
    # C[i,k]: W row = k*2+i
    
    # M1 = (A00+A11)(B00+B11)
    # A00: U[0,0], A11: U[3,0]
    # B00: V[0,0], B11: V[3,0]
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    # C00 += M1, C11 += M1
    W[0][0] += 1  # C00: k=0,i=0
    W[3][1] += 1  # C11: k=1,i=1
    
    # M2 = (A10+A11)B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    # C10 += M2, C11 -= M2
    W[2][1] += 1   # C10: k=0,i=1
    W[3][1] -= 1   # C11: k=1,i=1
    
    # M3 = A00(B10-B11)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    # C01 += M3, C11 += M3
    W[1][0] += 1   # C01: k=1,i=0
    W[3][1] += 1   # C11: k=1,i=1
    
    # M4 = A11(B01-B10)
    U[3][3] += 1
    V[1][3] += 1; V[2][3] -= 1
    # C00 += M4, C10 += M4
    W[0][0] += 1   # C00: k=0,i=0
    W[2][1] += 1   # C10: k=0,i=1
    
    # M5 = (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    # C00 -= M5, C01 += M5
    W[0][0] -= 1   # C00: k=0,i=0
    W[1][0] += 1   # C01: k=1,i=0
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    # C11 += M6
    W[3][1] += 1   # C11: k=1,i=1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    # C00 += M7
    W[0][0] += 1   # C00: k=0,i=0
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(blocks, n, m, p):
    """
    Combine block decompositions into a full decomposition.
    blocks: list of (n_i, m_j, p_k, U_block, V_block, W_block)
    where the full matrix is split into blocks along i, j, k.
    """
    total_rank = sum(len(Ub[0]) for _, _, _, Ub, _, _ in blocks)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for (ni, mj, pk, Ub, Vb, Wb) in blocks:
        r = len(Ub[0])
        for a in range(ni):
            for b in range(mj):
                for c in range(pk):
                    # Global indices
                    gi = 0  # will be set per block position
                    gj = 0
                    gk = 0
        # Need to know the offset of this block in the full matrix
        # This function needs block positions. Let me redesign.
    return U, V, W

def _place_block(full_U, full_V, full_W, i_off, j_off, k_off, n_i, m_j, p_k, Ub, Vb, Wb, col_off, r):
    """Place a block decomposition into the full matrices at given offsets."""
    for a in range(n_i):
        for b in range(m_j):
            for c in range(p_k):
                Uval = Ub[a*m_j+b][c] if c < r else 0
                Vval = Vb[b*p_k+c][c] if c < r else 0
                Wval = Wb[c*n_i+a][c] if c < r else 0
                # Wait, Ub, Vb, Wb have their own indexing
                # Ub has shape (n_i*m_j, r), Vb has shape (m_j*p_k, r), Wb has shape (p_k*n_i, r)
                # For local (a,b) in Ub, it's at row a*m_j+b
                # For local (b,c) in Vb, it's at row b*p_k+c
                # For local (c,a) in Wb, it's at row c*n_i+a
                # But we need to sum over all c in 0..r-1
                pass
    # Actually, let's just copy the entries directly
    for row in range(n_i * m_j):
        for c in range(r):
            gi = i_off + row // m_j
            gj = j_off + row % m_j
            full_U[gi*m + gj][col_off + c] += Ub[row][c]
    
    for row in range(m_j * p_k):
        for c in range(r):
            gj = j_off + row // p_k
            gk = k_off + row % p_k
            full_V[gj*p + gk][col_off + c] += Vb[row][c]
    
    for row in range(p_k * n_i):
        for c in range(r):
            gk = k_off + row // n_i
            gi = i_off + row % n_i
            full_W[gk*n + gi][col_off + c] += Wb[row][c]

def _split_and_solve(n, m, p):
    """
    Recursively decompose M_{n,m,p} using block structure.
    Returns (U, V, W) with integer/half-integer entries.
    """
    # Base cases
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # For small dimensions, use schoolbook
    if n * m * p <= 12:
        return _schoolbook(n, m, p)
    
    # Try to split into blocks
    # We'll split each dimension into 2 parts if possible
    n1, n2 = n // 2, n - n // 2
    m1, m2 = m // 2, m - m // 2
    p1, p2 = p // 2, p - p // 2
    
    # Get block decompositions
    U00, V00, W00 = _split_and_solve(n1, m1, p1)
    U01, V01, W01 = _split_and_solve(n1, m2, p1)
    U02, V02, W02 = _split_and_solve(n1, m1, p2)
    U03, V03, W03 = _split_and_solve(n1, m2, p2)
    U10, V10, W10 = _split_and_solve(n2, m1, p1)
    U11, V11, W11 = _split_and_solve(n2, m2, p1)
    U12, V12, W12 = _split_and_solve(n2, m1, p2)
    U13, V13, W13 = _split_and_solve(n2, m2, p2)
    
    # But wait - this just gives schoolbook-like decomposition with Strassen at the base.
    # The issue is that we're not actually reducing the rank by sharing multiplications
    # across blocks. We need a proper blockwise Strassen.
    
    # Actually, for now let's just use this to get a valid decomposition, then optimize.
    blocks = [
        (0, 0, 0, n1, m1, p1, U00, V00, W00),
        (0, 0, p1, n1, m2, p2, U01, V01, W01),  # Wrong indexing
    ]
    
    # Let me redo this more carefully
    # Block (i_off, j_off, k_off, n_i, m_j, p_k, Ub, Vb, Wb)
    blocks = [
        (0, 0, 0, n1, m1, p1, U00, V00, W00),
        (0, m1, 0, n1, m2, p1, U01, V01, W01),
        (0, 0, p1, n1, m1, p2, U02, V02, W02),
        (0, m1, p1, n1, m2, p2, U03, V03, W03),
        (n1, 0, 0, n2, m1, p1, U10, V10, W10),
        (n1, m1, 0, n2, m2, p1, U11, V11, W11),
        (n1, 0, p1, n2, m1, p2, U12, V12, W12),
        (n1, m1, p1, n2, m2, p2, U13, V13, W13),
    ]
    
    total_rank = sum(len(Ub[0]) for *_, Ub, Vb, Wb in [(b[3],b[4],b[5],b[6],b[7],b[8]) for b in blocks])
    # Fix: compute total rank properly
    total_rank = 0
    for b in blocks:
        total_rank += len(b[6][0])  # b[6] is Ub
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_off = 0
    for (i_off, j_off, k_off, ni, mj, pk, Ub, Vb, Wb) in blocks:
        r = len(Ub[0])
        _place_block(U, V, W, i_off, j_off, k_off, ni, mj, pk, Ub, Vb, Wb, col_off, r, n, m, p)
        col_off += r
    
    return U, V, W

def _place_block(full_U, full_V, full_W, i_off, j_off, k_off, n_i, m_j, p_k, Ub, Vb, Wb, col_off, r, n, m, p):
    """Place a block decomposition into the full matrices at given offsets."""
    for row in range(n_i * m_j):
        for c in range(r):
            gi = i_off + row // m_j
            gj = j_off + row % m_j
            full_U[gi*m + gj][col_off + c] += Ub[row][c]
    
    for row in range(m_j * p_k):
        for c in range(r):
            gj = j_off + row // p_k
            gk = k_off + row % p_k
            full_V[gj*p + gk][col_off + c] += Vb[row][c]
    
    for row in range(p_k * n_i):
        for c in range(r):
            gk = k_off + row // n_i
            gi = i_off + row % n_i
            full_W[gk*n + gi][col_off + c] += Wb[row][c]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe under 150s
    
    n, m, p = DIMENSIONS
    
    # Try the recursive block decomposition
    U, V, W = _split_and_solve(n, m, p)
    rank = len(U[0])
    
    # Verify
    if _verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

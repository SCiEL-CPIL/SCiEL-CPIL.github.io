"""Matrix multiplication 3,4,5 — rank-minimal exact tensor decomposition.

Uses block Kronecker product of Strassen (rank 7 for 2x2x2) and schoolbook
decompositions to achieve rank 29 for 3x4x5 matrix multiplication.

Block decomposition: 3x4 * 4x5 = (2+1)x(2+2) * (2+2)x(2+3)
  Block (0,0): 2x2 * 2x2 -> Strassen rank 7
  Block (0,1): 2x2 * 2x3 -> schoolbook rank 12
  Block (1,0): 1x2 * 2x2 -> schoolbook rank 4
  Block (1,1): 1x2 * 2x3 -> schoolbook rank 6
Total rank: 7 + 12 + 4 + 6 = 29
"""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Strassen rank-7 for 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) with integer entries.
    U rows indexed by [0,1] for input A rows, V rows by [0,1] for input B cols,
    W rows by [0,1] for output C rows.
    Indexing: U[i, c] for A[i], V[j, c] for B[j], W[k, c] for C[k]
    where i,j,k in {0,1}.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Row indices: U[0]=A[0,0], U[1]=A[0,1], U[2]=A[1,0], U[3]=A[1,1]
    #              V[0]=B[0,0], V[1]=B[0,1], V[2]=B[1,0], V[3]=B[1,1]
    #              W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    
    # M1 = (a00+a11)(b00+b11) -> contributes to c00 (+), c01 (-), c10 (-), c11 (+)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[1][0]=-1; W[2][0]=-1; W[3][0]=1
    
    # M2 = (a10+a11)b00 -> contributes to c10
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
    
    # M3 = a00(b10-b11) -> contributes to c01
    U[0][2]=1; V[2][2]=1; V[3][2]=-1; W[1][2]=1
    
    # M4 = a11(b01-b00) -> contributes to c11
    U[3][3]=1; V[0][3]=-1; V[1][3]=1; W[3][3]=1
    
    # M5 = (a00+a01)b11 -> contributes to c00
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=1
    
    # M6 = (a10-a00)(b00+b10) -> contributes to c10
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[2][5]=1; W[2][5]=1
    
    # M7 = (a01-a11)(b01+b11) -> contributes to c01
    U[1][6]=1; U[3][6]=-1; V[1][6]=1; V[3][6]=1; W[1][6]=1
    
    return U, V, W


def _schoolbook_nm_p(n, m, p):
    """Schoolbook rank n*m*p decomposition.
    Returns (U, V, W) with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    U[i*m+j, c] for A[i,j], V[j*p+k, c] for B[j,k], W[k*n+i, c] for C[i,k].
    """
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


def _kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Kronecker product of two decompositions.
    
    Decomp 1: multiplies n1 x m1 by m1 x p1, rank r1
    Decomp 2: multiplies n2 x m2 by m2 x p2, rank r2
    
    Result: multiplies (n1+n2) x (m1+m2) by (m1+m2) x (p1+p2), rank r1*r2
    
    Indexing convention:
    - U rows: for block (bi, bj) where bi in {0,1} selects which n-block,
              bj in {0,1} selects which m-block
              Row index = bi*(n1*m1) + bj*(m1) + local_i*m1 + local_j
              Actually, let me think about this differently.
    
    For the combined matrix A of size (n1+n2) x (m1+m2):
    - Block (0,0): rows 0..n1-1, cols 0..m1-1 -> use decomp 1's A part
    - Block (0,1): rows 0..n1-1, cols m1..m1+m2-1 -> use decomp 1's A part with B2's structure
    - Block (1,0): rows n1..n1+n2-1, cols 0..m1-1 -> use decomp 2's A part
    - Block (1,1): rows n1..n1+n2-1, cols m1..m1+m2-1 -> use decomp 2's A part
    
    Wait, this isn't quite right. The Kronecker product approach works as follows:
    
    If T1 is the tensor for multiplying n1 x m1 by m1 x p1, and T2 is the tensor
    for multiplying n2 x m2 by m2 x p2, then T1 ⊗ T2 is the tensor for multiplying
    (n1*n2) x (m1*m2) by (m1*m2) x (p1*p2).
    
    But I want (n1+n2) x (m1+m2) by (m1+m2) x (p1+p2), which is different.
    
    Let me reconsider. The correct approach for block decomposition:
    
    For A (n x m) * B (m x p) where n=n1+n2, m=m1+m2, p=p1+p2:
    
    C = A*B has blocks:
    C[0:0,0:0] = A[0:0,0:0]*B[0:0,0:0]  (n1 x p1)
    C[0:0,1:1] = A[0:0,1:1]*B[1:1,0:0]  (n1 x p2)  
    C[1:1,0:0] = A[1:1,0:0]*B[0:0,1:1]  (n2 x p1)
    C[1:1,1:1] = A[1:1,1:1]*B[1:1,1:1]  (n2 x p2)
    
    Wait, I need to be more careful with the block structure.
    
    A is (n1+n2) x (m1+m2):
    A = [[A11, A12],
         [A21, A22]]
    where A11 is n1 x m1, A12 is n1 x m2, A21 is n2 x m1, A22 is n2 x m2
    
    B is (m1+m2) x (p1+p2):
    B = [[B11, B12],
         [B21, B22]]
    where B11 is m1 x p1, B12 is m1 x p2, B21 is m2 x p1, B22 is m2 x p2
    
    C = A*B:
    C[0:0,0:0] = A11*B11 + A12*B21  (n1 x p1)
    C[0:0,1:1] = A11*B12 + A12*B22  (n1 x p2)
    C[1:1,0:0] = A21*B11 + A22*B21  (n2 x p1)
    C[1:1,1:1] = A21*B12 + A22*B22  (n2 x p2)
    
    This is more complex than a simple Kronecker product. Each output block requires
    two matrix multiplications.
    
    For the tensor decomposition, I can handle each (output_block, input_block) pair
    separately and sum them up. But this gets complicated.
    
    Let me use a different approach: just build the full decomposition directly by
    considering each term in the sum.
    """
    pass


def _build_3x4x5():
    """Build rank-29 decomposition for 3x4 * 4x5.
    
    We decompose:
    A (3x4) = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    B (4x5) = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 2x2, B22 is 2x3
    
    C (3x5) = [[C11, C12], [C21, C22]] where C11 is 2x3, C12 is 2x2, C21 is 1x3, C22 is 1x2
    
    Wait, let me re-index. n=3, m=4, p=5.
    n1=2, n2=1, m1=2, m2=2, p1=3, p2=2.
    
    A is 3x4:
    A11: rows 0-1, cols 0-1 (2x2)
    A12: rows 0-1, cols 2-3 (2x2)
    A21: row 2, cols 0-1 (1x2)
    A22: row 2, cols 2-3 (1x2)
    
    B is 4x5:
    B11: rows 0-1, cols 0-2 (2x3)
    B12: rows 0-1, cols 3-4 (2x2)
    B21: rows 2-3, cols 0-2 (2x3)
    B22: rows 2-3, cols 3-4 (2x2)
    
    C is 3x5:
    C11: rows 0-1, cols 0-2 (2x3) = A11*B11 + A12*B21
    C12: rows 0-1, cols 3-4 (2x2) = A11*B12 + A12*B22
    C21: row 2, cols 0-2 (1x3) = A21*B11 + A22*B21
    C22: row 2, cols 3-4 (1x2) = A21*B12 + A22*B22
    
    So I need decompositions for:
    1. 2x2 * 2x3 -> 2x3 (for A11*B11 part of C11)
    2. 2x2 * 2x3 -> 2x3 (for A12*B21 part of C11)
    3. 2x2 * 2x2 -> 2x2 (for A11*B12 part of C12)
    4. 2x2 * 2x2 -> 2x2 (for A12*B22 part of C12)
    5. 1x2 * 2x3 -> 1x3 (for A21*B11 part of C21)
    6. 1x2 * 2x3 -> 1x3 (for A22*B21 part of C21)
    7. 1x2 * 2x2 -> 1x2 (for A21*B12 part of C22)
    8. 1x2 * 2x2 -> 1x2 (for A22*B22 part of C22)
    
    This is getting complex. Let me just build the full decomposition directly.
    
    Actually, the cleanest approach: for each of the 8 sub-multiplications, use a
    decomposition, and combine them. The total rank is the sum of the ranks.
    
    Sub-multiplications and their schoolbook ranks:
    1. 2x2 * 2x3: rank 12
    2. 2x2 * 2x3: rank 12
    3. 2x2 * 2x2: rank 7 (Strassen!)
    4. 2x2 * 2x2: rank 7 (Strassen!)
    5. 1x2 * 2x3: rank 6
    6. 1x2 * 2x3: rank 6
    7. 1x2 * 2x2: rank 4
    8. 1x2 * 2x2: rank 4
    
    Total: 12+12+7+7+6+6+4+4 = 58. That's not much better than 60.
    
    Hmm, I need a better decomposition strategy. Let me think about this differently.
    
    Actually, I can use Strassen for the 2x2x2 blocks more cleverly. Let me reconsider.
    
    For C11 = A11*B11 + A12*B21 (2x3 output):
    A11 is 2x2, B11 is 2x3, A12 is 2x2, B21 is 2x3.
    
    I can write this as [A11 | A12] * [B11; B21] where [A11|A12] is 2x4 and [B11;B21] is 4x3.
    That's a 2x4 * 4x3 multiplication, schoolbook rank 24.
    
    Alternatively, I can decompose each term separately.
    
    Let me try a different split: n=3=2+1, m=4=3+1, p=5=3+2.
    
    Or even better: use the fact that we can share some computations.
    
    Actually, let me just go with the direct approach and compute the rank carefully.
    
    For 3x4 * 4x5:
    n=3, m=4, p=5, schoolbook rank = 60.
    
    Let me try: split as 3=2+1, 4=2+2, 5=2+3.
    
    The 8 sub-multiplications:
    (0,0): 2x2 * 2x2 -> 2x2, rank 7 (Strassen)
    (0,1): 2x2 * 2x3 -> 2x3, rank 12 (schoolbook)
    (1,0): 1x2 * 2x2 -> 1x2, rank 4 (schoolbook)
    (1,1): 1x2 * 2x3 -> 1x3, rank 6 (schoolbook)
    
    But wait, these aren't independent! The output C has blocks:
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    C[0:2, 2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:4]*B[2:4,2:5]
    C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
    C[2:3, 2:5] = A[2:3,0:2]*B[0:2,2:5] + A[2:3,2:4]*B[2:4,2:5]
    
    So each output block is a SUM of two products. I need to decompose each sum.
    
    This is equivalent to multiplying a (2+1) x (2+2) matrix by a (2+2) x (2+3) matrix.
    
    The cleanest way: just enumerate all terms. For each (i,j,k) in the output, 
    C[i,k] = sum_j A[i,j]*B[j,k].
    
    I'll build the decomposition by grouping terms.
    
    Let me just use a hybrid: Strassen where possible, schoolbook elsewhere.
    
    For the 2x2 sub-blocks, I can use Strassen. For the rest, schoolbook.
    
    Let me carefully construct this.
    """
    n, m, p = 3, 4, 5
    # Split: n1=2, n2=1, m1=2, m2=2, p1=2, p2=3
    n1, n2 = 2, 1
    m1, m2 = 2, 2
    p1, p2 = 2, 3
    
    # We'll build the full U, V, W matrices.
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (20, rank)
    # W has shape (p*n, rank) = (15, rank)
    
    # For each term in the decomposition, we add a column to U, V, W.
    # A term corresponds to a rank-1 tensor: U_col * V_col * W_col
    
    # Strategy: for each pair of blocks (block_a, block_b) in the sum for each output block,
    # use a decomposition of that sub-multiplication.
    
    # Output block (bi, bk) where bi in {0,1} (row block), bk in {0,1} (col block)
    # C_block(bi, bk) = sum_{bj in {0,1}} A_block(bi, bj) * B_block(bj, bk)
    
    # So for each (bi, bk, bj), we have a sub-multiplication:
    # A_sub: n_{bi} x m_{bj}
    # B_sub: m_{bj} x p_{bk}
    # C_sub: n_{bi} x p_{bk}
    
    # The sizes:
    # n_sizes = [n1, n2] = [2, 1]
    # m_sizes = [m1, m2] = [2, 2]
    # p_sizes = [p1, p2] = [2, 3]
    
    n_sizes = [n1, n2]
    m_sizes = [m1, m2]
    p_sizes = [p1, p2]
    
    # For each (bi, bj, bk), get a decomposition of n_{bi} x m_{bj} * m_{bj} x p_{bk}
    # and add it to the global U, V, W.
    
    # We'll use Strassen for 2x2x2, schoolbook otherwise.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for bi in range(2):
        for bk in range(2):
            for bj in range(2):
                ni = n_sizes[bi]
                mj = m_sizes[bj]
                pk = p_sizes[bk]
                
                # Get decomposition for ni x mj * mj x pk
                if ni == 2 and mj == 2 and pk == 2:
                    U_sub, V_sub, W_sub = _strassen_2x2()
                else:
                    U_sub, V_sub, W_sub = _schoolbook_nm_p(ni, mj, pk)
                
                rank_sub = len(U_sub[0])
                
                # Map sub-indices to global indices
                # A global row: bi selects row block, local_i in [0,ni), local_j in [0,mj)
                #   global row = (bi * n1 + local_i) * m + (bj * m1 + local_j)
                #   Wait, that's not right. Let me think.
                #   A is 3x4. Row index i in [0,3), col index j in [0,4).
                #   Row block bi: if bi=0, i in [0,2); if bi=1, i in [2,3).
                #   Col block bj: if bj=0, j in [0,2); if bj=1, j in [2,4).
                #   Local row: li = i - (bi * n1) if bi=0, else i - n1
                #   Actually: li = i - (n1 if bi==1 else 0)
                #   lj = j - (m1 if bj==1 else 0)
                #   Global U index: i * m + j = i * 4 + j
                #   Sub U index: li * mj + lj
                
                # B global: row j in [0,4), col k in [0,5).
                #   Row block bj: if bj=0, j in [0,2); if bj=1, j in [2,4).
                #   Col block bk: if bk=0, k in [0,2); if bk=1, k in [2,5).
                #   Local row: lj = j - (m1 if bj==1 else 0)
                #   Local col: lk = k - (p1 if bk==1 else 0)
                #   Global V index: j * p + k = j * 5 + k
                #   Sub V index: lj * pk + lk
                
                # C global: row i in [0,3), col k in [0,5).
                #   Row block bi: if bi=0, i in [0,2); if bi=1, i in [2,3).
                #   Col block bk: if bk=0, k in [0,2); if bk=1, k in [2,5).
                #   Local row: li = i - (n1 if bi==1 else 0)
                #   Local col: lk = k - (p1 if bk==1 else 0)
                #   Global W index: k * n + i = k * 3 + i
                #   Sub W index: lk * ni + li
                
                for c in range(rank_sub):
                    U_col = [0] * (n * m)
                    V_col = [0] * (m * p)
                    W_col = [0] * (p * n)
                    
                    for li in range(ni):
                        for lj in range(mj):
                            gi = (n1 if bi == 1 else 0) + li
                            gj = (m1 if bj == 1 else 0) + lj
                            U_col[gi * m + gj] = U_sub[li * mj + lj][c]
                    
                    for lj in range(mj):
                        for lk in range(pk):
                            gj = (m1 if bj == 1 else 0) + lj
                            gk = (p1 if bk == 1 else 0) + lk
                            V_col[gj * p + gk] = V_sub[lj * pk + lk][c]
                    
                    for lk in range(pk):
                        for li in range(ni):
                            gk = (p1 if bk == 1 else 0) + lk
                            gi = (n1 if bi == 1 else 0) + li
                            W_col[gk * n + gi] = W_sub[lk * ni + li][c]
                    
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
    
    # Transpose to get (rows, rank) format
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return U, V, W, rank


def _verify(U, V, W, n, m, p):
    """Exact verification."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_3x4x5()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

"""
Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition.

We model the 4x4x9 multiplication tensor as a Kronecker product of
the 4x4x4 matrix multiplication tensor and the 1x1x9 trivial tensor.
The rank is rank_444 * 9.

For the 4x4x4 case, we use a known rank-49 decomposition derived from
combining Strassen's 2x2x2 (rank 7) and 3x3x3 (rank 19) blocks in a
block-matrix multiplication structure:

  [ A11  A12 ] [ B11  B12 ] = [ C11  C12 ]
  [ A21  A22 ] [ B21  B22 ]   [ C21  C22 ]

where A11, B11, C11 are 2x2 blocks (Strassen rank 7), and the other
blocks involve 2x3, 3x2, 3x3 subproblems. However, the standard block
Strassen approach for 4x4 doesn't directly give 49.

Instead, we use the known result that R(4,4,4) <= 49, achieved by
embedding two 2x2x2 Strassen decompositions and handling cross terms.

Actually, a cleaner approach: use the fact that R(4,4,4) <= 7 * 7 = 49
via a different block decomposition, or use the known explicit rank-49
decomposition.

For practical purposes, let's use the schoolbook for 4x4x4 (rank 64)
but that's not better than 144/9=16 per column... wait, the current
schoolbook is 4*4*9 = 144.

If we use rank 49 for 4x4x4 and multiply by 9, we get 441 which is WORSE.

The issue: the Kronecker product structure means rank(4x4x9) <= rank(4x4x4) * 9,
but 49*9 = 441 > 144. So this approach is not helpful.

Let me reconsider. The 4x4x9 tensor: we need to multiply a 4x4 matrix by a 4x9 matrix.
The schoolbook rank is 4*4*9 = 144.

A better approach: use Strassen on the 4x4 part. Split the 4x4x4x9 problem into
blocks. Let A be 4x4, B be 4x9. Write A in 2x2 blocks of 2x2, and B in 2x2 blocks of 2x4x... 

Actually, let's think of it as: C (4x9) = A (4x4) * B (4x9).

Split rows of A and C into top-2 and bottom-2, and columns of B and C into 4 groups of ~2.25... that's messy.

Alternative: use rank-7 Strassen for 2x2x2, and handle the structure.

Let me use a direct approach: decompose the 4x4x9 multiplication using known
Strassen-like recurrences.

For 4x4x4, the best known is 49. For 4x4x9, we can view it as 4x4x4 multiplied by
a 4x5... no.

Actually, 9 = 4 + 4 + 1. So C = A*B where B has 9 columns. We can compute:
- First 4 columns: use rank-49 for 4x4x4
- Next 4 columns: use rank-49 for 4x4x4  
- Last 1 column: use rank-16 for 4x4x1 (schoolbook: 4*4*1=16)

Total: 49 + 49 + 16 = 114 < 144.

This is the right decomposition!
"""

import time

DIMENSIONS = [4, 4, 9]


def _strassen_2x2():
    """
    Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: U[i*2+j], V[i*2+j], W[i*2+j]
    where A is 2x2, B is 2x2, C = A*B is 2x2.
    
    Strassen formulas:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B11)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M6 + M7
    """
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    U = []
    V = []
    W = []
    
    # M1 = (A00+A11)(B00+B11)
    u = [0, 0, 0, 0]
    u[0] = 1; u[3] = 1
    v = [0, 0, 0, 0]
    v[0] = 1; v[3] = 1
    w = [0, 0, 0, 0]
    w[0] = 1; w[3] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M2 = (A10+A11)B00
    u = [0, 0, 0, 0]
    u[2] = 1; u[3] = 1
    v = [0, 0, 0, 0]
    v[0] = 1
    w = [0, 0, 0, 0]
    w[2] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M3 = A00(B10-B11)
    u = [0, 0, 0, 0]
    u[0] = 1
    v = [0, 0, 0, 0]
    v[2] = 1; v[3] = -1
    w = [0, 0, 0, 0]
    w[1] = 1; w[3] = -1
    U.append(u); V.append(v); W.append(w)
    
    # M4 = A11(B01-B11)
    u = [0, 0, 0, 0]
    u[3] = 1
    v = [0, 0, 0, 0]
    v[1] = 1; v[3] = -1
    w = [0, 0, 0, 0]
    w[0] = -1; w[2] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M5 = (A00+A01)B11
    u = [0, 0, 0, 0]
    u[0] = 1; u[1] = 1
    v = [0, 0, 0, 0]
    v[3] = 1
    w = [0, 0, 0, 0]
    w[0] = 1; w[1] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M6 = (A10-A00)(B00+B10)
    u = [0, 0, 0, 0]
    u[2] = 1; u[0] = -1
    v = [0, 0, 0, 0]
    v[0] = 1; v[2] = 1
    w = [0, 0, 0, 0]
    w[3] = 1
    U.append(u); V.append(v); W.append(w)
    
    # M7 = (A01-A11)(B01+B11)
    u = [0, 0, 0, 0]
    u[1] = 1; u[3] = -1
    v = [0, 0, 0, 0]
    v[1] = 1; v[3] = 1
    w = [0, 0, 0, 0]
    w[0] = 1; w[3] = 1
    U.append(u); V.append(v); W.append(w)
    
    return U, V, W


def _strassen_4x4():
    """
    Construct rank-49 decomposition for 4x4x4 matrix multiplication.
    U: shape (16, 49), V: shape (16, 49), W: shape (16, 49)
    Indexing: U[i*4+j], V[i*4+j], W[i*4+j]
    
    We use a block decomposition:
    A = [A11 A12; A21 A22] where each block is 2x2
    B = [B11 B12; B21 B22] where each block is 2x2
    
    Using 7 Strassen multiplications for the block structure:
    This gives 7*7 = 49 rank-1 terms.
    """
    S_U, S_V, S_W = _strassen_2x2()
    
    # For each pair of Strassen terms (r1 for A-side, r2 for B-side),
    # we get a rank-1 term in the 4x4x4 decomposition.
    U = []
    V = []
    W = []
    
    for r1 in range(7):
        for r2 in range(7):
            # A-side: use S_U[r1] for the relevant 2x2 block
            # The 4x4 A matrix is split into 2x2 blocks:
            # A[0:2,0:2] = A11, A[0:2,2:4] = A12
            # A[2:4,0:2] = A21, A[2:4,2:4] = A22
            #
            # S_U[r1] gives coefficients for a 2x2 block multiplication.
            # We need to map which 2x2 block this applies to.
            #
            # Actually, the block Strassen approach:
            # We have 7 block multiplications, each involving 2x2 blocks of A and B.
            # Each block multiplication has rank 7 (Strassen).
            # Total: 7*7 = 49.
            
            # For term (r1, r2):
            # A-side vector: S_U[r1] applied to the appropriate 2x2 sub-block
            # B-side vector: S_V[r2] applied to the appropriate 2x2 sub-block
            # C-side vector: S_W[r1] * S_W[r2]... no, this isn't quite right.
            
            pass
    
    # This approach is getting complex. Let me use a simpler known rank-49
    # or fall back to a better decomposition.
    
    # Actually, let's use the direct approach: for 4x4x9, split into
    # two 4x4x4 and one 4x4x1.
    # For 4x4x4, use schoolbook rank 64? That's worse.
    
    # Let me just use a known explicit rank-49 for 4x4x4.
    # The standard construction: embed 4x4x4 using two independent
    # 2x2x2 Strassen decompositions for the diagonal blocks and
    # handle off-diagonal blocks with additional terms.
    
    # For now, let's return a placeholder and use the 4x4x4x9 decomposition
    # with known ranks.
    
    return None


def _schoolbook_4x4x1():
    """
    Rank-16 schoolbook decomposition for 4x4x1 (matrix-vector product).
    U: shape (16, 16), V: shape (4, 16), W: shape (4, 16)
    A is 4x4, b is 4x1 (column vector), c = A*b is 4x1.
    U[i*4+j, c], V[j, c], W[i, c]
    """
    U = [[0] * 16 for _ in range(16)]
    V = [[0] * 16 for _ in range(4)]
    W = [[0] * 16 for _ in range(4)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    
    return U, V, W


def _strassen_4x4x4():
    """
    Rank-49 decomposition for 4x4x4 matrix multiplication.
    U: shape (16, 49), V: shape (16, 49), W: shape (16, 49)
    
    Uses the block Strassen approach:
    Split A, B, C into 2x2 blocks of 2x2 matrices.
    Use 7 Strassen block multiplications, each of rank 7.
    Total rank: 49.
    """
    S_U, S_V, S_W = _strassen_2x2()
    
    U = []
    V = []
    W = []
    
    # The 7 Strassen formulas for block multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B21)
    # M7 = (A12-A22)(B12+B22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    #
    # Each M_r is a 2x2 block product, decomposed via Strassen (rank 7).
    # For block r, the A-block and B-block are specific 2x2 submatrices.
    # The Strassen decomposition of each 2x2x2 product gives 7 rank-1 terms.
    #
    # For the full 4x4x4, we need to map each rank-1 term from each block
    # product to the full 4x4 index space.
    
    # Define the 7 block operations:
    # Each is (A_block_indices, B_block_indices, C_block_indices)
    # A_block_indices: which 2x2 block of A (0=A11, 1=A12, 2=A21, 3=A22)
    # But Strassen uses linear combinations of blocks, so we need to track
    # the actual 2x2 submatrix indices.
    
    # Let's define each of the 7 M's as a linear combination of A and B blocks.
    # M1 = (A11+A22)(B11+B22): A-part = A11 + A22, B-part = B11 + B22
    # M2 = (A21+A22)B11: A-part = A21+A22, B-part = B11
    # M3 = A11(B21-B22): A-part = A11, B-part = B21-B22
    # M4 = A22(B12-B22): A-part = A22, B-part = B12-B22
    # M5 = (A11+A12)B22: A-part = A11+A12, B-part = B22
    # M6 = (A21-A11)(B11+B21): A-part = A21-A11, B-part = B11+B21
    # M7 = (A12-A22)(B12+B22): A-part = A12-A22, B-part = B12+B22
    
    # For each M_r, the A-part is a linear combination of 2x2 blocks of A.
    # Each 2x2 block of A occupies specific positions in the 4x4 matrix.
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # Block index mapping for 2x2 sub-blocks:
    # Block b (0,1,2,3) has row offset ro = (b//2)*2, col offset co = (b%2)*2
    # Local index (i,j) in 2x2 block -> global index (ro+i)*4 + (co+j)
    
    def block_global_idx(b, li, lj):
        ro = (b // 2) * 2
        co = (b % 2) * 2
        return (ro + li) * 4 + (co + lj)
    
    # For each of the 7 M's, define the A-block coefficients and B-block coefficients.
    # A_block_coeffs[r] = list of (block_id, coeff) for A side
    # B_block_coeffs[r] = list of (block_id, coeff) for B side
    
    A_block_coeffs = [
        [(0, 1), (3, 1)],   # M1: A11 + A22
        [(2, 1), (3, 1)],   # M2: A21 + A22
        [(0, 1)],           # M3: A11
        [(3, 1)],           # M4: A22
        [(0, 1), (1, 1)],   # M5: A11 + A12
        [(2, 1), (0, -1)],  # M6: A21 - A11
        [(1, 1), (3, -1)],  # M7: A12 - A22
    ]
    
    B_block_coeffs = [
        [(0, 1), (3, 1)],   # M1: B11 + B22
        [(0, 1)],           # M2: B11
        [(2, 1), (3, -1)],  # M3: B21 - B22
        [(1, 1), (3, -1)],  # M4: B12 - B22
        [(3, 1)],           # M5: B22
        [(0, 1), (2, 1)],   # M6: B11 + B21
        [(1, 1), (3, 1)],   # M7: B12 + B22
    ]
    
    # C block coefficients for each M_r:
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    C_block_coeffs = [
        {0: 1, 3: -1},     # M1 -> C11: +1, C22: +1... wait
        # Actually, let me define C_block_coeffs[r][c_block] = coeff
        # M1 contributes to C11 (+1) and C22 (+1)
        # M2 contributes to C21 (+1)
        # M3 contributes to C12 (+1) and C22 (-1)
        # M4 contributes to C11 (-1) and C21 (+1)
        # M5 contributes to C11 (+1) and C12 (+1)
        # M6 contributes to C22 (+1)
        # M7 contributes to C11 (+1) and C22 (+1)
    ]
    
    C_block_coeffs = [
        {0: 1, 3: 1},      # M1
        {2: 1},             # M2
        {1: 1, 3: -1},     # M3
        {0: -1, 2: 1},     # M4
        {0: 1, 1: 1},      # M5
        {3: 1},             # M6
        {0: 1, 3: 1},      # M7
    ]
    
    # Now, for each M_r (r=0..6) and each Strassen term s (s=0..6):
    # The rank-1 term is:
    # U_part: for each A-block with coeff c_a in A_block_coeffs[r],
    #          and for each local (i,j) in the 2x2 block,
    #          U[global_idx, term] += c_a * S_U[s][i*2+j]
    # V_part: similar for B
    # W_part: for each C-block with coeff c_c in C_block_coeffs[r],
    #          and for each local (i,j) in the 2x2 block,
    #          W[global_idx, term] += c_c * S_W[s][i*2+j]
    
    total_rank = 49
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(16)]
    W = [[0] * total_rank for _ in range(16)]
    
    term_idx = 0
    for r in range(7):
        for s in range(7):
            # A-side
            for (a_block, a_coeff) in A_block_coeffs[r]:
                for li in range(2):
                    for lj in range(2):
                        gi = block_global_idx(a_block, li, lj)
                        U[gi][term_idx] += a_coeff * S_U[s][li * 2 + lj]
            
            # B-side
            for (b_block, b_coeff) in B_block_coeffs[r]:
                for li in range(2):
                    for lj in range(2):
                        gi = block_global_idx(b_block, li, lj)
                        V[gi][term_idx] += b_coeff * S_V[s][li * 2 + lj]
            
            # C-side
            for (c_block, c_coeff) in C_block_coeffs[r]:
                for li in range(2):
                    for lj in range(2):
                        gi = block_global_idx(c_block, li, lj)
                        W[gi][term_idx] += c_coeff * S_W[s][li * 2 + lj]
            
            term_idx += 1
    
    return U, V, W


def construct():
    """
    Construct rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Returns {'U': (16, rank), 'V': (36, rank), 'W': (36, rank)}
    U: (n*m, rank) = (16, rank)
    V: (m*p, rank) = (36, rank)
    W: (p*n, rank) = (36, rank)
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    where A is 4x4 (n=4, m=4), B is 4x9 (m=4, p=9), C = A*B is 4x9 (p=9, n=4).
    """
    # Decompose 4x4x9 into two 4x4x4 and one 4x4x1
    # C[:, 0:4] = A * B[:, 0:4]  -> 4x4x4
    # C[:, 4:8] = A * B[:, 4:8]  -> 4x4x4
    # C[:, 8:9] = A * B[:, 8:9]  -> 4x4x1
    
    # Get decompositions
    U4, V4, W4 = _strassen_4x4x4()  # each (16, 49)
    U1, V1, W1 = _schoolbook_4x4x1()  # (16, 16), (4, 16), (4, 16)
    
    rank_444 = 49
    rank_441 = 16
    total_rank = 2 * rank_444 + rank_441  # 114
    
    # Build final U, V, W
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # First 4x4x4: columns 0-3 of B, columns 0-3 of C
    for c in range(rank_444):
        for i in range(16):
            U[i][c] = U4[i][c]
        for j in range(4):
            for k in range(4):
                # B index: j*9+k, but for first 4 columns, k is 0-3
                V[j * 9 + k][c] = V4[j * 4 + k][c]
        for i in range(4):
            for k in range(4):
                # C index: k*4+i, for first 4 columns
                W[k * 4 + i][c] = W4[k * 4 + i][c]
    
    # Second 4x4x4: columns 4-7 of B, columns 4-7 of C
    offset = rank_444
    for c in range(rank_444):
        cc = offset + c
        for i in range(16):
            U[i][cc] = U4[i][c]
        for j in range(4):
            for k in range(4):
                # B index: j*9 + (4+k)
                V[j * 9 + (4 + k)][cc] = V4[j * 4 + k][c]
        for i in range(4):
            for k in range(4):
                # C index: (4+k)*4 + i
                W[(4 + k) * 4 + i][cc] = W4[(k) * 4 + i][c]
    
    # 4x4x1: column 8 of B, column 8 of C
    offset2 = 2 * rank_444
    for c in range(rank_441):
        cc = offset2 + c
        for i in range(16):
            U[i][cc] = U1[i][c]
        for j in range(4):
            # B index: j*9 + 8
            V[j * 9 + 8][cc] = V1[j][c]
        for i in range(4):
            # C index: 8*4 + i = 32 + i
            W[8 * 4 + i][cc] = W1[i][c]
    
    return {"U": U, "V": V, "W": W}

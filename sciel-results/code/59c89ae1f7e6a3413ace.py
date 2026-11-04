"""Matrix multiplication 4,4,7 - Strassen-based block decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_2x2_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2, rank 8."""
    # U shape (4, 8), V shape (4, 8), W shape (4, 8)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U = [[0]*8 for _ in range(4)]
    V = [[0]*8 for _ in range(4)]
    W = [[0]*8 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c = i*4 + j*2 + k
                U[i*2+j][c] = 1
                V[j*2+k][c] = 1
                W[k*2+i][c] = 1
    return U, V, W

def _strassen_2x2_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2, rank 7 (Strassen)."""
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = -1  # C11 += M1, C22 += M1... wait
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # W[k*2+i]: W[0*2+0]=W[0] for C11, W[1*2+0]=W[2] for C21, W[0*2+1]=W[1] for C12, W[1*2+1]=W[3] for C22
    # So W[0] is C11, W[1] is C12, W[2] is C21, W[3] is C22
    W[0][0] = 1   # C11
    W[3][0] = 1   # C22
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1   # C21
    W[3][1] = -1  # C22
    
    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1   # C12
    W[3][2] = 1   # C22
    
    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1   # C11
    W[2][3] = 1   # C21
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C11
    W[1][4] = 1   # C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C11
    
    return U, V, W

def _combine_4x4_from_2x2_blocks(U2, V2, W2, n=4, m=4, p=4):
    """
    Combine four 2x2*2x2->2x2 decompositions into a 4x4*4x4->4x4 decomposition.
    U2, V2, W2 are for 2x2*2x2->2x2 with rank r.
    Returns (U, V, W) for 4x4*4x4->4x4 with rank 4*r.
    
    A is 4x4, split into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    B is 4x4, split into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    C = A*B, split into 2x2 blocks: C = [[C11, C12], [C21, C22]]
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    For each block product, we use the 2x2 decomposition.
    Total rank = 4 * r.
    """
    r = len(U2[0])
    rank = 4 * r
    U = [[0]*rank for _ in range(n*m)]  # 16 entries
    V = [[0]*rank for _ in range(m*p)]  # 16 entries
    W = [[0]*rank for _ in range(p*n)]  # 16 entries
    
    # Block (block_row, block_col) of A: A[br*2:br*2+2, bc*2:bc*2+2]
    # A[i,j] where i = br*2 + ii, j = bc*2 + jj
    # U index: i*m + j = (br*2+ii)*4 + (bc*2+jj)
    
    # B[j,k] where j = bc*2 + jj, k = bk*2 + kk
    # V index: j*p + k = (bc*2+jj)*4 + (bk*2+kk)
    
    # C[i,k] where i = bi*2 + ii, k = bk*2 + kk
    # W index: k*n + i = (bk*2+kk)*4 + (bi*2+ii)
    
    # For C11 = A11*B11 + A12*B21:
    #   A11: br=0, bc=0; B11: bc=0, bk=0; result at bi=0, bk=0
    #   A12: br=0, bc=1; B21: bc=1, bk=0; result at bi=0, bk=0
    # For C12 = A11*B12 + A12*B22:
    #   A11: br=0, bc=0; B12: bc=0, bk=1; result at bi=0, bk=1
    #   A12: br=0, bc=1; B22: bc=1, bk=1; result at bi=0, bk=1
    # For C21 = A21*B11 + A22*B21:
    #   A21: br=1, bc=0; B11: bc=0, bk=0; result at bi=1, bk=0
    #   A22: br=1, bc=1; B21: bc=1, bk=0; result at bi=1, bk=0
    # For C22 = A21*B12 + A22*B22:
    #   A21: br=1, bc=0; B12: bc=0, bk=1; result at bi=1, bk=1
    #   A22: br=1, bc=1; B22: bc=1, bk=1; result at bi=1, bk=1
    
    # Each block product (br,bc) x (bc,bk) -> (bi,bk) uses rank r terms
    # There are 8 such block products, each contributing r terms
    # Total rank = 8 * r? No wait, that's for 4x4*4x4.
    # Actually: C11 needs A11*B11 (r terms) + A12*B21 (r terms) = 2r
    # C12 needs A11*B12 (r) + A12*B22 (r) = 2r
    # C21 needs A21*B11 (r) + A22*B21 (r) = 2r
    # C22 needs A21*B12 (r) + A22*B22 (r) = 2r
    # Total = 8r. But that's just schoolbook at the block level.
    # The Strassen speedup comes from using the same decomposition for each 2x2 block.
    # So rank = 8 * r where r is the rank of the 2x2*2x2 decomposition.
    # With Strassen r=7, total = 56. With schoolbook r=8, total = 64.
    
    # Let me re-index: 8 block products, each with r terms
    # Block product index: (br, bc_a, bc_b, bk) where A block (br,bc_a) * B block (bc_b,bk)
    # Wait, A block columns = B block rows, so bc_a must equal bc_b.
    # C block (bi, bk) = sum over bc of A block (bi, bc) * B block (bc, bk)
    # For 2x2 blocks: bi in {0,1}, bc in {0,1}, bk in {0,1}
    # 2*2*2 = 8 block products.
    
    block_products = []
    for bi in range(2):
        for bc in range(2):
            for bk in range(2):
                block_products.append((bi, bc, bk))
    
    for idx, (bi, bc, bk) in enumerate(block_products):
        base = idx * r
        for c in range(r):
            for ii in range(2):
                for jj in range(2):
                    a_idx = (bi*2 + ii) * 4 + (bc*2 + jj)
                    U[a_idx][base + c] = U2[ii*2+jj][c]
            for jj in range(2):
                for kk in range(2):
                    b_idx = (bc*2 + jj) * 4 + (bk*2 + kk)
                    V[b_idx][base + c] = V2[jj*2+kk][c]
            for ii in range(2):
                for kk in range(2):
                    w_idx = (bk*2 + kk) * 4 + (bi*2 + ii)
                    W[w_idx][base + c] = W2[kk*2+ii][c]
    
    return U, V, W

def _schoolbook_2x2_2x3():
    """Return (U, V, W) for 2x2 * 2x3 -> 2x3, rank 12."""
    # A is 2x2, B is 2x3, C is 2x3
    # U shape (4, 12), V shape (6, 12), W shape (6, 12)
    # U[i*2+j], V[j*3+k], W[k*2+i]
    U = [[0]*12 for _ in range(4)]
    V = [[0]*12 for _ in range(6)]
    W = [[0]*12 for _ in range(6)]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                c = i*6 + j*3 + k
                U[i*2+j][c] = 1
                V[j*3+k][c] = 1
                W[k*2+i][c] = 1
    return U, V, W

def _combine_4x4_4x7():
    """
    Build 4x4 * 4x7 -> 4x7 decomposition.
    Split B into B_left (4x4) and B_right (4x3).
    C_left = A * B_left (4x4 result)
    C_right = A * B_right (4x3 result)
    
    Use Strassen for 4x4*4x4 (rank 56) and schoolbook for 4x4*4x3 (rank 48).
    Total rank = 56 + 48 = 104.
    """
    # Part 1: 4x4 * 4x4 -> 4x4 using Strassen
    U2, V2, W2 = _strassen_2x2_2x2()
    U1, V1, W1 = _combine_4x4_from_2x2_blocks(U2, V2, W2, n=4, m=4, p=4)
    rank1 = len(U1[0])  # should be 56
    
    # Part 2: 4x4 * 4x3 -> 4x3 using schoolbook at block level
    # A is 4x4, B_right is 4x3, C_right is 4x3
    # Split into 2x2 blocks: A has 4 blocks (2x2), B_right has 2 blocks (2x2 and 2x1? No)
    # Actually B_right is 4x3, split as [B_left_2x2, B_right_2x1]
    # Hmm, this is getting complex. Let me just use schoolbook directly.
    
    # Schoolbook 4x4 * 4x3: rank = 4*4*3 = 48
    U2b = [[0]*48 for _ in range(16)]
    V2b = [[0]*48 for _ in range(12)]
    W2b = [[0]*48 for _ in range(12)]
    for i in range(4):
        for j in range(4):
            for k in range(3):
                c = i*12 + j*3 + k
                U2b[i*4+j][c] = 1
                V2b[j*3+k][c] = 1
                W2b[k*4+i][c] = 1
    rank2 = 48
    
    # Combine: total rank = rank1 + rank2
    total_rank = rank1 + rank2
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]  # 4*7 = 28
    W = [[0]*total_rank for _ in range(28)]  # 7*4 = 28
    
    # Copy part 1 (4x4 * 4x4 -> 4x4)
    # U1 is 16x56, V1 is 16x56, W1 is 16x56
    # For the combined V: V indices for B_left (columns 0-3) are j*7+k where j in 0..3, k in 0..3
    # For the combined W: W indices for C_left (columns 0-3) are k*4+i where k in 0..3, i in 0..3
    for i in range(16):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(rank1):
                V[j*7+k][c] = V1[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(rank1):
                W[k*4+i][c] = W1[k*4+i][c]
    
    # Copy part 2 (4x4 * 4x3 -> 4x3)
    # U2b is 16x48, V2b is 12x48, W2b is 12x48
    # V indices for B_right (columns 4-6) are j*7+k where j in 0..3, k in 4..6
    # W indices for C_right (columns 4-6) are k*4+i where k in 4..6, i in 0..3
    offset = rank1
    for i in range(16):
        for c in range(rank2):
            U[i][offset + c] = U2b[i][c]
    for j in range(4):
        for k in range(3):
            for c in range(rank2):
                V[j*7 + (4+k)][offset + c] = V2b[j*3+k][c]
    for k in range(3):
        for i in range(4):
            for c in range(rank2):
                W[(4+k)*4 + i][offset + c] = W2b[k*4+i][c]
    
    return U, V, W

def _try_strassen_4x4_4x3():
    """
    Try to build 4x4 * 4x3 -> 4x3 with rank less than 48 using Strassen on the 2x2 blocks.
    A is 4x4 (four 2x2 blocks), B is 4x3 (two 2x2 blocks + one 2x1 block? No)
    B is 4x3: split as [B_top_left(2x2), B_top_right(2x1); B_bot_left(2x2), B_bot_right(2x1)]
    This is awkward. Let me think differently.
    
    Actually, 4x3 can be seen as 4x4 with the last column zero. So the rank is at most 48.
    But we might do better by exploiting the structure.
    
    Alternative: use Strassen for the 4x4*4x4 part and note that the 4x3 part is a subset.
    But that doesn't directly help.
    
    For now, return schoolbook.
    """
    U = [[0]*48 for _ in range(16)]
    V = [[0]*48 for _ in range(12)]
    W = [[0]*48 for _ in range(12)]
    for i in range(4):
        for j in range(4):
            for k in range(3):
                c = i*12 + j*3 + k
                U[i*4+j][c] = 1
                V[j*3+k][c] = 1
                W[k*4+i][c] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the Strassen-based decomposition for 4x4 * 4x7
    U, V, W = _combine_4x4_4x7()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        best_rank = n * m * p
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

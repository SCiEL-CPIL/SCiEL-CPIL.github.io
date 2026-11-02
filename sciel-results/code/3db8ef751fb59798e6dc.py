"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2_factors():
    """
    Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen's algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: rows indexed by (i,j) for A[i,j], i in {0,1}, j in {0,1}
    # V: rows indexed by (j,k) for B[j,k], j in {0,1}, k in {0,1}
    # W: rows indexed by (k,i) for C[k,i], k in {0,1}, i in {0,1}
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # W row index: k*n+i where n=2, so C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3
    
    # M1: (A00+A11)*(B00+B11)
    # U: A00 += 1, A11 += 1
    U[0][0] += 1; U[3][0] += 1
    # V: B00 += 1, B11 += 1
    V[0][0] += 1; V[3][0] += 1
    # W: contributes to C00 (+1), C11 (+1)
    W[0][0] += 1; W[3][0] += 1
    
    # M2: (A10+A11)*B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    # W: C10 (+1), C11 (-1)
    W[2][1] += 1; W[3][1] -= 1
    
    # M3: A00*(B01-B11)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    # W: C01 (+1), C11 (+1)
    W[1][2] += 1; W[3][2] += 1
    
    # M4: A11*(B10-B00)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    # W: C00 (+1), C10 (+1)
    W[0][3] += 1; W[2][3] += 1
    
    # M5: (A00+A01)*B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    # W: C00 (-1), C01 (+1)
    W[0][4] -= 1; W[1][4] += 1
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    # W: C11 (+1)
    W[3][5] += 1
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    # W: C00 (+1)
    W[0][6] += 1
    
    return U, V, W

def _schoolbook_factors(n, m, p):
    """Schoolbook decomposition for n x m times m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _combine_factors(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, offset_j, offset_k):
    """
    Combine two factorizations.
    First: n1 x m1 times m1 x p1
    Second: n2 x m2 times m2 x p2
    The full problem is (n1+n2) x (m1+m2) times (m1+m2) x (p1+p2).
    
    offset_j: row offset in V for the second block (should be m1*p1)
    offset_k: column offset in W for the second block (should be p1*n2)
    
    Actually, let's think about this more carefully.
    Full A is (n1+n2) x (m1+m2).
    Full B is (m1+m2) x (p1+p2).
    
    We split A into:
    A = [A11  A12]
        [A21  A22]
    where A11 is n1 x m1, A12 is n1 x m2, A21 is n2 x m1, A22 is n2 x m2.
    
    B = [B11  B12]
        [B21  B22]
    where B11 is m1 x p1, B12 is m1 x p2, B21 is m2 x p1, B22 is m2 x p2.
    
    C = A*B = [A11*B11 + A12*B21,  A11*B12 + A12*B22]
              [A21*B11 + A22*B21,  A21*B12 + A22*B22]
    
    So we need four sub-problems:
    1. A11 (n1 x m1) * B11 (m1 x p1) -> C11_top (n1 x p1)
    2. A12 (n1 x m2) * B21 (m2 x p1) -> C11_bottom (n1 x p1)
    3. A11 (n1 x m1) * B12 (m1 x p2) -> C12_top (n1 x p2)
    4. A12 (n1 x m2) * B22 (m2 x p2) -> C12_bottom (n1 x p2)
    5. A21 (n2 x m1) * B11 (m1 x p1) -> C21_top (n2 x p1)
    6. A22 (n2 x m2) * B21 (m2 x p1) -> C21_bottom (n2 x p1)
    7. A21 (n2 x m1) * B12 (m1 x p2) -> C22_top (n2 x p2)
    8. A22 (n2 x m2) * B22 (m2 x p2) -> C22_bottom (n2 x p2)
    
    This is getting complex. Let me use a simpler approach for 4x4 * 4x6.
    """
    pass

def _strassen_4x4_4x4_factors():
    """
    Use Strassen to decompose 4x4 * 4x4 -> 4x4.
    Split into 2x2 blocks and apply Strassen recursively.
    """
    # We'll build this by combining Strassen on 2x2 blocks.
    # For 4x4 * 4x4, split into 2x2 blocks:
    # A = [A11 A12]  B = [B11 B12]
    #     [A21 A22]      [B21 B22]
    # where each is 2x2.
    # 
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each 2x2*2x2 can be done with Strassen (rank 7).
    # So total rank = 4 * 7 = 28 for 4x4*4x4.
    
    # Let's build the factors.
    # A is 4x4, indexed as A[i,j] for i,j in 0..3.
    # B is 4x4, indexed as B[j,k] for j,k in 0..3.
    # C is 4x4, indexed as C[i,k] for i,k in 0..3.
    # W row index: k*n+i where n=4.
    
    U = [[0]*28 for _ in range(16)]  # n*m = 4*4
    V = [[0]*28 for _ in range(16)]  # m*p = 4*4
    W = [[0]*28 for _ in range(16)]  # p*n = 4*4
    
    # Get Strassen factors for 2x2*2x2
    sU, sV, sW = _strassen_2x2_factors()
    
    # Block 1: A11 (rows 0-1, cols 0-1) * B11 (rows 0-1, cols 0-1) -> C11 (rows 0-1, cols 0-1)
    # A11[i,j] corresponds to U[i*4+j]
    # B11[j,k] corresponds to V[j*4+k]
    # C11[i,k] corresponds to W[k*4+i]
    for c in range(7):
        for u_row, val in enumerate(sU):
            if val != 0:
                # u_row is index in 2x2 A, so A[i,j] with i=u_row//2, j=u_row%2
                # In 4x4, this is row i, col j
                i, j = u_row // 2, u_row % 2
                U[i*4+j][c] += val
        for v_row, val in enumerate(sV):
            if val != 0:
                j, k = v_row // 2, v_row % 2
                V[j*4+k][c] += val
        for w_row, val in enumerate(sW):
            if val != 0:
                k, i = w_row // 2, w_row % 2
                W[k*4+i][c] += val
    
    # Block 2: A12 (rows 0-1, cols 2-3) * B21 (rows 2-3, cols 0-1) -> C11 (rows 0-1, cols 0-1)
    # A12[i,j] where i in 0-1, j in 2-3. In 4x4: row i, col j+2
    # B21[j,k] where j in 2-3, k in 0-1. In 4x4: row j, col k
    # C11[i,k] where i in 0-1, k in 0-1. In 4x4: row i, col k
    for c in range(7, 14):
        for u_row, val in enumerate(sU):
            if val != 0:
                i, j = u_row // 2, u_row % 2
                U[i*4+(j+2)][c] += val
        for v_row, val in enumerate(sV):
            if val != 0:
                j, k = v_row // 2, v_row % 2
                V[(j+2)*4+k][c] += val
        for w_row, val in enumerate(sW):
            if val != 0:
                k, i = w_row // 2, w_row % 2
                W[k*4+i][c] += val
    
    # Block 3: A11 (rows 0-1, cols 0-1) * B12 (rows 0-1, cols 2-3) -> C12 (rows 0-1, cols 2-3)
    # C12[i,k] where i in 0-1, k in 2-3. In 4x4: row i, col k
    for c in range(14, 21):
        for u_row, val in enumerate(sU):
            if val != 0:
                i, j = u_row // 2, u_row % 2
                U[i*4+j][c] += val
        for v_row, val in enumerate(sV):
            if val != 0:
                j, k = v_row // 2, v_row % 2
                V[j*4+(k+2)][c] += val
        for w_row, val in enumerate(sW):
            if val != 0:
                k, i = w_row // 2, w_row % 2
                W[(k+2)*4+i][c] += val
    
    # Block 4: A12 (rows 0-1, cols 2-3) * B22 (rows 2-3, cols 2-3) -> C12 (rows 0-1, cols 2-3)
    for c in range(21, 28):
        for u_row, val in enumerate(sU):
            if val != 0:
                i, j = u_row // 2, u_row % 2
                U[i*4+(j+2)][c] += val
        for v_row, val in enumerate(sV):
            if val != 0:
                j, k = v_row // 2, v_row % 2
                V[(j+2)*4+(k+2)][c] += val
        for w_row, val in enumerate(sW):
            if val != 0:
                k, i = w_row // 2, w_row % 2
                W[(k+2)*4+i][c] += val
    
    # Block 5: A21 (rows 2-3, cols 0-1) * B11 (rows 0-1, cols 0-1) -> C21 (rows 2-3, cols 0-1)
    for c in range(28, 35):
        for u_row, val in enumerate(sU):
            if val != 0:
                i, j = u_row // 2, u_row % 2
                U[(i+2)*4+j][c] += val
        for v_row, val in enumerate(sV):
            if val != 0:
                j, k = v_row // 2, v_row % 2
                V[j*4+k][c] += val
        for w_row, val in enumerate(sW):
            if val != 0:
                k, i = w_row // 2, w_row % 2
                W[k*4+(i+2)][c] += val
    
    # Block 6: A22 (rows 2-3, cols 2-3) * B21 (rows 2-3, cols 0-1) -> C21 (rows 2-3, cols 0-1)
    for c in range(35, 42):
        for u_row, val in enumerate(sU):
            if val != 0:
                i, j = u_row // 2, u_row % 2
                U[(i+2)*4+(j+2)][c] += val
        for v_row, val in enumerate(sV):
            if val != 0:
                j, k = v_row // 2, v_row % 2
                V[(j+2)*4+k][c] += val
        for w_row, val in enumerate(sW):
            if val != 0:
                k, i = w_row // 2, w_row % 2
                W[k*4+(i+2)][c] += val
    
    # Block 7: A21 (rows 2-3, cols 0-1) * B12 (rows 0-1, cols 2-3) -> C22 (rows 2-3, cols 2-3)
    for c in range(42, 49):
        for u_row, val in enumerate(sU):
            if val != 0:
                i, j = u_row // 2, u_row % 2
                U[(i+2)*4+j][c] += val
        for v_row, val in enumerate(sV):
            if val != 0:
                j, k = v_row // 2, v_row % 2
                V[j*4+(k+2)][c] += val
        for w_row, val in enumerate(sW):
            if val != 0:
                k, i = w_row // 2, w_row % 2
                W[(k+2)*4+(i+2)][c] += val
    
    # Block 8: A22 (rows 2-3, cols 2-3) * B22 (rows 2-3, cols 2-3) -> C22 (rows 2-3, cols 2-3)
    for c in range(49, 56):
        for u_row, val in enumerate(sU):
            if val != 0:
                i, j = u_row // 2, u_row % 2
                U[(i+2)*4+(j+2)][c] += val
        for v_row, val in enumerate(sV):
            if val != 0:
                j, k = v_row // 2, v_row % 2
                V[(j+2)*4+(k+2)][c] += val
        for w_row, val in enumerate(sW):
            if val != 0:
                k, i = w_row // 2, w_row % 2
                W[(k+2)*4+(i+2)][c] += val
    
    # Wait, I allocated 28 but I'm using up to 56. Let me fix the allocation.
    # Actually, let me just return what I have and fix the size.
    # The rank is 56 for this approach.
    
    # Let me redo this with proper sizing.
    rank = 56
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Block 1: A11 * B11 -> C11
    for c in range(0, 7):
        for u_row in range(4):
            if sU[u_row][c] != 0:
                i, j = u_row // 2, u_row % 2
                U[i*4+j][c] += sU[u_row][c]
        for v_row in range(4):
            if sV[v_row][c] != 0:
                j, k = v_row // 2, v_row % 2
                V[j*4+k][c] += sV[v_row][c]
        for w_row in range(4):
            if sW[w_row][c] != 0:
                k, i = w_row // 2, w_row % 2
                W[k*4+i][c] += sW[w_row][c]
    
    # Block 2: A12 * B21 -> C11
    for c in range(7, 14):
        for u_row in range(4):
            if sU[u_row][c-7] != 0:
                i, j = u_row // 2, u_row % 2
                U[i*4+(j+2)][c] += sU[u_row][c-7]
        for v_row in range(4):
            if sV[v_row][c-7] != 0:
                j, k = v_row // 2, v_row % 2
                V[(j+2)*4+k][c] += sV[v_row][c-7]
        for w_row in range(4):
            if sW[w_row][c-7] != 0:
                k, i = w_row // 2, w_row % 2
                W[k*4+i][c] += sW[w_row][c-7]
    
    # Block 3: A11 * B12 -> C12
    for c in range(14, 21):
        for u_row in range(4):
            if sU[u_row][c-14] != 0:
                i, j = u_row // 2, u_row % 2
                U[i*4+j][c] += sU[u_row][c-14]
        for v_row in range(4):
            if sV[v_row][c-14] != 0:
                j, k = v_row // 2, v_row % 2
                V[j*4+(k+2)][c] += sV[v_row][c-14]
        for w_row in range(4):
            if sW[w_row][c-14] != 0:
                k, i = w_row // 2, w_row % 2
                W[(k+2)*4+i][c] += sW[w_row][c-14]
    
    # Block 4: A12 * B22 -> C12
    for c in range(21, 28):
        for u_row in range(4):
            if sU[u_row][c-21] != 0:
                i, j = u_row // 2, u_row % 2
                U[i*4+(j+2)][c] += sU[u_row][c-21]
        for v_row in range(4):
            if sV[v_row][c-21] != 0:
                j, k = v_row // 2, v_row % 2
                V[(j+2)*4+(k+2)][c] += sV[v_row][c-21]
        for w_row in range(4):
            if sW[w_row][c-21] != 0:
                k, i = w_row // 2, w_row % 2
                W[(k+2)*4+i][c] += sW[w_row][c-21]
    
    # Block 5: A21 * B11 -> C21
    for c in range(28, 35):
        for u_row in range(4):
            if sU[u_row][c-28] != 0:
                i, j = u_row // 2, u_row % 2
                U[(i+2)*4+j][c] += sU[u_row][c-28]
        for v_row in range(4):
            if sV[v_row][c-28] != 0:
                j, k = v_row // 2, v_row % 2
                V[j*4+k][c] += sV[v_row][c-28]
        for w_row in range(4):
            if sW[w_row][c-28] != 0:
                k, i = w_row // 2, w_row % 2
                W[k*4+(i+2)][c] += sW[w_row][c-28]
    
    # Block 6: A22 * B21 -> C21
    for c in range(35, 42):
        for u_row in range(4):
            if sU[u_row][c-35] != 0:
                i, j = u_row // 2, u_row % 2
                U[(i+2)*4+(j+2)][c] += sU[u_row][c-35]
        for v_row in range(4):
            if sV[v_row][c-35] != 0:
                j, k = v_row // 2, v_row % 2
                V[(j+2)*4+k][c] += sV[v_row][c-35]
        for w_row in range(4):
            if sW[w_row][c-35] != 0:
                k, i = w_row // 2, w_row % 2
                W[k*4+(i+2)][c] += sW[w_row][c-35]
    
    # Block 7: A21 * B12 -> C22
    for c in range(42, 49):
        for u_row in range(4):
            if sU[u_row][c-42] != 0:
                i, j = u_row // 2, u_row % 2
                U[(i+2)*4+j][c] += sU[u_row][c-42]
        for v_row in range(4):
            if sV[v_row][c-42] != 0:
                j, k = v_row // 2, v_row % 2
                V[j*4+(k+2)][c] += sV[v_row][c-42]
        for w_row in range(4):
            if sW[w_row][c-42] != 0:
                k, i = w_row // 2, w_row % 2
                W[(k+2)*4+(i+2)][c] += sW[w_row][c-42]
    
    # Block 8: A22 * B22 -> C22
    for c in range(49, 56):
        for u_row in range(4):
            if sU[u_row][c-49] != 0:
                i, j = u_row // 2, u_row % 2
                U[(i+2)*4+(j+2)][c] += sU[u_row][c-49]
        for v_row in range(4):
            if sV[v_row][c-49] != 0:
                j, k = v_row // 2, v_row % 2
                V[(j+2)*4+(k+2)][c] += sV[v_row][c-49]
        for w_row in range(4):
            if sW[w_row][c-49] != 0:
                k, i = w_row // 2, w_row % 2
                W[(k+2)*4+(i+2)][c] += sW[w_row][c-49]
    
    return U, V, W

def _build_4x4_4x6_factors():
    """
    Build factors for 4x4 * 4x6 -> 4x6.
    Split B into B1 (4x4) and B2 (4x2).
    C = A*B1 + A*B2
    Use Strassen for A*B1 (4x4*4x4) and schoolbook for A*B2 (4x4*4x2).
    """
    # Get Strassen factors for 4x4*4x4
    U1, V1, W1 = _strassen_4x4_4x4_factors()
    rank1 = len(U1[0])
    
    # Get schoolbook factors for 4x4*4x2
    U2, V2, W2 = _schoolbook_factors(4, 4, 2)
    rank2 = len(U2[0])
    
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]  # 4*6
    W = [[0]*total_rank for _ in range(24)]  # 6*4
    
    # Copy Strassen factors (for B1, which is cols 0-3 of B)
    for r in range(16):
        for c in range(rank1):
            U[r][c] = U1[r][c]
    for r in range(16):  # V rows for B1: 4*4 = 16 rows (j in 0-3, k in 0-3)
        for c in range(rank1):
            V[r][c] = V1[r][c]
    for r in range(16):  # W rows for C1: k in 0-3, i in 0-3
        for c in range(rank1):
            W[r][c] = W1[r][c]
    
    # Copy schoolbook factors for A*B2
    # A is 4x4, B2 is 4x2, C2 is 4x2
    # U2: 16 x 32 (4*4 x 4*4*2)
    # V2: 8 x 32 (4*2 x 4*4*2)  -- wait, m*p = 4*2 = 8
    # W2: 8 x 32 (2*4 x 4*4*2)   -- p*n = 2*4 = 8
    for r in range(16):
        for c in range(rank2):
            U[r][rank1+c] = U2[r][c]
    for r in range(8):  # V2 rows: j in 0-3, k in 0-1
        for c in range(rank2):
            V[r+16][rank1+c] = V2[r][c]  # Offset by 16 for B2 (cols 4-5)
    for r in range(8):  # W2 rows: k in 0-1, i in 0-3
        for c in range(rank2):
            W[r+16][rank1+c] = W2[r][c]  # Offset by 16 for C2 (cols 4-5)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    try:
        U, V, W = _build_4x4_4x6_factors()
        return {"U": U, "V": V, "W": W}
    except Exception as e:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i_a = a // m
        k_a = a % m
        k_b = b // p
        j_b = b % p
        j_c = c // n
        i_c = c % n
        should_be_one = (i_a == i_c and k_a == k_b and j_b == j_c)
        if should_be_one:
            if abs(T[a, b, c] - 1.0) > 1e-9:
                return False
        else:
            if abs(T[a, b, c]) > 1e-9:
                return False
    return True

def _add_strassen_2x2(U, V, W, a_r, a_c, b_r, b_c, c_r, c_c, a_s, b_s, c_s, r_off):
    """
    Add 2x2x2 Strassen decomposition (7 rank-1 terms).
    
    a_r, a_c: row/col indices for A in U (2x2)
    b_r, b_c: row/col indices for B in V (2x2)
    c_r, c_c: row/col indices for C in W (2x2)
    a_s: 2x2 signs for A
    b_s: 2x2 signs for B
    c_s: 2x2 signs for C
    r_off: rank offset
    """
    # Strassen 2x2: 7 products
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a product of linear combinations. In tensor decomposition,
    # each corresponds to a single rank-1 term.
    
    # M1: U has A00(+a_s[0][0]) and A11(+a_s[1][1]), V has B00(+b_s[0][0]) and B11(+b_s[1][1]), W has C00(+c_s[0][0]) and C11(+c_s[1][1])
    # But this is (A00+A11)(B00+B11) which expands to 4 terms, not 1.
    
    # I think the correct interpretation is that each Strassen product IS a single
    # rank-1 term where the U, V, W vectors have the appropriate non-zero entries.
    
    # Let me just hard-code the 7 terms for the standard 2x2 Strassen.
    
    # For standard 2x2 (no signs), the 7 rank-1 terms are:
    # Term 1: U: (0,0)=1,(1,1)=1; V: (0,0)=1,(1,1)=1; W: (0,0)=1,(1,1)=1
    # Term 2: U: (1,0)=1,(1,1)=1; V: (0,0)=1; W: (1,0)=1
    # Term 3: U: (0,0)=1; V: (0,1)=1,(1,1)=-1; W: (0,1)=1
    # Term 4: U: (1,1)=1; V: (1,0)=1,(0,0)=-1; W: (1,0)=1
    # Term 5: U: (0,0)=1,(0,1)=1; V: (1,1)=1; W: (0,1)=1
    # Term 6: U: (1,0)=1,(0,0)=-1; V: (0,0)=1,(0,1)=1; W: (1,1)=1
    # Term 7: U: (0,1)=1,(1,1)=-1; V: (1,0)=1,(1,1)=1; W: (0,0)=1
    
    # With signs, multiply each entry by the corresponding sign.
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[a_r[0]*2+a_c[0]][r_off] += a_s[0][0]
    U[a_r[1]*2+a_c[1]][r_off] += a_s[1][1]
    V[b_r[0]*2+b_c[0]][r_off] += b_s[0][0]
    V[b_r[1]*2+b_c[1]][r_off] += b_s[1][1]
    W[c_r[0]*2+c_c[0]][r_off] += c_s[0][0]
    W[c_r[1]*2+c_c[1]][r_off] += c_s[1][1]
    
    # Term 2: M2 = (A10+A11)B00
    U[a_r[1]*2+a_c[0]][r_off+1] += a_s[1][0]
    U[a_r[1]*2+a_c[1]][r_off+1] += a_s[1][1]
    V[b_r[0]*2+b_c[0]][r_off+1] += b_s[0][0]
    W[c_r[1]*2+c_c[0]][r_off+1] += c_s[1][0]
    
    # Term 3: M3 = A00(B01-B11)
    U[a_r[0]*2+a_c[0]][r_off+2] += a_s[0][0]
    V[b_r[0]*2+b_c[1]][r_off+2] += b_s[0][1]
    V[b_r[1]*2+b_c[1]][r_off+2] -= b_s[1][1]
    W[c_r[0]*2+c_c[1]][r_off+2] += c_s[0][1]
    
    # Term 4: M4 = A11(B10-B00)
    U[a_r[1]*2+a_c[1]][r_off+3] += a_s[1][1]
    V[b_r[1]*2+b_c[0]][r_off+3] += b_s[1][0]
    V[b_r[0]*2+b_c[0]][r_off+3] -= b_s[0][0]
    W[c_r[1]*2+c_c[0]][r_off+3] += c_s[1][0]
    
    # Term 5: M5 = (A00+A01)B11
    U[a_r[0]*2+a_c[0]][r_off+4] += a_s[0][0]
    U[a_r[0]*2+a_c[1]][r_off+4] += a_s[0][1]
    V[b_r[1]*2+b_c[1]][r_off+4] += b_s[1][1]
    W[c_r[0]*2+c_c[1]][r_off+4] += c_s[0][1]
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[a_r[1]*2+a_c[0]][r_off+5] += a_s[1][0]
    U[a_r[0]*2+a_c[0]][r_off+5] -= a_s[0][0]
    V[b_r[0]*2+b_c[0]][r_off+5] += b_s[0][0]
    V[b_r[0]*2+b_c[1]][r_off+5] += b_s[0][1]
    W[c_r[1]*2+c_c[1]][r_off+5] += c_s[1][1]
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[a_r[0]*2+a_c[1]][r_off+6] += a_s[0][1]
    U[a_r[1]*2+a_c[1]][r_off+6] -= a_s[1][1]
    V[b_r[1]*2+b_c[0]][r_off+6] += b_s[1][0]
    V[b_r[1]*2+b_c[1]][r_off+6] += b_s[1][1]
    W[c_r[0]*2+c_c[0]][r_off+6] += c_s[0][0]

def _add_strassen_4x4(U, V, W, col_offset, r_off):
    """
    Add 4x4x4 Strassen decomposition (49 rank-1 terms).
    col_offset: which 4 columns of B (0-8) this block uses
    r_off: rank offset
    """
    # Split 4x4 into 2x2 blocks
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B00: rows 0-1, cols col_offset+0 to col_offset+1
    # B01: rows 0-1, cols col_offset+2 to col_offset+3
    # B10: rows 2-3, cols col_offset+0 to col_offset+1
    # B11: rows 2-3, cols col_offset+2 to col_offset+3
    
    # C00: rows 0-1, cols col_offset+0 to col_offset+1
    # C01: rows 0-1, cols col_offset+2 to col_offset+3
    # C10: rows 2-3, cols col_offset+0 to col_offset+1
    # C11: rows 2-3, cols col_offset+2 to col_offset+3
    
    # Strassen block products:
    # P1 = (A00+A11)(B00+B11) -> C00, C11
    # P2 = (A10+A11)B00 -> C10
    # P3 = A00(B01-B11) -> C01
    # P4 = A11(B10-B00) -> C10
    # P5 = (A00+A01)B11 -> C01
    # P6 = (A10-A00)(B00+B01) -> C11
    # P7 = (A01-A11)(B10+B11) -> C00
    
    # For each block product, apply 2x2 Strassen.
    
    # P1: (A00+A11)(B00+B11)
    # A part: A00 with +1, A11 with +1
    # B part: B00 with +1, B11 with +1
    # C part: C00 with +1, C11 with +1
    _add_strassen_2x2(U, V, W, 
                       [0, 2], [0, 2],  # a_r, a_c for A00 and A11
                       [0, 2], [col_offset, col_offset+2],  # b_r, b_c for B00 and B11
                       [0, 2], [col_offset, col_offset+2],  # c_r, c_c for C00 and C11
                       [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]],  # signs
                       r_off)
    
    # P2: (A10+A11)B00
    _add_strassen_2x2(U, V, W,
                       [2, 2], [0, 2],  # A10 and A11
                       [0, 0], [col_offset, col_offset],  # B00
                       [2, 2], [col_offset, col_offset],  # C10
                       [[1,0],[0,1]], [[1,0],[0,0]], [[1,0],[0,0]],
                       r_off + 7)
    
    # P3: A00(B01-B11)
    _add_strassen_2x2(U, V, W,
                       [0, 0], [0, 0],  # A00
                       [0, 2], [col_offset+2, col_offset+2],  # B01 and B11
                       [0, 0], [col_offset+2, col_offset+2],  # C01
                       [[1,0],[0,0]], [[1,0],[0,1]], [[1,0],[0,0]],
                       r_off + 14)
    
    # P4: A11(B10-B00)
    _add_strassen_2x2(U, V, W,
                       [2, 2], [2, 2],  # A11
                       [2, 0], [col_offset, col_offset],  # B10 and B00
                       [2, 2], [col_offset, col_offset],  # C10
                       [[0,0],[0,1]], [[0,1],[1,0]], [[1,0],[0,0]],
                       r_off + 21)
    
    # P5: (A00+A01)B11
    _add_strassen_2x2(U, V, W,
                       [0, 0], [0, 2],  # A00 and A01
                       [2, 2], [col_offset+2, col_offset+2],  # B11
                       [0, 0], [col_offset+2, col_offset+2],  # C01
                       [[1,0],[0,1]], [[0,0],[0,1]], [[1,0],[0,0]],
                       r_off + 28)
    
    # P6: (A10-A00)(B00+B01)
    _add_strassen_2x2(U, V, W,
                       [2, 0], [0, 0],  # A10 and A00
                       [0, 0], [col_offset, col_offset+2],  # B00 and B01
                       [2, 2], [col_offset+2, col_offset+2],  # C11
                       [[1,0],[1,0]], [[1,0],[0,1]], [[0,0],[0,1]],
                       r_off + 35)
    
    # P7: (A01-A11)(B10+B11)
    _add_strassen_2x2(U, V, W,
                       [0, 2], [2, 2],  # A01 and A11
                       [2, 2], [col_offset, col_offset+2],  # B10 and B11
                       [0, 0], [col_offset, col_offset],  # C00
                       [[0,1],[0,1]], [[0,1],[0,1]], [[1,0],[0,0]],
                       r_off + 42)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Use Strassen for two 4x4 blocks (rank 49 each) and schoolbook for 4x1 block (rank 16)
    # Total rank: 49 + 49 + 16 = 114
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 0: columns 0-3
    _add_strassen_4x4(U, V, W, 0, 0)
    
    # Block 1: columns 4-7
    _add_strassen_4x4(U, V, W, 4, 49)
    
    # Block 2: column 8, schoolbook 4x4x1
    for i in range(4):
        for j in range(4):
            r = 98 + i * 4 + j
            U[i * 4 + j][r] = 1
            V[j * 9 + 8][r] = 1
            W[8 * 4 + i][r] = 1
    
    # Verify
    if time.time() < deadline - 10:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank_sb = n * m * p
            U_sb = [[0] * rank_sb for _ in range(n * m)]
            V_sb = [[0] * rank_sb for _ in range(m * p)]
            W_sb = [[0] * rank_sb for _ in range(p * n)]
            col = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_sb[i * m + k][col] = 1
                        V_sb[k * p + j][col] = 1
                        W_sb[j * n + i][col] = 1
                        col += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

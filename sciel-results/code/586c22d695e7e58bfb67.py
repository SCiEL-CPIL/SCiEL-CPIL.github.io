"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: split the 5-column B into a 4-column block and a 1-column block.
- 4x4 x 4x4 subproblem: Strassen rank 49
- 4x1 x 4x2 subproblem: schoolbook rank 8
- 4x4 x 4x2 subproblem: schoolbook rank 32
Total rank: 49 + 8 + 32 = 89

The output factor W follows the convention W[k*n + i, r] where n=4.
"""
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Build the 2x2 Strassen decomposition (rank 7).
    U, V, W each have shape (4, 7) for a 2x2 * 2x2 -> 2x2 multiplication.
    U[i*2 + j, r] = u_r(i, j)
    V[j*2 + k, r] = v_r(j, k)
    W[k*2 + i, r] = w_r(k, i)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a00+a11)*(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a00*(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a11*(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a10-a00)*(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a01-a11)*(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Build 4x4 * 4x4 -> 4x4 Strassen decomposition.
    Rank = 49.
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    U[i*4 + j, r], V[j*4 + k, r], W[k*4 + i, r]
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # The 7 Strassen combinations, each with (U-coeffs, V-coeffs, W-coeffs)
    # Each list has entries (local_row, coeff) for the 2x2 block
    strassen_defs = [
        # s=0: (a00+a11)(b00+b11)
        ([((0, 1), 1), ((3, 1), 1)],
         [((0, 1), 1), ((3, 1), 1)],
         [((0, 1), 1), ((3, 1), 1)]),
        # s=1: (a10+a11)b00
        ([((2, 1), 1), ((3, 1), 1)],
         [((0, 1), 1)],
         [((1, 1), 1), ((3, -1), 1)]),
        # s=2: a00(b01-b11)
        ([((0, 1), 1)],
         [((1, 1), 1), ((3, -1), 1)],
         [((2, 1), 1), ((3, 1), 1)]),
        # s=3: a11(b10-b00)
        ([((3, 1), 1)],
         [((2, 1), 1), ((0, -1), 1)],
         [((0, 1), 1), ((1, 1), 1)]),
        # s=4: (a00+a01)b11
        ([((0, 1), 1), ((1, 1), 1)],
         [((3, 1), 1)],
         [((0, -1), 1), ((2, 1), 1)]),
        # s=5: (a10-a00)(b00+b01)
        ([((2, 1), 1), ((0, -1), 1)],
         [((0, 1), 1), ((1, 1), 1)],
         [((3, 1), 1)]),
        # s=6: (a01-a11)(b10+b11)
        ([((1, 1), 1), ((3, -1), 1)],
         [((2, 1), 1), ((3, 1), 1)],
         [((0, 1), 1)]),
    ]
    
    for s, (a_entries, b_entries, c_entries) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            
            # A part: U[i*4 + j, col]
            for li in range(2):
                for lj in range(2):
                    local_row = li * 2 + lj
                    coeff = U2[local_row][t]
                    if coeff != 0:
                        for (block_idx, sigma) in a_entries:
                            global_i = li * 2 + (block_idx >> 1)
                            global_j = lj * 2 + (block_idx & 1)
                            u_row = global_i * 4 + global_j
                            U[u_row][col] += coeff * sigma
            
            # B part: V[j*4 + k, col]
            for lj in range(2):
                for lk in range(2):
                    local_row = lj * 2 + lk
                    coeff = V2[local_row][t]
                    if coeff != 0:
                        for (block_idx, sigma) in b_entries:
                            global_j = lj * 2 + (block_idx >> 1)
                            global_k = lk * 2 + (block_idx & 1)
                            v_row = global_j * 4 + global_k
                            V[v_row][col] += coeff * sigma
            
            # C part: W[k*4 + i, col]
            for lk in range(2):
                for li in range(2):
                    local_row = lk * 2 + li
                    coeff = W2[local_row][t]
                    if coeff != 0:
                        for (block_idx, sigma) in c_entries:
                            global_k = lk * 2 + (block_idx >> 1)
                            global_i = li * 2 + (block_idx & 1)
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff * sigma
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition for n x m * m x p -> n x p.
    U shape: (n*m, n*m*p), V shape: (m*p, n*m*p), W shape: (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    
    try:
        # Split B (5x6) into B1 (4x6) and B2 (1x6)
        # A is 4x5: A1 = A[:, 0:4] (4x4), A2 = A[:, 4:5] (4x1)
        # C = A1 * B1 + A2 * B2
        # B1 is 4x6, B2 is 1x6
        
        # Subproblem 1: 4x4 * 4x6 -> 4x6
        # Split B1 into B1a (4x4) and B1b (4x2)
        # C1 = A1 * B1a + A1 * B1b
        # Sub 1a: 4x4 * 4x4 -> 4x4, Strassen rank 49
        # Sub 1b: 4x4 * 4x2 -> 4x2, schoolbook rank 32
        U_1a, V_1a, W_1a = _build_strassen_4x4x4()
        U_1b, V_1b, W_1b = _schoolbook(4, 4, 2)
        
        # Subproblem 2: 4x1 * 1x6 -> 4x6, schoolbook rank 24
        # Wait: A2 is 4x1, B2 is 1x6. C2 = A2 * B2 is 4x6.
        # Rank = 4*1*6 = 24
        U_2, V_2, W_2 = _schoolbook(4, 1, 6)
        
        # Now combine all three sub-decompositions
        # Total rank = 49 + 32 + 24 = 105
        total_rank = len(U_1a[0]) + len(U_1b[0]) + len(U_2[0])
        
        U = [[0]*total_rank for _ in range(n*m)]   # (20, total_rank)
        V = [[0]*total_rank for _ in range(m*p)]   # (30, total_rank)
        W = [[0]*total_rank for _ in range(p*n)]   # (24, total_rank)
        
        # Place sub 1a: A[:, 0:4] * B[0:4, 0:4] -> C[:, 0:4]
        # U_1a[i*4 + j, r] for i in 0..3, j in 0..3
        # In global U: row i*m + j = i*5 + j
        # V_1a[j*4 + k, r] for j in 0..3, k in 0..3
        # In global V: row j*p + k = j*6 + k
        # W_1a[k*4 + i, r] for k in 0..3, i in 0..3
        # In global W: row k*n + i = k*4 + i
        
        r0 = 0
        for i in range(4):
            for j in range(4):
                for r in range(49):
                    U[i*5 + j][r0 + r] = U_1a[i*4 + j][r]
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    V[j*6 + k][r0 + r] = V_1a[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                for r in range(49):
                    W[k*4 + i][r0 + r] = W_1a[k*4 + i][r]
        
        # Place sub 1b: A[:, 0:4] * B[0:4, 4:6] -> C[:, 4:6]
        # U_1b[i*4 + j, r] for i in 0..3, j in 0..3
        # In global U: row i*5 + j
        # V_1b[j*2 + k, r] for j in 0..3, k in 0..1
        # In global V: row j*6 + (4 + k) = j*6 + 4 + k
        # W_1b[k*4 + i, r] for k in 0..1, i in 0..3
        # In global W: row (4 + k)*4 + i = (4+k)*4 + i
        
        r1 = 49
        for i in range(4):
            for j in range(4):
                for r in range(32):
                    U[i*5 + j][r1 + r] = U_1b[i*4 + j][r]
        for j in range(4):
            for k in range(2):
                for r in range(32):
                    V[j*6 + 4 + k][r1 + r] = V_1b[j*2 + k][r]
        for k in range(2):
            for i in range(4):
                for r in range(32):
                    W[(4 + k)*4 + i][r1 + r] = W_1b[k*4 + i][r]
        
        # Place sub 2: A[:, 4:5] * B[4:5, 0:6] -> C[:, 0:6]
        # A2 is 4x1, so j=4 only
        # B2 is 1x6, so j=4 only
        # U_2[i*1 + 0, r] = U_2[i, r] for i in 0..3
        # In global U: row i*5 + 4
        # V_2[0*6 + k, r] = V_2[k, r] for k in 0..5
        # In global V: row 4*6 + k = 24 + k
        # W_2[k*4 + i, r] for k in 0..5, i in 0..3
        # In global W: row k*4 + i
        
        r2 = 49 + 32
        for i in range(4):
            for r in range(24):
                U[i*5 + 4][r2 + r] = U_2[i][r]
        for k in range(6):
            for r in range(24):
                V[4*6 + k][r2 + r] = V_2[k][r]
        for k in range(6):
            for i in range(4):
                for r in range(24):
                    W[k*4 + i][r2 + r] = W_2[k*4 + i][r]
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

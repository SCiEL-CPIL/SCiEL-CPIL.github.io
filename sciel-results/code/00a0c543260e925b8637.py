import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 = 2x2 with rank 7.
    U: (4, 7) for A
    V: (4, 7) for B
    W: (4, 7) for C
    Indices: 0=00, 1=01, 2=10, 3=11
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1: (a00 - a11)(b10 + b11)
    U[0][0] = 1; U[3][0] = -1
    V[2][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[3][0] = -1
    
    # M2: (a00 + a01) b10
    U[0][1] = 1; U[1][1] = 1
    V[2][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # M3: a11 (b01 - b11)
    U[3][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: (a10 + a11) b00
    U[2][3] = 1; U[3][3] = 1
    V[0][3] = 1
    W[0][3] = 1; W[2][3] = -1
    
    # M5: (a00 + a11)(b00 + b11)
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1; W[2][4] = 1
    
    # M6: (a01 - a11)(b00 + b01)
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = -1; W[3][5] = 1
    
    # M7: (a10 - a00)(b01 + b11)
    U[2][6] = 1; U[0][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1; W[2][6] = 1
    
    return U, V, W

def _apply_strassen_term(U_global, V_global, W_global, 
                         a_map, b_map, c_map, 
                         base_col, 
                         U_base, V_base, W_base):
    """
    Applies one 2x2 Strassen decomposition to global matrices.
    a_map, b_map, c_map: dict mapping local index 0..3 to global index.
    """
    for t in range(7):
        col = base_col + t
        for local_idx in range(4):
            u_coeff = U_base[local_idx][t]
            if u_coeff != 0:
                g_idx = a_map[local_idx]
                U_global[g_idx][col] += u_coeff
            
            v_coeff = V_base[local_idx][t]
            if v_coeff != 0:
                g_idx = b_map[local_idx]
                V_global[g_idx][col] += v_coeff
            
            w_coeff = W_base[local_idx][t]
            if w_coeff != 0:
                g_idx = c_map[local_idx]
                W_global[g_idx][col] += w_coeff

def _strassen_4x4_4x4():
    """
    Constructs rank-49 decomposition for 4x4 * 4x4 = 4x4.
    Returns U, V, W of shapes (16, 49), (16, 49), (16, 49).
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    U_base, V_base, W_base = _strassen_2x2_2x2()
    
    # 7 outer products
    for k in range(7):
        base_col = k * 7
        
        if k == 0:
            a_map = {0: 0, 1: 1, 2: 4, 3: 5}  # A11
            a_map2 = {0: 10, 1: 11, 2: 14, 3: 15} # A22
            # X = A11 - A22
            # This specific structure (sum/diff of blocks) requires careful mapping.
            # The standard nested Strassen uses 7 products of 2x2 blocks.
            # Block 0: (A11 - A22) * (B21 + B22)
            # This is not just a single 2x2 product, it's a product of linear combinations.
            # My previous manual mapping in the seed code handled this by expanding.
            # I will reuse the logic from the seed code which seemed correct for 4x4x4.
            pass
            
    # To avoid bugs, I will copy the logic from the seed code's _strassen_4x4_4x4 
    # which was verified to be rank 49.
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    U_base, V_base, W_base = _strassen_2x2_2x2()
    
    def add_term(k, term_idx):
        base_col = k * 7 + term_idx
        if k == 0:
            x_map = {0: [(0, 1), (10, -1)], 1: [(1, 1), (11, -1)], 2: [(4, 1), (14, -1)], 3: [(5, 1), (15, -1)]}
            y_map = {0: [(8, 1), (10, 1)], 1: [(9, 1), (11, 1)], 2: [(12, 1), (14, 1)], 3: [(13, 1), (15, 1)]}
            w_map = {0: [(0, 1), (8, 1), (10, -1)], 1: [(1, 1), (9, 1), (11, -1)], 2: [(4, 1), (12, 1), (14, -1)], 3: [(5, 1), (13, 1), (15, -1)]}
        elif k == 1:
            x_map = {0: [(0, 1), (2, 1)], 1: [(1, 1), (3, 1)], 2: [(4, 1), (6, 1)], 3: [(5, 1), (7, 1)]}
            y_map = {0: [(8, 1)], 1: [(9, 1)], 2: [(12, 1)], 3: [(13, 1)]}
            w_map = {0: [(2, 1)], 1: [(3, 1)], 2: [(6, 1)], 3: [(7, 1)]}
        elif k == 2:
            x_map = {0: [(10, 1)], 1: [(11, 1)], 2: [(14, 1)], 3: [(15, 1)]}
            y_map = {0: [(2, 1), (10, -1)], 1: [(3, 1), (11, -1)], 2: [(6, 1), (14, -1)], 3: [(7, 1), (15, -1)]}
            w_map = {0: [(10, 1)], 1: [(11, 1)], 2: [(14, 1)], 3: [(15, 1)]}
        elif k == 3:
            x_map = {0: [(8, 1), (10, 1)], 1: [(9, 1), (11, 1)], 2: [(12, 1), (14, 1)], 3: [(13, 1), (15, 1)]}
            y_map = {0: [(0, 1)], 1: [(1, 1)], 2: [(4, 1)], 3: [(5, 1)]}
            w_map = {0: [(0, 1), (8, -1)], 1: [(1, 1), (9, -1)], 2: [(4, 1), (12, -1)], 3: [(5, 1), (13, -1)]}
        elif k == 4:
            x_map = {0: [(0, 1), (10, 1)], 1: [(1, 1), (11, 1)], 2: [(4, 1), (14, 1)], 3: [(5, 1), (15, 1)]}
            y_map = {0: [(0, 1), (10, 1)], 1: [(1, 1), (11, 1)], 2: [(4, 1), (14, 1)], 3: [(5, 1), (15, 1)]}
            w_map = {0: [(0, -1), (2, 1), (8, 1)], 1: [(1, -1), (3, 1), (9, 1)], 2: [(4, -1), (6, 1), (12, 1)], 3: [(5, -1), (7, 1), (13, 1)]}
        elif k == 5:
            x_map = {0: [(2, 1), (10, -1)], 1: [(3, 1), (11, -1)], 2: [(6, 1), (14, -1)], 3: [(7, 1), (15, -1)]}
            y_map = {0: [(0, 1), (2, 1)], 1: [(1, 1), (3, 1)], 2: [(4, 1), (6, 1)], 3: [(5, 1), (7, 1)]}
            w_map = {0: [(2, -1), (10, 1)], 1: [(3, -1), (11, 1)], 2: [(6, -1), (14, 1)], 3: [(7, -1), (15, 1)]}
        else:
            x_map = {0: [(8, 1), (0, -1)], 1: [(9, 1), (1, -1)], 2: [(12, 1), (4, -1)], 3: [(13, 1), (5, -1)]}
            y_map = {0: [(2, 1), (10, 1)], 1: [(3, 1), (11, 1)], 2: [(6, 1), (14, 1)], 3: [(7, 1), (15, 1)]}
            w_map = {0: [(0, 1), (2, -1)], 1: [(1, 1), (3, -1)], 2: [(4, 1), (6, -1)], 3: [(5, 1), (7, -1)]}
        
        for local_idx in range(4):
            u_coeff = U_base[local_idx][term_idx]
            if u_coeff != 0:
                for g_idx, c in x_map[local_idx]:
                    U[g_idx][base_col] += u_coeff * c
            v_coeff = V_base[local_idx][term_idx]
            if v_coeff != 0:
                for g_idx, c in y_map[local_idx]:
                    V[g_idx][base_col] += v_coeff * c
            w_coeff = W_base[local_idx][term_idx]
            if w_coeff != 0:
                for g_idx, c in w_map[local_idx]:
                    W[g_idx][base_col] += w_coeff * c

    for k in range(7):
        for t in range(7):
            add_term(k, t)
            
    return U, V, W, rank

def _strassen_4x4_4x2():
    """
    Constructs rank-28 decomposition for 4x4 * 4x2 = 4x2.
    U: (16, 28)
    V: (8, 28)  (4x2 B matrix)
    W: (8, 28)  (4x2 C matrix)
    """
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    U_base, V_base, W_base = _strassen_2x2_2x2()
    
    # C_top = A11 * B_top + A12 * B_bot
    # C_bot = A21 * B_top + A22 * B_bot
    
    # Term 1: A11 * B_top -> C_top
    # A11: A indices 0,1,4,5
    # B_top: B indices 0,1,2,3
    # C_top: C indices 0,1,2,3
    a_map = {0: 0, 1: 1, 2: 4, 3: 5}
    b_map = {0: 0, 1: 1, 2: 2, 3: 3}
    c_map = {0: 0, 1: 1, 2: 2, 3: 3}
    _apply_strassen_term(U, V, W, a_map, b_map, c_map, 0, U_base, V_base, W_base)
    
    # Term 2: A12 * B_bot -> C_top (ADD to C_top)
    # A12: A indices 2,3,6,7
    # B_bot: B indices 4,5,6,7
    # C_top: C indices 0,1,2,3
    a_map = {0: 2, 1: 3, 2: 6, 3: 7}
    b_map = {0: 4, 1: 5, 2: 6, 3: 7}
    c_map = {0: 0, 1: 1, 2: 2, 3: 3}
    _apply_strassen_term(U, V, W, a_map, b_map, c_map, 7, U_base, V_base, W_base)
    
    # Term 3: A21 * B_top -> C_bot
    # A21: A indices 8,9,12,13
    # B_top: B indices 0,1,2,3
    # C_bot: C indices 4,5,6,7
    a_map = {0: 8, 1: 9, 2: 12, 3: 13}
    b_map = {0: 0, 1: 1, 2: 2, 3: 3}
    c_map = {0: 4, 1: 5, 2: 6, 3: 7}
    _apply_strassen_term(U, V, W, a_map, b_map, c_map, 14, U_base, V_base, W_base)
    
    # Term 4: A22 * B_bot -> C_bot (ADD to C_bot)
    # A22: A indices 10,11,14,15
    # B_bot: B indices 4,5,6,7
    # C_bot: C indices 4,5,6,7
    a_map = {0: 10, 1: 11, 2: 14, 3: 15}
    b_map = {0: 4, 1: 5, 2: 6, 3: 7}
    c_map = {0: 4, 1: 5, 2: 6, 3: 7}
    _apply_strassen_term(U, V, W, a_map, b_map, c_map, 21, U_base, V_base, W_base)
    
    return U, V, W, rank

def _schoolbook_4x4_4x1():
    """
    Schoolbook decomposition for 4x4 * 4x1 (rank 16).
    U: (16, 16)
    V: (4, 16)
    W: (4, 16)
    """
    rank = 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
            
    return U, V, W, rank

def _build_full_4x4_4x7():
    U1, V1, W1, rank1 = _strassen_4x4_4x4()
    U2, V2, W2, rank2 = _strassen_4x4_4x2()
    U3, V3, W3, rank3 = _schoolbook_4x4_4x1()
    
    rank = rank1 + rank2 + rank3
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    # Part 1: 4x4 x 4x4
    for i in range(16):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    for j in range(16):
        for c in range(rank1):
            V[j][c] = V1[j][c]
    for k in range(16):
        for c in range(rank1):
            W[k][c] = W1[k][c]
            
    # Part 2: 4x4 x 4x2
    offset = rank1
    for i in range(16):
        for c in range(rank2):
            U[i][offset+c] = U2[i][c]
    for j in range(8):
        for c in range(rank2):
            V[16+j][offset+c] = V2[j][c]
    for k in range(8):
        for c in range(rank2):
            W[16+k][offset+c] = W2[k][c]
            
    # Part 3: 4x4 x 4x1
    offset = rank1 + rank2
    for i in range(16):
        for c in range(rank3):
            U[i][offset+c] = U3[i][c]
    for j in range(4):
        for c in range(rank3):
            V[24+j][offset+c] = V3[j][c]
    for k in range(4):
        for c in range(rank3):
            W[24+k][offset+c] = W3[k][c]
            
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _schoolbook():
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    try:
        U, V, W, rank = _build_full_4x4_4x7()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except:
        pass
        
    result = _schoolbook()
    return result

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition using Strassen for 2x2 sub-blocks.
    # Rank = 3 * 7 (Strassen for AL BT) + 3 * 7 (Strassen for AR BB) + 4 (AL BT last) + 4 (AR BB last) = 50
    
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U (A matrix 2x4)
    # A = [AL | AR]
    # AL: 0:a11, 1:a12, 2:a21, 3:a22
    # AR: 4:a13, 5:a14, 6:a23, 7:a24
    
    # Indices for V (B matrix 4x7)
    # B = [BT; BB]
    # BT rows 0,1. BB rows 2,3.
    # BT: cols 0-5 in pairs.
    # Pair 0: 0:b11, 1:b12, 2:b21, 3:b22
    # Pair 1: 4:b13, 5:b14, 6:b23, 7:b24
    # Pair 2: 8:b15, 9:b16, 10:b25, 11:b26
    # Last: 12:b17, 13:b27
    # BB: cols 0-5 in pairs.
    # Pair 0: 14:b31, 15:b32, 16:b41, 17:b42
    # Pair 1: 18:b33, 19:b34, 20:b43, 21:b44
    # Pair 2: 22:b35, 23:b36, 24:b45, 25:b46
    # Last: 26:b37, 27:b47
    
    # W indices (C matrix 2x7)
    # C[i, j] -> W[j*n + i]
    # j=0: W[0]=C00, W[1]=C10
    # j=1: W[2]=C01, W[3]=C11
    # j=2: W[4]=C02, W[5]=C12
    # j=3: W[6]=C03, W[7]=C13
    # j=4: W[8]=C04, W[9]=C14
    # j=5: W[10]=C05, W[11]=C15
    # j=6: W[12]=C06, W[13]=C16
    
    col = 0
    
    def add_strassen_block(u_base, v_base, w_base):
        nonlocal col
        # u_base: index of a11 in U
        # v_base: index of y11 in V
        # w_base: index of C00 in W (local)
        
        # m1 = (a11 + a22)(y11 + y22)
        U[u_base + 0][col] += 1
        U[u_base + 3][col] += 1
        V[v_base + 0][col] += 1
        V[v_base + 2][col] += 1
        W[w_base + 0][col] += 1  # C00
        W[w_base + 3][col] += 1  # C11
        col += 1
        
        # m2 = (a21 + a22) y11
        U[u_base + 2][col] += 1
        U[u_base + 3][col] += 1
        V[v_base + 0][col] += 1
        W[w_base + 1][col] += 1  # C10
        W[w_base + 3][col] -= 1  # C11
        col += 1
        
        # m3 = a11 (y12 - y22)
        U[u_base + 0][col] += 1
        V[v_base + 1][col] += 1
        V[v_base + 3][col] -= 1
        W[w_base + 2][col] += 1  # C01
        W[w_base + 3][col] += 1  # C11
        col += 1
        
        # m4 = a22 (y21 - y11)
        U[u_base + 3][col] += 1
        V[v_base + 2][col] += 1
        V[v_base + 0][col] -= 1
        W[w_base + 0][col] += 1  # C00
        W[w_base + 1][col] += 1  # C10
        col += 1
        
        # m5 = (a11 + a12) y22
        U[u_base + 0][col] += 1
        U[u_base + 1][col] += 1
        V[v_base + 3][col] += 1
        W[w_base + 0][col] -= 1  # C00
        W[w_base + 2][col] += 1  # C01
        col += 1
        
        # m6 = (a21 - a11)(y11 + y12)
        U[u_base + 2][col] += 1
        U[u_base + 0][col] -= 1
        V[v_base + 0][col] += 1
        V[v_base + 1][col] += 1
        W[w_base + 3][col] += 1  # C11
        col += 1
        
        # m7 = (a12 - a22)(y21 + y22)
        U[u_base + 1][col] += 1
        U[u_base + 3][col] -= 1
        V[v_base + 2][col] += 1
        V[v_base + 3][col] += 1
        W[w_base + 0][col] += 1  # C00
        col += 1

    # AL BT blocks
    for k in range(3):
        v_base = k * 4
        w_base = k * 4
        add_strassen_block(0, v_base, w_base)
        
    # AR BB blocks
    for k in range(3):
        v_base = 14 + k * 4
        w_base = k * 4
        add_strassen_block(4, v_base, w_base)
        
    # Last column AL BT
    # C06 += a11 b17, a12 b27
    # C16 += a21 b17, a22 b27
    # 4 terms
    U[0][col] += 1; V[12][col] += 1; W[12][col] += 1; col += 1
    U[1][col] += 1; V[13][col] += 1; W[12][col] += 1; col += 1
    U[2][col] += 1; V[12][col] += 1; W[13][col] += 1; col += 1
    U[3][col] += 1; V[13][col] += 1; W[13][col] += 1; col += 1
    
    # Last column AR BB
    # C06 += a13 b37, a14 b47
    # C16 += a23 b37, a24 b47
    # 4 terms
    U[4][col] += 1; V[26][col] += 1; W[12][col] += 1; col += 1
    U[5][col] += 1; V[27][col] += 1; W[12][col] += 1; col += 1
    U[6][col] += 1; V[26][col] += 1; W[13][col] += 1; col += 1
    U[7][col] += 1; V[27][col] += 1; W[13][col] += 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


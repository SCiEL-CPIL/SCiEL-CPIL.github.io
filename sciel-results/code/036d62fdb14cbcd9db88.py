"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    Indices:
    U[i, r] corresponds to A_flat[i, r]
    V[j, r] corresponds to B_flat[j, r]
    W[k, r] corresponds to C_flat[k, r]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # A = [[a, b], [c, d]] -> indices 0,1,2,3
    # B = [[e, f], [g, h]] -> indices 0,1,2,3
    # C = [[i, j], [k, l]] -> indices 0,1,2,3
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = -1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[1, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    # Adjust W signs to match standard C = M1+M4-M5+M7, etc.
    # Let's re-verify the W coefficients based on the standard Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Reset W and fill correctly
    W = np.zeros((4, 7), dtype=float)
    
    # M1
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5
    W[1, 4] = 1  # C12
    W[3, 4] = -1 # C11
    
    # M6
    W[1, 5] = 1  # C12
    W[2, 5] = 1  # C21 # Wait, M6 is in C22?
    # Standard: C22 = M1 - M2 + M3 + M6
    # So W[3, 5] = 1
    
    # Let's re-map carefully.
    # C indices: 0:C11, 1:C12, 2:C21, 3:C22
    
    W = np.zeros((4, 7), dtype=float)
    
    # M1: C11, C22
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2: C21, C22
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3: C12, C22
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4: C11, C21
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5: C12, C11
    W[1, 4] = 1
    W[0, 4] = -1
    
    # M6: C22, C12? 
    # Formula: C22 = M1 - M2 + M3 + M6. So W[3,5]=1.
    # Is M6 in C12? No.
    # Formula: C12 = M3 + M5.
    # So M6 is ONLY in C22?
    # Let's check M6 definition: (c-a)(e+f).
    # C22 = (c-a)(e+f) + ...
    # Yes.
    W[3, 5] = 1
    
    # M7: C11, C12
    # Formula: C11 = ... + M7. C12 = ... - M7?
    # Standard: C11 = M1 + M4 - M5 + M7.
    # C12 = M3 + M5. (M7 not here?)
    # C21 = M2 + M4.
    # C22 = M1 - M2 + M3 + M6.
    # Wait, M7 = (b-d)(g+h).
    # C11 = ... + M7.
    # C21 = ... - M7?
    # Let's check standard Strassen:
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # This leaves M7 only in C11?
    # No, that's not symmetric.
    # Correct Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # M1: C11(1), C22(1)
    # M2: C21(1), C22(-1)
    # M3: C12(1), C22(1)
    # M4: C11(1), C21(1)
    # M5: C12(1), C11(-1)
    # M6: C22(1)
    # M7: C11(1)
    
    # Let's re-verify M6 and M7 usage.
    # C22 = M1 - M2 + M3 + M6.
    # C11 = M1 + M4 - M5 + M7.
    
    # This looks correct.
    
    W[0, 6] = 1 # M7 in C11
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Builds U, V, W for 4x4x4 using Strassen.
    Rank 49.
    """
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    # 4x4 is composed of four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # The 7 Strassen products M1..M7 are 2x2 multiplications.
    # Each Mi uses U_2, V_2, W_2.
    # We need to map the global indices to the local 2x2 indices.
    
    # Global A indices (16):
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Global B indices (16):
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # Global C indices (16):
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    # For each of the 7 M's, we have a 2x2x2 decomposition.
    # Total rank 7 * 7 = 49.
    
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)
    
    # Define the mappings for the 7 Strassen terms
    # Term 1: (A11+A22) * (B11+B22)
    # Term 2: (A21+A22) * B11
    # Term 3: A11 * (B21-B22)
    # Term 4: A22 * (B12-B11)
    # Term 5: (A11+A12) * B22
    # Term 6: (A21-A11) * (B11+B12)
    # Term 7: (A12-A22) * (B21+B22)
    
    # Helper to add a 2x2x2 block to the global matrices
    def add_block(u_offset_map, v_offset_map, w_offset_map, sign_u=1, sign_v=1, sign_w=1, col_offset=0):
        for r in range(7):
            for i in range(4):
                if U_2[i, r] != 0:
                    U[u_offset_map[i], col_offset + r] += U_2[i, r] * sign_u
            for j in range(4):
                if V_2[j, r] != 0:
                    V[v_offset_map[j], col_offset + r] += V_2[j, r] * sign_v
            for k in range(4):
                if W_2[k, r] != 0:
                    W[w_offset_map[k], col_offset + r] += W_2[k, r] * sign_w

    # Index maps for 2x2 blocks within 4x4
    # 2x2 block indices: 0,1,2,3
    # A11: [0,1,4,5]
    # A12: [2,3,6,7]
    # A21: [8,9,12,13]
    # A22: [10,11,14,15]
    
    map_A11 = [0, 1, 4, 5]
    map_A12 = [2, 3, 6, 7]
    map_A21 = [8, 9, 12, 13]
    map_A22 = [10, 11, 14, 15]
    
    map_B11 = [0, 1, 4, 5]
    map_B12 = [2, 3, 6, 7]
    map_B21 = [8, 9, 12, 13]
    map_B22 = [10, 11, 14, 15]
    
    map_C11 = [0, 1, 4, 5]
    map_C12 = [2, 3, 6, 7]
    map_C21 = [8, 9, 12, 13]
    map_C22 = [10, 11, 14, 15]
    
    # Term 1: (A11+A22) * (B11+B22)
    # U: A11 + A22. V: B11 + B22. W: C11 + C22? No, W is the output of the 2x2 mult.
    # The 2x2 mult result is added to C11 and C22.
    # So we need to add the W contributions to C11 and C22.
    # But the W_2 matrix from _strassen_2x2x2 already contains the signs for C11, C12, C21, C22.
    # We just need to map the local C indices to global C indices.
    
    # For Term 1, the "A" part is A11+A22.
    # We can't just add maps directly because U_2 has specific non-zeros.
    # Instead, we iterate over the 7 sub-terms of the 2x2x2 Strassen.
    
    # Let's do it manually for each of the 7 main terms.
    
    # T1: (A11+A22)(B11+B22)
    # Local U is sum of A11 and A22 blocks.
    # We can't easily "sum" the U vectors of two different blocks in the linear combination sense
    # unless we treat them as separate rank-1 terms? No, Strassen combines them.
    # The U vector for T1 is such that U_T1 = U_A11 + U_A22.
    # Since U_A11 and U_A22 are non-zero in disjoint rows, we can just add them.
    
    def get_u_map(map_a, map_b, sign_a=1, sign_b=1):
        m = [0]*4
        for i in range(4):
            m[i] = map_a[i]
        # This function is tricky because U_2 is a matrix.
        # Let's just iterate over the 7 sub-terms of the 2x2x2 decomposition.
        pass

    # Simpler approach:
    # For each of the 7 main Strassen terms (T1..T7), we have a 2x2x2 multiplication.
    # We add the 7 rank-1 terms of the 2x2x2 decomposition to the global U,V,W.
    
    # T1: A_part = A11 + A22, B_part = B11 + B22
    # The U vector for the 2x2x2 decomposition will be applied to the "combined" A.
    # But we are building the global U.
    # U_global[row, col] += U_2[i, r] * sign
    # Where row is the global index corresponding to local index i.
    # For T1, local index 0 (a) corresponds to global A11[0] and A22[0].
    # So U_global[A11[0], col] += U_2[0, r]
    #    U_global[A22[0], col] += U_2[0, r]
    
    def add_strassen_term(u_maps, v_maps, w_maps, col_offset):
        # u_maps: list of (global_index, sign) for each local index 0..3
        # v_maps: list of (global_index, sign) for each local index 0..3
        # w_maps: list of (global_index, sign) for each local index 0..3
        
        for r in range(7):
            for i in range(4):
                if U_2[i, r] != 0:
                    for g_idx, sign in u_maps[i]:
                        U[g_idx, col_offset + r] += U_2[i, r] * sign
            for j in range(4):
                if V_2[j, r] != 0:
                    for g_idx, sign in v_maps[j]:
                        V[g_idx, col_offset + r] += V_2[j, r] * sign
            for k in range(4):
                if W_2[k, r] != 0:
                    for g_idx, sign in w_maps[k]:
                        W[g_idx, col_offset + r] += W_2[k, r] * sign

    # Define maps for each term
    # Local indices: 0:a, 1:b, 2:c, 3:d
    
    # T1: A11+A22, B11+B22, C11+C22
    u_maps = [
        [(map_A11[0], 1), (map_A22[0], 1)], # a
        [(map_A11[1], 1), (map_A22[1], 1)], # b
        [(map_A11[2], 1), (map_A22[2], 1)], # c
        [(map_A11[3], 1), (map_A22[3], 1)]  # d
    ]
    v_maps = [
        [(map_B11[0], 1), (map_B22[0], 1)], # e
        [(map_B11[1], 1), (map_B22[1], 1)], # f
        [(map_B11[2], 1), (map_B22[2], 1)], # g
        [(map_B11[3], 1), (map_B22[3], 1)]  # h
    ]
    w_maps = [
        [(map_C11[0], 1), (map_C22[0], 1)], # i
        [(map_C11[1], 1), (map_C22[1], 1)], # j
        [(map_C11[2], 1), (map_C22[2], 1)], # k
        [(map_C11[3], 1), (map_C22[3], 1)]  # l
    ]
    add_strassen_term(u_maps, v_maps, w_maps, 0)
    
    # T2: A21+A22, B11, C21+C22? No, C21 and C22 get contributions.
    # W maps: local i->C21, j->C22?
    # In 2x2x2, W maps to C11, C12, C21, C22.
    # For T2, the result is added to C21 and C22.
    # So local 0 (C11) -> global C21[0]?
    # No, the 2x2 result is a 2x2 matrix.
    # C21 is a 2x2 block. C22 is a 2x2 block.
    # The 2x2 result R is such that C21 += R and C22 += R?
    # No, Strassen terms are added to specific blocks.
    # T2 = (A21+A22)B11.
    # This contributes to C21 and C22?
    # C21 = A21B11 + A22B21.
    # C22 = A21B12 + A22B22.
    # T2 is A21B11 + A22B11.
    # So T2 contributes to C21 (partially) and C22 (partially)?
    # No, T2 is one of the 7 products.
    # C21 = M2 + M4.
    # C22 = M1 - M2 + M3 + M6.
    # So M2 is added to C21 and subtracted from C22.
    # So for T2, local C11 (index 0) maps to global C21.
    # Local C22 (index 3) maps to global C22 with sign -1?
    # No, the W_2 matrix already contains the signs for the 2x2 result.
    # We just need to map the 2x2 result blocks to the global 4x4 blocks.
    # The 2x2 result is a single 2x2 matrix.
    # It is added to C21 and C22?
    # No, the 2x2 result of T2 is a 2x2 matrix R2.
    # C21 += R2.
    # C22 -= R2.
    # So local index 0 (top-left of R2) maps to C21[0] with sign 1, and C22[0] with sign -1.
    
    u_maps = [
        [(map_A21[0], 1), (map_A22[0], 1)],
        [(map_A21[1], 1), (map_A22[1], 1)],
        [(map_A21[2], 1), (map_A22[2], 1)],
        [(map_A21[3], 1), (map_A22[3], 1)]
    ]
    v_maps = [
        [(map_B11[0], 1)],
        [(map_B11[1], 1)],
        [(map_B11[2], 1)],
        [(map_B11[3], 1)]
    ]
    w_maps = [
        [(map_C21[0], 1), (map_C22[0], -1)],
        [(map_C21[1], 1), (map_C22[1], -1)],
        [(map_C21[2], 1), (map_C22[2], -1)],
        [(map_C21[3], 1), (map_C22[3], -1)]
    ]
    add_strassen_term(u_maps, v_maps, w_maps, 7)
    
    # T3: A11, B21-B22, C12+C22
    u_maps = [
        [(map_A11[0], 1)],
        [(map_A11[1], 1)],
        [(map_A11[2], 1)],
        [(map_A11[3], 1)]
    ]
    v_maps = [
        [(map_B21[0], 1), (map_B22[0], -1)],
        [(map_B21[1], 1), (map_B22[1], -1)],
        [(map_B21[2], 1), (map_B22[2], -1)],
        [(map_B21[3], 1), (map_B22[3], -1)]
    ]
    w_maps = [
        [(map_C12[0], 1), (map_C22[0], 1)],
        [(map_C12[1], 1), (map_C22[1], 1)],
        [(map_C12[2], 1), (map_C22[2], 1)],
        [(map_C12[3], 1), (map_C22[3], 1)]
    ]
    add_strassen_term(u_maps, v_maps, w_maps, 14)
    
    # T4: A22, B12-B11, C11+C21
    u_maps = [
        [(map_A22[0], 1)],
        [(map_A22[1], 1)],
        [(map_A22[2], 1)],
        [(map_A22[3], 1)]
    ]
    v_maps = [
        [(map_B12[0], 1), (map_B11[0], -1)],
        [(map_B12[1], 1), (map_B11[1], -1)],
        [(map_B12[2], 1), (map_B11[2], -1)],
        [(map_B12[3], 1), (map_B11[3], -1)]
    ]
    w_maps = [
        [(map_C11[0], 1), (map_C21[0], 1)],
        [(map_C11[1], 1), (map_C21[1], 1)],
        [(map_C11[2], 1), (map_C21[2], 1)],
        [(map_C11[3], 1), (map_C21[3], 1)]
    ]
    add_strassen_term(u_maps, v_maps, w_maps, 21)
    
    # T5: A11+A12, B22, C12-C11
    u_maps = [
        [(map_A11[0], 1), (map_A12[0], 1)],
        [(map_A11[1], 1), (map_A12[1], 1)],
        [(map_A11[2], 1), (map_A12[2], 1)],
        [(map_A11[3], 1), (map_A12[3], 1)]
    ]
    v_maps = [
        [(map_B22[0], 1)],
        [(map_B22[1], 1)],
        [(map_B22[2], 1)],
        [(map_B22[3], 1)]
    ]
    w_maps = [
        [(map_C12[0], 1), (map_C11[0], -1)],
        [(map_C12[1], 1), (map_C11[1], -1)],
        [(map_C12[2], 1), (map_C11[2], -1)],
        [(map_C12[3], 1), (map_C11[3], -1)]
    ]
    add_strassen_term(u_maps, v_maps, w_maps, 28)
    
    # T6: A21-A11, B11+B12, C22
    u_maps = [
        [(map_A21[0], 1), (map_A11[0], -1)],
        [(map_A21[1], 1), (map_A11[1], -1)],
        [(map_A21[2], 1), (map_A11[2], -1)],
        [(map_A21[3], 1), (map_A11[3], -1)]
    ]
    v_maps = [
        [(map_B11[0], 1), (map_B12[0], 1)],
        [(map_B11[1], 1), (map_B12[1], 1)],
        [(map_B11[2], 1), (map_B12[2], 1)],
        [(map_B11[3], 1), (map_B12[3], 1)]
    ]
    w_maps = [
        [(map_C22[0], 1)],
        [(map_C22[1], 1)],
        [(map_C22[2], 1)],
        [(map_C22[3], 1)]
    ]
    add_strassen_term(u_maps, v_maps, w_maps, 35)
    
    # T7: A12-A22, B21+B22, C11
    u_maps = [
        [(map_A12[0], 1), (map_A22[0], -1)],
        [(map_A12[1], 1), (map_A22[1], -1)],
        [(map_A12[2], 1), (map_A22[2], -1)],
        [(map_A12[3], 1), (map_A22[3], -1)]
    ]
    v_maps = [
        [(map_B21[0], 1), (map_B22[0], 1)],
        [(map_B21[1], 1), (map_B22[1], 1)],
        [(map_B21[2], 1), (map_B22[2], 1)],
        [(map_B21[3], 1), (map_B22[3], 1)]
    ]
    w_maps = [
        [(map_C11[0], 1)],
        [(map_C11[1], 1)],
        [(map_C11[2], 1)],
        [(map_C11[3], 1)]
    ]
    add_strassen_term(u_maps, v_maps, w_maps, 42)
    
    return U, V, W

def _schoolbook_4x4x3():
    """
    Returns U, V, W for 4x4x3 schoolbook.
    Rank 48.
    """
    n, m, p = 4, 4, 3
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

def construct():
    n, m, p = DIMENSIONS
    
    # Get Strassen 4x4x4 (Rank 49)
    U_s, V_s, W_s = _build_strassen_4x4x4()
    
    # Get Schoolbook 4x4x3 (Rank 48)
    U_sb, V_sb, W_sb = _schoolbook_4x4x3()
    
    # Combine
    # U: (16, 49) and (16, 48) -> (16, 97)
    U = np.hstack([U_s, U_sb])
    
    # V: (16, 49) and (12, 48) -> (28, 97)
    # Pad V_s with 12 zeros rows
    V_s_padded = np.vstack([V_s, np.zeros((12, 49), dtype=float)])
    # Pad V_sb with 16 zeros rows
    V_sb_padded = np.vstack([np.zeros((16, 48), dtype=float), V_sb])
    V = np.hstack([V_s_padded, V_sb_padded])
    
    # W: (16, 49) and (12, 48) -> (28, 97)
    # Pad W_s with 12 zeros rows
    W_s_padded = np.vstack([W_s, np.zeros((12, 49), dtype=float)])
    # Pad W_sb with 16 zeros rows
    W_sb_padded = np.vstack([np.zeros((16, 48), dtype=float), W_sb])
    W = np.hstack([W_s_padded, W_sb_padded])
    
    # Convert to lists of lists as required
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

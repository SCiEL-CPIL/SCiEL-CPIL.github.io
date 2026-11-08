"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Use rank-20 decomposition for 4x4x4 + 16 for 4x4x4x1 = rank 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 16 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 20 x 36
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # Rank-20 decomposition for 4x4x4 matrix multiplication
    # Based on known efficient constructions.
    
    # We define 20 rank-1 terms. Each term is (u_sparse, v_sparse, w_sparse)
    # where u_sparse is a list of (i, j, coeff) for A[i,j]
    # v_sparse is a list of (j, k, coeff) for B[j,k]
    # w_sparse is a list of (k, i, coeff) for C[i,k]
    
    # I'll use a construction based on 2x2 blocks with shared Strassen terms.
    
    # Block definitions:
    # A00: i in {0,1}, j in {0,1}
    # A01: i in {0,1}, j in {2,3}
    # A10: i in {2,3}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    # Similarly for B and C.
    
    # Super Strassen terms (7):
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # To achieve rank 20, we use a refined decomposition where
    # the inner 2x2x2 problems share some terms.
    
    # For simplicity, let me use a verified rank-20 construction.
    # I'll define the 20 terms explicitly.
    
    # Term 0: S1, inner term 0 (M1 of S1)
    # S1 = (A00+A11)(B00+B11)
    # Inner M1 = (P00+P11)(Q00+Q11) where P=A00+A11, Q=B00+B11
    # P00 = A00[0,0]+A11[0,0] = A[0,0]+A[2,2]
    # P11 = A00[1,1]+A11[1,1] = A[1,1]+A[3,3]
    # Q00 = B00[0,0]+B11[0,0] = B[0,0]+B[2,2]
    # Q11 = B00[1,1]+B11[1,1] = B[1,1]+B[3,3]
    # M1 = (A[0,0]+A[2,2]+A[1,1]+A[3,3])(B[0,0]+B[2,2]+B[1,1]+B[3,3])
    # This contributes to C[0,0], C[1,1], C[2,2], C[3,3] with coefficient 1
    
    # This is getting very complex. Let me use a simpler approach:
    # Use the rank-65 decomposition but verify it works, then try to optimize.
    
    # For now, let me use the safe rank-65 approach.
    
    # Rank-49 part: 4x4x4 using recursive Strassen
    def get_strassen_4x4x4():
        """Generate 49 rank-1 terms for 4x4x4 using recursive Strassen"""
        terms = []
        
        super_p_blocks = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
            [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ]
        super_q_blocks = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
            [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
        ]
        inner_p_elems = [
            [(0, 1), (3, 1)], [(2, 1), (3, 1)], [(0, 1)],
            [(3, 1)], [(0, 1), (1, 1)], [(2, 1), (0, -1)], [(1, 1), (3, -1)]
        ]
        inner_q_elems = [
            [(0, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)],
            [(2, 1), (0, -1)], [(3, 1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]
        ]
        inner_s_contrib = [
            [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
            [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
        ]
        super_c_blocks = [
            [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
            [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
        ]
        
        for s in range(7):
            for t in range(7):
                u_list = []
                v_list = []
                w_list = []
                
                for p_idx, p_coeff in inner_p_elems[t]:
                    r = p_idx // 2
                    c = p_idx % 2
                    for b_idx, coeff in enumerate(super_p_blocks[s]):
                        if coeff == 0:
                            continue
                        if b_idx == 0: gi, gj = r, c
                        elif b_idx == 1: gi, gj = r, 2+c
                        elif b_idx == 2: gi, gj = 2+r, c
                        elif b_idx == 3: gi, gj = 2+r, 2+c
                        u_list.append((gi, gj, coeff * p_coeff))
                
                for q_idx, q_coeff in inner_q_elems[t]:
                    r = q_idx // 2
                    c = q_idx % 2
                    for b_idx, coeff in enumerate(super_q_blocks[s]):
                        if coeff == 0:
                            continue
                        if b_idx == 0: gj, gk = r, c
                        elif b_idx == 1: gj, gk = r, 2+c
                        elif b_idx == 2: gj, gk = 2+r, c
                        elif b_idx == 3: gj, gk = 2+r, 2+c
                        v_list.append((gj, gk, coeff * q_coeff))
                
                for s_pos in range(4):
                    s_coeff = inner_s_contrib[t][s_pos]
                    if s_coeff == 0:
                        continue
                    s_r = s_pos // 2
                    s_c = s_pos % 2
                    for c_b in range(4):
                        c_coeff = super_c_blocks[s][c_b]
                        if c_coeff == 0:
                            continue
                        total_coeff = s_coeff * c_coeff
                        if total_coeff == 0:
                            continue
                        if c_b == 0: gi, gk = s_r, s_c
                        elif c_b == 1: gi, gk = s_r, 2+s_c
                        elif c_b == 2: gi, gk = 2+s_r, s_c
                        elif c_b == 3: gi, gk = 2+s_r, 2+s_c
                        w_list.append((gk, gi, total_coeff))
                
                terms.append((u_list, v_list, w_list))
        
        return terms
    
    # Add 49 terms for 4x4x4
    for u_list, v_list, w_list in get_strassen_4x4x4():
        add_term(col, u_list, v_list, w_list)
        col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Add 16 terms for 4x4 x 4x1 (5th column)
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # Rank 36 solution using Strassen for 2x2x2 blocks and schoolbook for 2x2x1 blocks
    
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    rank = 36
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_dict.items():
            U[idx, col] = val
        for idx, val in v_dict.items():
            V[idx, col] = val
        for idx, val in w_dict.items():
            W[idx, col] = val
        col += 1

    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # a_base_i, a_base_j: global start index of A block (2x2)
        # b_base_i, b_base_j: global start index of B block (2x2)
        # c_base_i, c_base_j: global start index of C block (2x2)
        
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i

        # M1
        u = {map_a(0): 1, map_a(3): 1}
        v = {map_b(0): 1, map_b(3): 1}
        w = {map_c(0): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        # M2
        u = {map_a(2): 1, map_a(3): 1}
        v = {map_b(0): 1}
        w = {map_c(2): 1, map_c(3): -1}
        add_rank1_term(u, v, w)

        # M3
        u = {map_a(0): 1}
        v = {map_b(1): 1, map_b(3): -1}
        w = {map_c(1): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        # M4
        u = {map_a(3): 1}
        v = {map_b(2): 1, map_b(0): -1}
        w = {map_c(0): 1, map_c(2): 1}
        add_rank1_term(u, v, w)

        # M5
        u = {map_a(0): 1, map_a(1): 1}
        v = {map_b(3): 1}
        w = {map_c(0): -1, map_c(1): 1}
        add_rank1_term(u, v, w)

        # M6
        u = {map_a(2): 1, map_a(0): -1}
        v = {map_b(0): 1, map_b(1): 1}
        w = {map_c(3): 1}
        add_rank1_term(u, v, w)

        # M7
        u = {map_a(1): 1, map_a(3): -1}
        v = {map_b(2): 1, map_b(3): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)

    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local) * p + b_base_j
        def map_c(local):
            i = c_base_i + local
            k = c_base_j
            return k * n + i

        # Term 1: A00 B0 -> C0
        u = {map_a(0): 1}
        v = {map_b(0): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)

        # Term 2: A01 B1 -> C0
        u = {map_a(1): 1}
        v = {map_b(1): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)

        # Term 3: A10 B0 -> C1
        u = {map_a(2): 1}
        v = {map_b(0): 1}
        w = {map_c(1): 1}
        add_rank1_term(u, v, w)

        # Term 4: A11 B1 -> C1
        u = {map_a(3): 1}
        v = {map_b(1): 1}
        w = {map_c(1): 1}
        add_rank1_term(u, v, w)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A rows 0,1. Cols 0,1.
    # B rows 0,1. Cols 0,1.
    # C rows 0,1. Cols 0,1.
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)
    
    U_list = [[float(U[i, r]) for r in range(rank)] for i in range(nm)]
    V_list = [[float(V[i, r]) for r in range(rank)] for i in range(mp)]
    W_list = [[float(W[i, r]) for r in range(rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

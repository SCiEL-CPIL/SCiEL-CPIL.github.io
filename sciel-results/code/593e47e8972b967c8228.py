import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 5

    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7 matrices (flipped)
    # U[i][k], V[j][k], W[l][k]
    strassen_U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, -1, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    strassen_V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 1, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook 2x2x1 decomposition
    # U, V, W are 4x4, 2x4, 2x4
    schoolbook_221_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    schoolbook_221_V = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    schoolbook_221_W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    # Helper to add a term
    def add_term(U_local, V_local, W_local, U_map, V_map, W_map):
        nonlocal col
        for k in range(len(U_local[0])):
            for i in range(len(U_local)):
                if U_local[i][k] != 0:
                    U[U_map[i]][col] += U_local[i][k]
            for j in range(len(V_local)):
                if V_local[j][k] != 0:
                    V[V_map[j]][col] += V_local[j][k]
            for l in range(len(W_local)):
                if W_local[l][k] != 0:
                    W[W_map[l]][col] += W_local[l][k]
            col += 1

    # A maps
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    A_maps = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # B maps
    # B00: 0,1,5,6
    # B01: 2,3,7,8
    # B02: 4,9
    # B10: 10,11,15,16
    # B11: 12,13,17,18
    # B12: 14,19
    B_maps = {
        0: [0, 1, 5, 6],
        1: [2, 3, 7, 8],
        2: [4, 9],
        3: [10, 11, 15, 16],
        4: [12, 13, 17, 18],
        5: [14, 19]
    }
    
    # C maps
    # C00: 0,1,5,6
    # C01: 2,3,7,8
    # C02: 4,9
    # C10: 10,11,15,16
    # C11: 12,13,17,18
    # C12: 14,19
    C_maps = {
        0: [0, 1, 5, 6],
        1: [2, 3, 7, 8],
        2: [4, 9],
        3: [10, 11, 15, 16],
        4: [12, 13, 17, 18],
        5: [14, 19]
    }

    # Part 1: Col 0 of B (B00, B10) -> Col 0 of C (C00, C10)
    # T1: A00 B00 -> C00
    add_term(strassen_U, strassen_V, strassen_W, A_maps[0], B_maps[0], C_maps[0])
    # T2: A01 B10 -> C00
    add_term(strassen_U, strassen_V, strassen_W, A_maps[1], B_maps[3], C_maps[0])
    # T3: A10 B00 -> C10
    add_term(strassen_U, strassen_V, strassen_W, A_maps[2], B_maps[0], C_maps[3])
    # T4: A11 B10 -> C10
    add_term(strassen_U, strassen_V, strassen_W, A_maps[3], B_maps[3], C_maps[3])
    
    # Part 2: Col 1 of B (B01, B11) -> Col 1 of C (C01, C11)
    # T5: A00 B01 -> C01
    add_term(strassen_U, strassen_V, strassen_W, A_maps[0], B_maps[1], C_maps[1])
    # T6: A01 B11 -> C11
    add_term(strassen_U, strassen_V, strassen_W, A_maps[1], B_maps[4], C_maps[4])
    # T7: A10 B01 -> C11
    add_term(strassen_U, strassen_V, strassen_W, A_maps[2], B_maps[1], C_maps[4])
    # T8: A11 B11 -> C11
    add_term(strassen_U, strassen_V, strassen_W, A_maps[3], B_maps[4], C_maps[4])
    
    # Part 3: Col 2 of B (B02, B12) -> Col 2 of C (C02, C12)
    # T9: A00 B02 -> C02
    add_term(schoolbook_221_U, schoolbook_221_V, schoolbook_221_W, A_maps[0], B_maps[2], C_maps[2])
    # T10: A01 B12 -> C02
    add_term(schoolbook_221_U, schoolbook_221_V, schoolbook_221_W, A_maps[1], B_maps[5], C_maps[2])
    # T11: A10 B02 -> C12
    add_term(schoolbook_221_U, schoolbook_221_V, schoolbook_221_W, A_maps[2], B_maps[2], C_maps[5])
    # T12: A11 B12 -> C12
    add_term(schoolbook_221_U, schoolbook_221_V, schoolbook_221_W, A_maps[3], B_maps[5], C_maps[5])

    return {"U": U, "V": V, "W": W}

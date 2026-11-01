import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2
    # A: [A11, A12, A21, A22]
    # B: [B11, B12, B21, B22]
    # C: [C11, C21, C12, C22] (mapped to W rows 0,1,2,3 of the block)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    col = 0
    for k in range(5):
        for a_vec, b_vec, c_vec in strassen_terms:
            # U
            for i in range(4):
                U[i][col] = a_vec[i]
            
            # V
            # B11 -> 2k
            # B12 -> 2k+1
            # B21 -> 10+2k
            # B22 -> 10+2k+1
            V[2*k][col] = b_vec[0]
            V[2*k+1][col] = b_vec[1]
            V[10+2*k][col] = b_vec[2]
            V[10+2*k+1][col] = b_vec[3]
            
            # W
            # C11 -> 4k
            # C21 -> 4k+1
            # C12 -> 4k+2
            # C22 -> 4k+3
            W[4*k][col] = c_vec[0]
            W[4*k+1][col] = c_vec[1]
            W[4*k+2][col] = c_vec[2]
            W[4*k+3][col] = c_vec[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

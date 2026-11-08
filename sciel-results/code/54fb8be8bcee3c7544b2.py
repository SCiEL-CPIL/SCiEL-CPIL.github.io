import numpy as np

DIMENSIONS = [6, 6, 6]

# Strassen 2x2 decomposition
# U: 4x7, V: 4x7, W: 4x7
# Indices:
# U rows: a00, a01, a10, a11
# V rows: b00, b01, b10, b11
# W rows: c00, c01, c10, c11

U_2x2 = [
    [1, 0, 1, 0, 1, 0, 0], # a00
    [0, 0, 0, 1, 1, 0, -1], # a01
    [0, 1, 1, 0, 0, 1, 0], # a10
    [0, 0, 0, 1, 0, 1, 0]  # a11
]

V_2x2 = [
    [1, 1, 0, 0, 0, 0, 0], # b00
    [0, 0, 1, 0, 0, 1, 0], # b01
    [0, 0, 0, 1, 0, 0, 1], # b10
    [0, 0, 0, 0, 1, 1, 0]  # b11
]

W_2x2 = [
    [1, 0, 0, 1, -1, 0, 1], # c00
    [0, 0, 1, 0, 1, 0, 0],  # c01
    [0, 1, 0, 1, 0, 0, 0],  # c10
    [1, -1, 0, 0, 0, 1, 0]  # c11
]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 9 blocks * 7 Strassen ops = 63
    rank = 63
    
    # Initialize matrices
    # U shape: (n*m, rank) = (36, 63)
    # V shape: (m*p, rank) = (36, 63)
    # W shape: (p*n, rank) = (36, 63)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block indices
    # I: row block of A (0,1,2)
    # K: col block of A / row block of B (0,1,2)
    # J: col block of B (0,1,2)
    
    col_offset = 0
    for I in range(3):
        for K in range(3):
            for J in range(3):
                for s in range(7):
                    # Add Strassen coefficient s to this block product
                    # U contribution: A[I, K]
                    # V contribution: B[K, J]
                    # W contribution: C[I, J]
                    
                    # Local indices
                    for i_local in range(2):
                        for k_local in range(2):
                            # U index: row = 2*I + i_local, col = 2*K + k_local
                            u_idx = (2 * I + i_local) * m + (2 * K + k_local)
                            U[u_idx][col_offset] += U_2x2[i_local * 2 + k_local][s]
                            
                    for k_local in range(2):
                        for j_local in range(2):
                            # V index: row = 2*K + k_local, col = 2*J + j_local
                            v_idx = (2 * K + k_local) * p + (2 * J + j_local)
                            V[v_idx][col_offset] += V_2x2[k_local * 2 + j_local][s]
                            
                    for i_local in range(2):
                        for j_local in range(2):
                            # W index: col = 2*J + j_local, row = 2*I + i_local
                            # Task convention: W[k*n + i] where k is col, i is row
                            w_idx = (2 * J + j_local) * n + (2 * I + i_local)
                            W[w_idx][col_offset] += W_2x2[i_local * 2 + j_local][s]
                            
                    col_offset += 1
                    
    return {"U": U, "V": V, "W": W}

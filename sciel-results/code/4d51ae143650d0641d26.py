import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Current best known simple construction is 39.
    # We attempt to find a better one via a simple heuristic or return 39.
    
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, coeffs=None):
        nonlocal col
        if coeffs is None:
            coeffs = [1] * len(u_indices)
        for idx, c in zip(u_indices, coeffs):
            U[idx][col] += c
        # V and W indices might have different coeffs?
        # In the seed, V and W coeffs are handled separately if not 1.
        # Here we assume standard 1s unless specified.
        # The seed program handles -1s separately.
        pass

    # Re-implementing the seed logic for 39 rank
    def add_strassen_block(k_start):
        nonlocal col
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1
        U[0][col] += 1; U[4][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2
        U[3][col] += 1; U[4][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] += -1
        col += 1
        
        # M3
        U[0][col] += 1
        V[b01][col] += 1; V[b11][col] += -1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4
        U[4][col] += 1
        V[b10][col] += 1; V[b00][col] += -1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[b11][col] += 1
        W[c00][col] += -1; W[c01][col] += 1
        col += 1
        
        # M6
        U[3][col] += 1; U[0][col] += -1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1; U[4][col] += -1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # k=6
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    U[0][col] += 1; V[b06][col] += 1; W[c06][col] += 1; col += 1
    U[1][col] += 1; V[b16][col] += 1; W[c06][col] += 1; col += 1
    U[3][col] += 1; V[b06][col] += 1; W[c16][col] += 1; col += 1
    U[4][col] += 1; V[b16][col] += 1; W[c16][col] += 1; col += 1
    
    # j=2
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        U[2][col] += 1; V[b2k][col] += 1; W[c0k][col] += 1; col += 1
        U[5][col] += 1; V[b2k][col] += 1; W[c1k][col] += 1; col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

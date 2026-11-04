import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Blocks of B: 4 columns of 2x2 blocks
    # A: 2 columns of 2x2 blocks
    # We process 4 output blocks of C, each 2x2
    # C_block_j = A_0 * B_0j + A_1 * B_1j
    # Each term is 2x2 * 2x2
    
    for j in range(4):
        # A_0 indices: 0, 1, 4, 5 (row 0 col 0,1; row 1 col 0,1)
        # A_1 indices: 2, 3, 6, 7
        # B_0j indices: row 0 cols 2j, 2j+1; row 1 cols 2j, 2j+1
        # B_1j indices: row 2 cols 2j, 2j+1; row 3 cols 2j, 2j+1
        # C_j indices: row 0 cols 2j, 2j+1; row 1 cols 2j, 2j+1
        
        # Strassen for A_0 * B_0j
        # A_0 = [[a00, a01], [a10, a11]]
        # B_0j = [[b00, b01], [b10, b11]]
        
        a00, a01, a10, a11 = 0, 1, 4, 5
        b00, b01, b10, b11 = 0*8 + 2*j, 0*8 + 2*j + 1, 1*8 + 2*j, 1*8 + 2*j + 1
        c00, c01, c10, c11 = 2*j, 2*j+1, 2*j+2, 2*j+3 # W index: k*n+i -> (2j)*2+0, (2j+1)*2+0, (2j)*2+1, (2j+1)*2+1
        
        # P1 = (a00 + a11) * (b00 + b11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P2 = (a10 + a11) * b00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        # P3 = a00 * (b01 - b11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P4 = a11 * (b10 - b00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # P5 = (a00 + a01) * b11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # P6 = (a10 - a00) * (b00 + b01)
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P7 = (a01 - a11) * (b10 + b11)
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
        # Strassen for A_1 * B_1j
        a00, a01, a10, a11 = 2, 3, 6, 7
        b00, b01, b10, b11 = 2*8 + 2*j, 2*8 + 2*j + 1, 3*8 + 2*j, 3*8 + 2*j + 1
        # C indices are the same
        
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1

    assert col == 56
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

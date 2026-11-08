import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block Strassen with Schoolbook inner blocks
    # Rank 126
    
    # Strassen coefficients for 2x2 blocks
    # t: 1..7
    # A_coeffs[t][alpha][beta]
    # B_coeffs[t][beta][gamma]
    # W_coeffs[t][alpha][gamma] (for reconstruction)
    
    A_coeffs = {}
    B_coeffs = {}
    W_coeffs = {}
    
    # M1 = (A00 + A11)(B00 + B11)
    A_coeffs[1] = {(0,0): 1, (1,1): 1}
    B_coeffs[1] = {(0,0): 1, (1,1): 1}
    W_coeffs[1] = {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}
    
    # M2 = (A10 + A11)B00
    A_coeffs[2] = {(1,0): 1, (1,1): 1}
    B_coeffs[2] = {(0,0): 1}
    W_coeffs[2] = {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}
    
    # M3 = A00(B01 - B11)
    A_coeffs[3] = {(0,0): 1}
    B_coeffs[3] = {(0,1): 1, (1,1): -1}
    W_coeffs[3] = {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}
    
    # M4 = A11(B10 - B00)
    A_coeffs[4] = {(1,1): 1}
    B_coeffs[4] = {(1,0): 1, (0,0): -1}
    W_coeffs[4] = {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}
    
    # M5 = (A00 + A01)B11
    A_coeffs[5] = {(0,0): 1, (0,1): 1}
    B_coeffs[5] = {(1,1): 1}
    W_coeffs[5] = {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): -1}
    
    # M6 = (A10 - A00)(B00 + B01)
    A_coeffs[6] = {(1,0): 1, (0,0): -1}
    B_coeffs[6] = {(0,0): 1, (0,1): 1}
    W_coeffs[6] = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}
    
    # M7 = (A01 - A11)(B10 + B11)
    A_coeffs[7] = {(0,1): 1, (1,1): -1}
    B_coeffs[7] = {(1,0): 1, (1,1): 1}
    W_coeffs[7] = {(0,0): -1, (0,1): 0, (1,0): 0, (1,1): 1}
    
    rank = 0
    U = []
    V = []
    W = []
    
    for t in range(1, 8):
        for alpha in range(2):
            for beta in range(2):
                ca = A_coeffs[t].get((alpha, beta), 0)
                if ca == 0: continue
                for gamma in range(2):
                    cb = B_coeffs[t].get((beta, gamma), 0)
                    if cb == 0: continue
                    cw = W_coeffs[t].get((alpha, gamma), 0)
                    if cw == 0: continue
                    
                    # Inner schoolbook for 3x3
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                # A index: 2*alpha + i, 2*beta + j
                                # B index: 2*beta + j, 2*gamma + k
                                # C index: 2*alpha + i, 2*gamma + k
                                
                                u_row = (2*alpha + i) * 6 + (2*beta + j)
                                v_row = (2*beta + j) * 6 + (2*gamma + k)
                                w_row = (2*gamma + k) * 6 + (2*alpha + i)
                                
                                u_val = ca
                                v_val = cb
                                w_val = cw
                                
                                # Append to matrices
                                U.append([0] * (rank + 1))
                                V.append([0] * (rank + 1))
                                W.append([0] * (rank + 1))
                                
                                # Fill existing rows
                                for r in range(rank):
                                    U[r].append(0)
                                    V[r].append(0)
                                    W[r].append(0)
                                
                                # Fill new column
                                U[u_row] = [0] * (rank + 1) # Reset? No, we need to accumulate.
                                # This approach of appending is inefficient and incorrect for accumulation.
                                # We should pre-allocate or use a dictionary.
                                pass
                                
    # Redo with pre-allocation
    # Max rank estimate: 126
    max_rank = 126
    U = [[0] * max_rank for _ in range(36)]
    V = [[0] * max_rank for _ in range(36)]
    W = [[0] * max_rank for _ in range(36)]
    
    col = 0
    for t in range(1, 8):
        for alpha in range(2):
            for beta in range(2):
                ca = A_coeffs[t].get((alpha, beta), 0)
                if ca == 0: continue
                for gamma in range(2):
                    cb = B_coeffs[t].get((beta, gamma), 0)
                    if cb == 0: continue
                    cw = W_coeffs[t].get((alpha, gamma), 0)
                    if cw == 0: continue
                    
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                u_row = (2*alpha + i) * 6 + (2*beta + j)
                                v_row = (2*beta + j) * 6 + (2*gamma + k)
                                w_row = (2*gamma + k) * 6 + (2*alpha + i)
                                
                                U[u_row][col] += ca
                                V[v_row][col] += cb
                                W[w_row][col] += cw
                                col += 1
                                
    # Trim matrices to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

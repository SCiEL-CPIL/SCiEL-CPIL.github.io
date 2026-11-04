import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0  # safety margin before 150s kill
    
    # Block-Strassen decomposition for 6x6x6 matrix multiplication
    # Split into 3x3 blocks of 2x2 matrices
    # Each 2x2 block multiplication uses Strassen's 7-term algorithm
    # Total rank = 27 * 7 = 189
    
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    
                    c00 = (2*j) * 6 + 2*i
                    c01 = (2*j+1) * 6 + 2*i
                    c10 = (2*j) * 6 + (2*i+1)
                    c11 = (2*j+1) * 6 + (2*i+1)
                    
                    if s == 0:
                        U[a00, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1
                        W[c11, idx] = 1
                    elif s == 1:
                        U[a10, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        W[c10, idx] = 1
                        W[c11, idx] = -1
                    elif s == 2:
                        U[a00, idx] = 1
                        V[b01, idx] = 1
                        V[b11, idx] = -1
                        W[c01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 3:
                        U[a11, idx] = 1
                        V[b10, idx] = 1
                        V[b00, idx] = -1
                        W[c00, idx] = 1
                        W[c10, idx] = 1
                    elif s == 4:
                        U[a00, idx] = 1
                        U[a01, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = -1
                        W[c01, idx] = 1
                    elif s == 5:
                        U[a10, idx] = 1
                        U[a00, idx] = -1
                        V[b00, idx] = 1
                        V[b01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 6:
                        U[a01, idx] = 1
                        U[a11, idx] = -1
                        V[b10, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1
                    
                    idx += 1
    
    assert idx == rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

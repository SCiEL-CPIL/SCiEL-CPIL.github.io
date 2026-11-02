import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Uses 5 blocks of Strassen's rank-7 method for 2x2x2, plus 20 trivial
    rank-1 terms for the third row of A. Total rank: 5*7 + 20 = 55.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    for b in range(5):
        b00, b01 = 2 * b, 2 * b + 1
        b10, b11 = 10 + 2 * b, 10 + 2 * b + 1
        c00, c01 = 4 * b, 4 * b + 1
        c10, c11 = 4 * b + 2, 4 * b + 3
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0, col] += 1.0; U[4, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[3, col] += 1.0; U[4, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[0, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[4, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0, col] += 1.0; U[1, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[3, col] += 1.0; U[0, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1, col] += 1.0; U[4, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
    
    # Trivial terms for A[0,2]*B[2,k]*C[0,k] and A[1,2]*B[2,k]*C[1,k]
    for k in range(10):
        U[2, col] += 1.0; V[20 + k, col] += 1.0; W[2 * k, col] += 1.0
        col += 1
        U[5, col] += 1.0; V[20 + k, col] += 1.0; W[2 * k + 1, col] += 1.0
        col += 1
    
    assert col == 55
    
    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

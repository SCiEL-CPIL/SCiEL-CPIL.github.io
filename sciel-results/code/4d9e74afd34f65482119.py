"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize tensors with zeros
    # U: (n*m, rank) = (6, 17)
    # V: (m*p, rank) = (9, 17)
    # W: (p*n, rank) = (6, 17)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper indices
    # A is 2x3, index (i, j) -> i*m + j
    # B is 3x3, index (j, k) -> j*p + k
    # C is 2x3, index (i, k) -> k*n + i (per problem statement W[k*n+i])
    
    # --- Strassen's Algorithm for A[0:2, 0:2] * B[0:2, 0:2] ---
    # Indices 0-6
    
    # M1 = (A00 + A11) * (B00 + B11)
    # A00: 0*3+0=0, A11: 1*3+1=4
    # B00: 0*3+0=0, B11: 1*3+1=4
    # C00: 0*2+0=0, C11: 1*2+1=4 (Wait, W index is k*n+i. C[i,k]. 
    # C[0,0] -> i=0,k=0 -> idx 0.
    # C[1,1] -> i=1,k=1 -> idx 1*2+1=3? No, W shape is (p*n). 
    # Problem says W[k*n+i]. 
    # C[0,0]: k=0, i=0 -> 0*2+0=0.
    # C[0,1]: k=1, i=0 -> 1*2+0=2.
    # C[0,2]: k=2, i=0 -> 2*2+0=4.
    # C[1,0]: k=0, i=1 -> 0*2+1=1.
    # C[1,1]: k=1, i=1 -> 1*2+1=3.
    # C[1,2]: k=2, i=1 -> 2*2+1=5.
    # Let's re-verify W index mapping.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Yes, k is col of C, i is row of C.
    
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (A10 + A11) * B00
    # A10: 1*3+0=3, A11: 4
    # B00: 0
    # C10: i=1,k=0 -> idx 1.
    # C11: i=1,k=1 -> idx 3.
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C[1,0]
    W[3, 1] = -1 # C[1,1]
    
    # M3 = A00 * (B01 - B11)
    # A00: 0
    # B01: 0*3+1=1, B11: 4
    # C01: i=0,k=1 -> idx 2.
    # C11: i=1,k=1 -> idx 3.
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = A11 * (B10 - B00)
    # A11: 4
    # B10: 1*3+0=3, B00: 0
    # C00: idx 0.
    # C10: idx 1.
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C[0,0]
    W[1, 3] = 1  # C[1,0]
    
    # M5 = (A00 + A01) * B11
    # A00: 0, A01: 0*3+1=1
    # B11: 4
    # C01: idx 2.
    # C00: idx 0.
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1  # C[0,1]
    W[0, 4] = -1 # C[0,0]
    
    # M6 = (A10 - A00) * (B00 + B01)
    # A10: 3, A00: 0
    # B00: 0, B01: 1
    # C11: idx 3.
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A01 - A11) * (B10 + B11)
    # A01: 1, A11: 4
    # B10: 3, B11: 4
    # C00: idx 0.
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    # --- Schoolbook for A[:, 2] * B[2, :] (contribution to C[:, 0:2]) ---
    # Indices 7-10
    
    # Term: A02 * B20 -> C00
    # A02: 0*3+2=2
    # B20: 2*3+0=6
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1  # C[0,0]
    
    # Term: A02 * B21 -> C01
    # B21: 2*3+1=7
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1  # C[0,1]
    
    # Term: A12 * B20 -> C10
    # A12: 1*3+2=5
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1  # C[1,0]
    
    # Term: A12 * B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1  # C[1,1]
    
    # --- Schoolbook for A[:, 0:2] * B[:, 2] (contribution to C[:, 2]) ---
    # Indices 11-14
    
    # Term: A00 * B02 -> C02
    # B02: 0*3+2=2
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1  # C[0,2] (k=2, i=0 -> 4)
    
    # Term: A01 * B12 -> C02
    # B12: 1*3+2=5
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1  # C[0,2]
    
    # Term: A10 * B02 -> C12
    U[3, 13] = 1
    V[2, 13] = 1
    W[5, 13] = 1  # C[1,2] (k=2, i=1 -> 5)
    
    # Term: A11 * B12 -> C12
    U[4, 14] = 1
    V[5, 14] = 1
    W[5, 14] = 1  # C[1,2]
    
    # --- Schoolbook for A[:, 2] * B[2, 2] (contribution to C[:, 2]) ---
    # Indices 15-16
    
    # Term: A02 * B22 -> C02
    # B22: 2*3+2=8
    U[2, 15] = 1
    V[8, 15] = 1
    W[4, 15] = 1  # C[0,2]
    
    # Term: A12 * B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1  # C[1,2]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

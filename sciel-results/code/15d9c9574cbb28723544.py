import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Use the known rank-55 decomposition based on Strassen for 2x2 blocks 
      and schoolbook for the remaining 2x1 block.
    - This is currently the best known simple construction.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    def build_rank_55():
        """Build the standard rank-55 decomposition."""
        rank = 55
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        # A indices: A[i, j] -> U[i*m + j]
        a00, a01, a10, a11 = 0, 1, 3, 4
        a02, a12 = 2, 5
        
        # Process 5 blocks of 2 columns for the Strassen part (A_L * B_L)
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # B indices: B[j, k] -> V[j*p + k]
            b00, b01 = k0, k1
            b10, b11 = 10 + k0, 10 + k1
            
            # C indices: C[i, k] -> W[k*n + i]
            c00, c01 = 2 * k0, 2 * k1
            c10, c11 = 2 * k0 + 1, 2 * k1 + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[a00, col] += 1; U[a11, col] += 1
            V[b00, col] += 1; V[b11, col] += 1
            W[c00, col] += 1; W[c11, col] += 1
            col += 1
            
            # M2 = (A10 + A11)B00
            U[a10, col] += 1; U[a11, col] += 1
            V[b00, col] += 1
            W[c10, col] += 1; W[c11, col] -= 1
            col += 1
            
            # M3 = A00(B01 - B11)
            U[a00, col] += 1
            V[b01, col] += 1; V[b11, col] -= 1
            W[c01, col] += 1; W[c11, col] += 1
            col += 1
            
            # M4 = A11(B10 - B00)
            U[a11, col] += 1
            V[b10, col] += 1; V[b00, col] -= 1
            W[c00, col] += 1; W[c10, col] += 1
            col += 1
            
            # M5 = (A00 + A01)B11
            U[a00, col] += 1; U[a01, col] += 1
            V[b11, col] += 1
            W[c00, col] -= 1; W[c01, col] += 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[a10, col] += 1; U[a00, col] -= 1
            V[b00, col] += 1; V[b01, col] += 1
            W[c11, col] += 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[a01, col] += 1; U[a11, col] -= 1
            V[b10, col] += 1; V[b11, col] += 1
            W[c00, col] += 1
            col += 1
        
        # Schoolbook part for A_R * B_R (2x1 * 1x10)
        for k in range(p):
            b2k = 2 * p + k  # Index for B[2, k]
            
            # Term 1: A02 * B2k -> C0k
            U[a02, col] += 1
            V[b2k, col] += 1
            W[2 * k, col] += 1
            col += 1
            
            # Term 2: A12 * B2k -> C1k
            U[a12, col] += 1
            V[b2k, col] += 1
            W[2 * k + 1, col] += 1
            col += 1
        
        return U, V, W, rank
    
    U55, V55, W55, rank55 = build_rank_55()
    
    return {"U": U55.tolist(), "V": V55.tolist(), "W": W55.tolist()}

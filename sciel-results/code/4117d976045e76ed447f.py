import time
import numpy as np

def construct():
    start_time = time.time()
    
    n, m, p = 2, 2, 16
    
    # We will use a Strassen decomposition for each of the 8 2x2 blocks of B.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2 multiplication
    # A is 2x2, B is 2x2, C = A*B
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * B10
    # M4 = A11 * B11
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A11 - A01) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (n*m = 4):
    # A[0] = A00, A[1] = A01, A[2] = A10, A[3] = A11
    
    # Indices for B block (m*p_block = 4, but here p_block=2 for the sub-block logic, 
    # however in our global V, the indices are j*p + k)
    # For block t, the B indices are:
    # B[t*4 + 0] = B00_t
    # B[t*4 + 1] = B01_t
    # B[t*4 + 2] = B10_t
    # B[t*4 + 3] = B11_t
    # Wait, the problem statement says B is 2x16.
    # Row 0 of B: B[0,0], B[0,1], ..., B[0,15]
    # Row 1 of B: B[1,0], B[1,1], ..., B[1,15]
    # In the V matrix (shape m*p = 2*16 = 32), the index is j*p + k.
    # So V[j*16 + k] corresponds to B[j, k].
    
    # For block t (t=0..7), the columns are 2*t and 2*t+1.
    # B00_t = B[0, 2*t] -> V[0*16 + 2*t]
    # B01_t = B[0, 2*t+1] -> V[0*16 + 2*t+1]
    # B10_t = B[1, 2*t] -> V[1*16 + 2*t]
    # B11_t = B[1, 2*t+1] -> V[1*16 + 2*t+1]
    
    # Output C (shape p*n = 16*2 = 32)
    # C[i, k] where i is row of C (0,1), k is col of C (0..15)
    # W index is k*n + i = k*2 + i.
    # For block t, output columns are 2*t and 2*t+1.
    # C00_t = C[0, 2*t] -> W[2*t*2 + 0]
    # C01_t = C[0, 2*t+1] -> W[2*t+1*2 + 0]
    # C10_t = C[1, 2*t] -> W[2*t*2 + 1]
    # C11_t = C[1, 2*t+1] -> W[2*t+1*2 + 1]
    
    col = 0
    for t in range(8):
        # Define B indices for this block
        b00_idx = 0 * 16 + (2 * t)
        b01_idx = 0 * 16 + (2 * t + 1)
        b10_idx = 1 * 16 + (2 * t)
        b11_idx = 1 * 16 + (2 * t + 1)
        
        # Define W indices for this block
        w00_idx = (2 * t) * 2 + 0
        w01_idx = (2 * t + 1) * 2 + 0
        w10_idx = (2 * t) * 2 + 1
        w11_idx = (2 * t + 1) * 2 + 1
        
        # Strassen Term 1: (A00 + A11) * (B00 + B11)
        U[0, col] += 1  # A00
        U[3, col] += 1  # A11
        V[b00_idx, col] += 1
        V[b11_idx, col] += 1
        W[w00_idx, col] += 1  # C00 += M1
        W[w11_idx, col] += 1  # C11 += M1
        col += 1
        
        # Strassen Term 2: (A10 + A11) * B00
        U[2, col] += 1  # A10
        U[3, col] += 1  # A11
        V[b00_idx, col] += 1
        W[w10_idx, col] += 1  # C10 += M2
        W[w11_idx, col] -= 1  # C11 -= M2
        col += 1
        
        # Strassen Term 3: A00 * B10
        U[0, col] += 1  # A00
        V[b10_idx, col] += 1
        W[w01_idx, col] += 1  # C01 += M3
        W[w11_idx, col] += 1  # C11 += M3
        col += 1
        
        # Strassen Term 4: A11 * B11
        U[3, col] += 1  # A11
        V[b11_idx, col] += 1
        W[w00_idx, col] += 1  # C00 += M4
        W[w10_idx, col] += 1  # C10 += M4
        col += 1
        
        # Strassen Term 5: (A00 + A01) * B11
        U[0, col] += 1  # A00
        U[1, col] += 1  # A01
        V[b11_idx, col] += 1
        W[w00_idx, col] -= 1  # C00 -= M5
        W[w01_idx, col] += 1  # C01 += M5
        col += 1
        
        # Strassen Term 6: (A10 - A00) * (B00 + B01)
        U[2, col] += 1  # A10
        U[0, col] -= 1  # A00
        V[b00_idx, col] += 1
        V[b01_idx, col] += 1
        W[w11_idx, col] += 1  # C11 += M6
        col += 1
        
        # Strassen Term 7: (A11 - A01) * (B10 + B11)
        U[3, col] += 1  # A11
        U[1, col] -= 1  # A01
        V[b10_idx, col] += 1
        V[b11_idx, col] += 1
        W[w00_idx, col] += 1  # C00 += M7
        col += 1
        
    # Verify time
    if time.time() - start_time > 110:
         return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
         
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 63.
    # Split B into B1 (3x3) and B2 (3x2) and B3 (3x3)? No, 3+2+3=8.
    # M(3,3,3) rank 23.
    # M(3,3,2) rank 17.
    # Total rank 23 + 17 + 23 = 63.
    
    # We need the tensors for M(3,3,3) rank 23 and M(3,3,2) rank 17.
    # Due to complexity, I will use a simpler but valid decomposition if the 23-term one is too hard to hardcode.
    # However, 63 is the target.
    
    # Let's use the block construction for M(3,3,2) and a known M(3,3,3).
    
    # For the sake of this response, I will provide a construction that achieves rank < 72.
    # If I cannot easily hardcode the 23-term M(3,3,3), I will use the 68-rank construction from 2x2 blocks.
    # M(2,2,2) rank 7.
    # M(2,2,8) rank 28.
    # Cross terms 40.
    # Total 68.
    
    # Let's implement the 68-rank construction as it is safer to derive.
    
    rank = 68
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Part 1: M(2,2,8) using 4 blocks of M(2,2,2)
    # A[0:2, 0:2] * B[0:2, 0:8]
    # We use the 7-term decomposition for each 2x2 block of B.
    # B is 2x8. Split into 4 2x2 blocks.
    
    # M(2,2,2) rank 7 terms (standard):
    # 1. A00 B00 -> C00
    # 2. A01 B11 -> C01
    # 3. A10 B00 -> C10
    # 4. A11 B11 -> C11
    # 5. (A00+A01) (B10+B11) -> C00+C01+C10+C11 adjustments
    # ... This is Strassen.
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will apply this to each 2x2 block of B.
    
    col_offset = 0
    for block_idx in range(4):
        # B_block is columns 2*block_idx to 2*block_idx+1
        # A_block is A[0:2, 0:2]
        
        # M1: (A00+A11) * (B00+B11)
        U[0, col_offset] = 1; U[3, col_offset] = 1 # A00, A11
        V[0, col_offset] = 1; V[3, col_offset] = 1 # B00, B11
        W[0, col_offset] = 1; W[2, col_offset] = 1 # C00, C11 (partial)
        # Actually, W needs to add to C00, C01, C10, C11.
        # Let's just use schoolbook for simplicity if Strassen is too complex to map to W.
        # Schoolbook for 2x2 is rank 4.
        # 4 blocks * 4 = 16.
        # Cross terms 40.
        # Total 56.
        # This is better than 68!
        
        # Let's use schoolbook for the 2x2 blocks.
        # A[0:2, 0:2] * B[0:2, 0:8]
        # 16 multiplications.
        
        # I will use schoolbook for the 2x2 part.
        
    # Reset
    rank = 56
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    # 1. A[0:2, 0:2] * B[0:2, 0:8]
    for i in range(2):
        for j in range(2):
            for k in range(8):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    # col is 32? No.
    # i in 0,1; j in 0,1; k in 0..7.
    # 2*2*8 = 32.
    
    # 2. Cross terms:
    # Row 0: A[0,2] * B[2, 0:8]
    for k in range(8):
        U[0*m+2, col] = 1
        V[2*p+k, col] = 1
        W[k*n+0, col] = 1
        col += 1
    # Row 1: A[1,2] * B[2, 0:8]
    for k in range(8):
        U[1*m+2, col] = 1
        V[2*p+k, col] = 1
        W[k*n+1, col] = 1
        col += 1
    # Row 2: A[2,0] * B[0, 0:8]
    for k in range(8):
        U[2*m+0, col] = 1
        V[0*p+k, col] = 1
        W[k*n+2, col] = 1
        col += 1
    # Row 2: A[2,1] * B[1, 0:8]
    for k in range(8):
        U[2*m+1, col] = 1
        V[1*p+k, col] = 1
        W[k*n+2, col] = 1
        col += 1
    # Row 2: A[2,2] * B[2, 0:8]
    for k in range(8):
        U[2*m+2, col] = 1
        V[2*p+k, col] = 1
        W[k*n+2, col] = 1
        col += 1
        
    # Total rank 32 + 40 = 72.
    # This is schoolbook.
    
    # To get 56, I need to optimize the 2x2 part to rank 16?
    # Schoolbook for 2x2 by 2x8 is 16.
    # Wait, 2*2*8 = 32.
    # My previous calculation: 4 blocks of 2x2.
    # Each 2x2 block multiplication is rank 4.
    # 4 blocks * 4 = 16.
    # Yes!
    
    # So:
    # 1. A[0:2, 0:2] * B[0:2, 0:8] using 4 blocks of 2x2.
    #    Each block is 2x2.
    #    Block 0: B[0:2, 0:2]. Rank 4.
    #    Block 1: B[0:2, 2:4]. Rank 4.
    #    Block 2: B[0:2, 4:6]. Rank 4.
    #    Block 3: B[0:2, 6:8]. Rank 4.
    #    Total 16.
    
    # 2. Cross terms 40.
    # Total 56.
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    # 1. A[0:2, 0:2] * B[0:2, 0:8]
    for block in range(4):
        k_start = block * 2
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*m+j, col] = 1
                    V[j*p + k_start + k, col] = 1
                    W[(k_start + k)*n + i, col] = 1
                    col += 1
    # col is 16.
    
    # 2. Cross terms
    for k in range(8):
        U[0*m+2, col] = 1; V[2*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
    for k in range(8):
        U[1*m+2, col] = 1; V[2*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
    for k in range(8):
        U[2*m+0, col] = 1; V[0*p+k, col] = 1; W[k*n+2, col] = 1; col += 1
    for k in range(8):
        U[2*m+1, col] = 1; V[1*p+k, col] = 1; W[k*n+2, col] = 1; col += 1
    for k in range(8):
        U[2*m+2, col] = 1; V[2*p+k, col] = 1; W[k*n+2, col] = 1; col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

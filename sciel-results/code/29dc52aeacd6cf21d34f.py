import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Part 2: 4x4x1 Schoolbook (Rank 16)
    rank = 49 + 16
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 4x4x4 ---
    # Inner Strassen for 2x2
    # U_inner: 4x7, V_inner: 4x7, W_inner: 4x7
    # Strassen 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Coefficients for A blocks (00, 01, 10, 11) for each M_r
    # A_factor = sum beta_r_b * A_b
    beta = [
        [1, 0, 0, 1], # M0
        [0, 0, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 0, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 1, 0, -1] # M6
    ]
    
    # Coefficients for B blocks
    gamma = [
        [1, 0, 0, 1], # M0
        [1, 0, 0, 0], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, -1],# M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, 1]  # M6
    ]
    
    # Coefficients for C blocks
    # C00 = M0 - M3 - M5 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 - M4 + M6
    delta = [
        [1, 0, 0, 0], # M0 contributes to C00
        [0, 0, 1, 0], # M1 contributes to C10
        [0, 1, 0, 0], # M2 contributes to C01
        [0, 0, 1, 0], # M3 contributes to C10 (and C00 with -1? Wait)
        # Let's re-verify Strassen C formulas.
        # C00 = M0 - M3 - M5 - M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M2 - M4 + M6
        # So:
        # M0: C00+1, C11+1
        # M1: C10+1
        # M2: C01+1, C11-1
        # M3: C00-1, C10+1
        # M4: C01+1, C11-1
        # M5: C00-1
        # M6: C00-1, C11+1
        
        # My delta array should be delta[r][b] where b is C block index.
        # Let's construct delta properly.
    ]
    
    # Re-construct delta
    delta = [
        [1, 0, 0, 1], # M0: C00+1, C11+1
        [0, 0, 1, 0], # M1: C10+1
        [0, 1, 0, -1],# M2: C01+1, C11-1
        [-1, 0, 1, 0],# M3: C00-1, C10+1
        [0, 1, 0, -1],# M4: C01+1, C11-1
        [-1, 0, 0, 0],# M5: C00-1
        [-1, 0, 0, 1]  # M6: C00-1, C11+1
    ]
    
    # Inner Strassen matrices
    # U_inner[i, s]
    # s=0: (A00+A11) -> A00+1, A11+1
    # s=1: (A10+A11) -> A10+1, A11+1
    # s=2: A00 -> A00+1
    # s=3: A11 -> A11+1
    # s=4: (A00+A01) -> A00+1, A01+1
    # s=5: (A10-A00) -> A10+1, A00-1
    # s=6: (A01-A11) -> A01+1, A11-1
    
    U_inner = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    
    V_inner = np.array([
        [1, 1, 0, 0, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0], # B01
        [0, 0, 0, 1, 0, 0, 1], # B10
        [1, 0, -1, -1, 1, 0, 1]# B11
    ])
    
    W_inner = np.array([
        [1, 0, 0, -1, 0, -1, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, 0, -1, 0, -1, 0, 1]   # C11
    ])
    
    for r in range(7):
        for s in range(7):
            c = r * 7 + s
            
            # U
            # U[:, c] is formed by beta[r][b] * U_inner[:, s] for each block b
            # Block 00 is A[0:2, 0:2] -> indices 0..3
            # Block 01 is A[0:2, 2:4] -> indices 4..7
            # Block 10 is A[2:4, 0:2] -> indices 8..11
            # Block 11 is A[2:4, 2:4] -> indices 12..15
            
            for b in range(4):
                if beta[r][b] != 0:
                    start = b * 4
                    U[start:start+4, c] += beta[r][b] * U_inner[:, s]
            
            # V
            # V[:, c] for B blocks (first 4 columns of B)
            # Block 00 is B[0:2, 0:2] -> indices 0..3 (j*p+k, p=5)
            # Wait, V index is j*p + k.
            # B is 4x5.
            # Block 00: j=0,1; k=0,1. Indices: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6.
            # Block 01: j=0,1; k=2,3. Indices: 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8.
            # Block 10: j=2,3; k=0,1. Indices: 2*5+0=10, 2*5+1=11, 3*5+0=15, 3*5+1=16.
            # Block 11: j=2,3; k=2,3. Indices: 2*5+2=12, 2*5+3=13, 3*5+2=17, 3*5+3=18.
            
            # Map V_inner indices (0..3) to V global indices for each block.
            # V_inner[0] -> B00 entry (0,0)
            # V_inner[1] -> B01 entry (0,1)
            # V_inner[2] -> B10 entry (1,0)
            # V_inner[3] -> B11 entry (1,1)
            
            # Block 00 (B[0:2, 0:2]):
            # Local 0 -> Global 0
            # Local 1 -> Global 1
            # Local 2 -> Global 5
            # Local 3 -> Global 6
            # Block 01 (B[0:2, 2:4]):
            # Local 0 -> Global 2
            # Local 1 -> Global 3
            # Local 2 -> Global 7
            # Local 3 -> Global 8
            # Block 10 (B[2:4, 0:2]):
            # Local 0 -> Global 10
            # Local 1 -> Global 11
            # Local 2 -> Global 15
            # Local 3 -> Global 16
            # Block 11 (B[2:4, 2:4]):
            # Local 0 -> Global 12
            # Local 1 -> Global 13
            # Local 2 -> Global 17
            # Local 3 -> Global 18
            
            v_map = [
                [0, 1, 5, 6],
                [2, 3, 7, 8],
                [10, 11, 15, 16],
                [12, 13, 17, 18]
            ]
            
            for b in range(4):
                if gamma[r][b] != 0:
                    for local_idx in range(4):
                        global_idx = v_map[b][local_idx]
                        V[global_idx, c] += gamma[r][b] * V_inner[local_idx, s]
                        
            # W
            # W index k*n + i.
            # C blocks for first 4 columns.
            # Block 00: k=0,1; i=0,1. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
            # Block 01: k=0,1; i=2,3. Indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
            # Block 10: k=2,3; i=0,1. Indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13.
            # Block 11: k=2,3; i=2,3. Indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15.
            
            w_map = [
                [0, 1, 4, 5],
                [2, 3, 6, 7],
                [8, 9, 12, 13],
                [10, 11, 14, 15]
            ]
            
            for b in range(4):
                if delta[r][b] != 0:
                    for local_idx in range(4):
                        global_idx = w_map[b][local_idx]
                        W[global_idx, c] += delta[r][b] * W_inner[local_idx, s]

    # --- Part 2: 4x4x1 (Rank 16) ---
    # For i in 0..3, j in 0..3:
    # C[i, 4] += A[i, j] B[j, 4]
    # U[i*4+j, 49 + i*4+j] = 1
    # V[j*5+4, 49 + i*4+j] = 1
    # W[4*4+i, 49 + i*4+j] = 1
    
    for i in range(4):
        for j in range(4):
            c = 49 + i * 4 + j
            U[i * 4 + j, c] = 1
            V[j * 5 + 4, c] = 1
            W[4 * 4 + i, c] = 1
            
    return {"U": U, "V": V, "W": W}

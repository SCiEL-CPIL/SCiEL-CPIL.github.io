import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen decomposition for 2x2 blocks.
    # Split m=4 into two groups of 2: J1={0,1}, J2={2,3}
    # Split p=6 into three groups of 2: K1={0,1}, K2={2,3}, K3={4,5}
    
    # Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B01)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U factors (A indices)
    # A is 2x2 in local coords:
    # 00, 01
    # 10, 11
    # Global U indices for A (n=2, m=4):
    # A[i, j] -> i*m + j
    # For J1 (j=0,1):
    # A00 -> 0*4+0 = 0
    # A01 -> 0*4+1 = 1
    # A10 -> 1*4+0 = 4
    # A11 -> 1*4+1 = 5
    # For J2 (j=2,3):
    # A00 -> 0*4+2 = 2
    # A01 -> 0*4+3 = 3
    # A10 -> 1*4+2 = 6
    # A11 -> 1*4+3 = 7
    
    # V factors (B indices)
    # B is 2x2 in local coords:
    # 00, 01
    # 10, 11
    # Global V indices for B (m=4, p=6):
    # B[j, k] -> j*p + k
    # For K1 (k=0,1) and J1 (j=0,1):
    # B00 -> 0*6+0 = 0
    # B01 -> 0*6+1 = 1
    # B10 -> 1*6+0 = 6
    # B11 -> 1*6+1 = 7
    # For K2 (k=2,3) and J1 (j=0,1):
    # B00 -> 0*6+2 = 2
    # B01 -> 0*6+3 = 3
    # B10 -> 1*6+2 = 8
    # B11 -> 1*6+3 = 9
    # For K3 (k=4,5) and J1 (j=0,1):
    # B00 -> 0*6+4 = 4
    # B01 -> 0*6+5 = 5
    # B10 -> 1*6+4 = 10
    # B11 -> 1*6+5 = 11
    
    # Similarly for J2 (j=2,3):
    # K1: B20=12, B21=13, B30=18, B31=19
    # K2: B22=14, B23=15, B32=20, B33=21
    # K3: B24=16, B25=17, B34=22, B35=23
    
    # W factors (C indices)
    # C is 2x2 in local coords:
    # 00, 01
    # 10, 11
    # Global W indices for C (p=6, n=2):
    # C[k, i] -> k*n + i
    # For K1 (k=0,1):
    # C00 -> 0*2+0 = 0
    # C01 -> 0*2+1 = 1
    # C10 -> 1*2+0 = 2
    # C11 -> 1*2+1 = 3
    # For K2 (k=2,3):
    # C00 -> 2*2+0 = 4
    # C01 -> 2*2+1 = 5
    # C10 -> 3*2+0 = 6
    # C11 -> 3*2+1 = 7
    # For K3 (k=4,5):
    # C00 -> 4*2+0 = 8
    # C01 -> 4*2+1 = 9
    # C10 -> 5*2+0 = 10
    # C11 -> 5*2+1 = 11

    # We will create 6 blocks of 2x2x2 multiplication.
    # Block 1: A(J1) * B(J1, K1) -> C(K1)
    # Block 2: A(J1) * B(J1, K2) -> C(K2)
    # Block 3: A(J1) * B(J1, K3) -> C(K3)
    # Block 4: A(J2) * B(J2, K1) -> C(K1)
    # Block 5: A(J2) * B(J2, K2) -> C(K2)
    # Block 6: A(J2) * B(J2, K3) -> C(K3)
    
    # Each block has rank 7. Total rank 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients
    # M1: A00, A11, B00, B11
    # M2: A10, A11, B00
    # M3: A00, B11, B01
    # M4: A11, B10, B00
    # M5: A00, A01, B11
    # M6: A10, A00, B00, B01
    # M7: A01, A11, B10, B11
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [1, 0, 1, 0, 1, 1, 0], # A00: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1], # A01: M5, M7
        [0, 1, 0, 0, 0, 1, 0], # A10: M2, M6
        [1, 1, 0, 1, 0, 0, -1] # A11: M1, M2, M4, M7
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0], # B00: M1, M2, M4, M6
        [0, 0, -1, 0, 0, 1, 0], # B01: M3, M6
        [0, 0, 0, 1, 0, 0, 1], # B10: M4, M7
        [1, 0, 1, 0, 1, 0, 0]  # B11: M1, M3, M5
    ]
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1], # C00: M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0], # C01: M3, M5
        [0, 1, 0, 1, 0, 0, 0], # C10: M2, M4
        [1, -1, 1, 0, 0, 1, 0]  # C11: M1, M2, M3, M6
    ]
    
    # Map local indices to global indices
    # A indices: 0,1,4,5 for J1; 2,3,6,7 for J2
    # B indices: 0,1,6,7 for J1K1; 2,3,8,9 for J1K2; 4,5,10,11 for J1K3
    #           12,13,18,19 for J2K1; 14,15,20,21 for J2K2; 16,17,22,23 for J2K3
    # C indices: 0,1,2,3 for K1; 4,5,6,7 for K2; 8,9,10,11 for K3
    
    # Block definitions: (A_offset, B_offset, C_offset)
    # A_offset: 0 for J1, 2 for J2 (shift in columns)
    # B_offset: 
    #   J1K1: 0
    #   J1K2: 2
    #   J1K3: 4
    #   J2K1: 12
    #   J2K2: 14
    #   J2K3: 16
    # C_offset:
    #   K1: 0
    #   K2: 4
    #   K3: 8
    
    blocks = [
        (0, 0, 0),   # J1, K1
        (0, 2, 4),   # J1, K2
        (0, 4, 8),   # J1, K3
        (2, 12, 0),  # J2, K1
        (2, 14, 4),  # J2, K2
        (2, 16, 8)   # J2, K3
    ]
    
    for b_idx, (a_off, b_off, c_off) in enumerate(blocks):
        for m_idx in range(7):
            col = b_idx * 7 + m_idx
            
            # Update U
            for i in range(4):
                val = strassen_U[i][m_idx]
                if val != 0:
                    # Local A indices: 0,1,4,5
                    # Global A index = local + a_off
                    # Wait, local indices are 0,1,4,5 relative to 0?
                    # No, my strassen_U is indexed 0..3 corresponding to A00, A01, A10, A11.
                    # Global indices for J1: 0, 1, 4, 5.
                    # Global indices for J2: 2, 3, 6, 7.
                    # So global = local_global_map[i] + a_off?
                    # J1 map: [0, 1, 4, 5]
                    # J2 map: [2, 3, 6, 7]
                    # Actually, if a_off=0, indices are 0,1,4,5.
                    # If a_off=2, indices are 2,3,6,7.
                    # So global = base_indices[i] + a_off.
                    pass
            
            # Let's hardcode the base indices for clarity
            if a_off == 0:
                a_indices = [0, 1, 4, 5]
            else:
                a_indices = [2, 3, 6, 7]
                
            if b_off == 0:
                b_indices = [0, 1, 6, 7]
            elif b_off == 2:
                b_indices = [2, 3, 8, 9]
            elif b_off == 4:
                b_indices = [4, 5, 10, 11]
            elif b_off == 12:
                b_indices = [12, 13, 18, 19]
            elif b_off == 14:
                b_indices = [14, 15, 20, 21]
            elif b_off == 16:
                b_indices = [16, 17, 22, 23]
                
            if c_off == 0:
                c_indices = [0, 1, 2, 3]
            elif c_off == 4:
                c_indices = [4, 5, 6, 7]
            elif c_off == 8:
                c_indices = [8, 9, 10, 11]
                
            for i in range(4):
                val = strassen_U[i][m_idx]
                if val != 0:
                    U[a_indices[i]][col] = val
                    
                val = strassen_V[i][m_idx]
                if val != 0:
                    V[b_indices[i]][col] = val
                    
                val = strassen_W[i][m_idx]
                if val != 0:
                    W[c_indices[i]][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

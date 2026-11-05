import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block structure: 3x3 blocks of 2x2 matrices.
    # The 3x3 block multiplication is schoolbook (rank 27).
    # The 2x2 block multiplication is Strassen (rank 7).
    # Total rank = 27 * 7 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Indices for 2x2:
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # We define the 7 rank-1 tensors for Strassen
    # Each is a tuple (U2, V2, W2) where U2, V2, W2 are dicts mapping (i,j) to coeff
    # Or just lists of length 4.
    
    # Let's define the 7 multiplications M1..M7
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will generate the U, V, W vectors for the 2x2 case (length 4 each)
    # Then combine with the 3x3 schoolbook case.
    
    # Schoolbook 3x3:
    # For each i3, j3, k3 in 0..2:
    # U3 has 1 at (i3, j3)
    # V3 has 1 at (j3, k3)
    # W3 has 1 at (k3, i3)  (Note: W index convention in prompt is k*n+i)
    
    # Strassen 2x2:
    # We have 7 terms.
    # For each term s in 0..6:
    # U2_s is a vector of length 4 (for A00, A01, A10, A11)
    # V2_s is a vector of length 4 (for B00, B01, B10, B11)
    # W2_s is a vector of length 4 (for C00, C01, C10, C11)
    
    # Let's define U2, V2, W2 for the 7 terms.
    # Index mapping for 2x2:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    U2 = []
    V2 = []
    W2 = []
    
    # Term 1: M1
    # U: A00 + A11 -> [1, 0, 0, 1]
    # V: B00 + B11 -> [1, 0, 0, 1]
    # W: C00 + C11 -> [1, 0, 0, 1]
    U2.append([1, 0, 0, 1])
    V2.append([1, 0, 0, 1])
    W2.append([1, 0, 0, 1])
    
    # Term 2: M2
    # U: A10 + A11 -> [0, 0, 1, 1]
    # V: B00 -> [1, 0, 0, 0]
    # W: C10 -> [0, 0, 1, 0]
    U2.append([0, 0, 1, 1])
    V2.append([1, 0, 0, 0])
    W2.append([0, 0, 1, 0])
    
    # Term 3: M3
    # U: A00 -> [1, 0, 0, 0]
    # V: B01 - B11 -> [0, 1, 0, -1]
    # W: C01 -> [0, 1, 0, 0]
    U2.append([1, 0, 0, 0])
    V2.append([0, 1, 0, -1])
    W2.append([0, 1, 0, 0])
    
    # Term 4: M4
    # U: A11 -> [0, 0, 0, 1]
    # V: B10 - B00 -> [-1, 0, 1, 0]
    # W: C00 + C10 -> [1, 0, 1, 0]
    U2.append([0, 0, 0, 1])
    V2.append([-1, 0, 1, 0])
    W2.append([1, 0, 1, 0])
    
    # Term 5: M5
    # U: A00 + A01 -> [1, 1, 0, 0]
    # V: B11 -> [0, 0, 0, 1]
    # W: -C00 + C01 -> [-1, 1, 0, 0]
    U2.append([1, 1, 0, 0])
    V2.append([0, 0, 0, 1])
    W2.append([-1, 1, 0, 0])
    
    # Term 6: M6
    # U: A10 - A00 -> [-1, 0, 1, 0]
    # V: B00 + B01 -> [1, 1, 0, 0]
    # W: C11 -> [0, 0, 0, 1]
    U2.append([-1, 0, 1, 0])
    V2.append([1, 1, 0, 0])
    W2.append([0, 0, 0, 1])
    
    # Term 7: M7
    # U: A01 - A11 -> [0, 1, 0, -1]
    # V: B10 + B11 -> [0, 0, 1, 1]
    # W: C00 -> [1, 0, 0, 0]
    U2.append([0, 1, 0, -1])
    V2.append([0, 0, 1, 1])
    W2.append([1, 0, 0, 0])
    
    # Now combine with 3x3 schoolbook
    # Loop over 3x3 indices i3, j3, k3
    col = 0
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                # For each of the 7 Strassen terms
                for s in range(7):
                    # U index in 6x6:
                    # i = i3 * 2 + i2
                    # j = j3 * 2 + j2
                    # Index = i * 6 + j
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i3 * 2 + i2
                            j = j3 * 2 + j2
                            idx_u = i * 6 + j
                            val = 1 * U2[s][i2 * 2 + j2] # U3 is 1 at (i3, j3)
                            if val != 0:
                                U[idx_u][col] += val
                            
                            # V index in 6x6:
                            # j = j3 * 2 + j2
                            # k = k3 * 2 + k2
                            # Index = j * 6 + k
                            idx_v = j * 6 + (k3 * 2 + k2)
                            val_v = 1 * V2[s][j2 * 2 + k2] # V3 is 1 at (j3, k3)
                            if val_v != 0:
                                V[idx_v][col] += val_v
                            
                            # W index in 6x6:
                            # k = k3 * 2 + k2
                            # i = i3 * 2 + i2
                            # Index = k * 6 + i (per prompt convention W[k*n+i])
                            idx_w = (k3 * 2 + k2) * 6 + i
                            val_w = 1 * W2[s][k2 * 2 + i2] # W3 is 1 at (k3, i3)
                            if val_w != 0:
                                W[idx_w][col] += val_w
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

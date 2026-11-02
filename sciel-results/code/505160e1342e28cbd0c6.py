import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use rank 42 via a structured decomposition for 2x2x12
    # The trivial rank is 48. We use a known decomposition pattern.
    rank = 42
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For 2x2 matrix multiplication, Strassen's algorithm uses 7 multiplications.
    # For 2x2x12, we can decompose the 12 columns into 6 blocks of 2 columns each.
    # For each block of 2 columns, we use 7 multiplications (Strassen).
    # Total rank = 6 * 7 = 42.
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    
    # For each block b (0..5) of 2 columns (k = 2b, 2b+1):
    # The B matrix block is B[:, 2b:2b+2] which is 2x2.
    # We use 7 Strassen products.
    
    c = 0
    for b in range(6):
        # For block b, the B submatrix indices are j*p + (2b) and j*p + (2b+1)
        # Strassen product 1: (A00 + A11)(B00 + B11)
        # U indices: 0 (A00), 3 (A11)
        # V indices: 0*p+2b (B00 of block), 1*p+2b (B10 of block)... wait
        
        # Let me be more careful with the indexing.
        # A is 2x2: A[i,j] where i=0,1 (rows), j=0,1 (cols)
        # B is 2x12: B[j,k] where j=0,1 (rows), k=0..11 (cols)
        # C is 2x12: C[i,k] where i=0,1 (rows), k=0..11 (cols)
        
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # For block b, k = 2b and k = 2b+1
        # B[j, 2b] and B[j, 2b+1] for j=0,1
        
        # Strassen for block b:
        # Let B0 = B[:, 2b:2b+2] (2x2 matrix)
        # B0[0,0] = B[0, 2b], B0[0,1] = B[0, 2b+1]
        # B0[1,0] = B[1, 2b], B0[1,1] = B[1, 2b+1]
        
        # Strassen products:
        # M1 = (A00 + A11) * (B0[0,0] + B0[1,1]) = (A[0,0] + A[1,1]) * (B[0,2b] + B[1,2b+1])
        # M2 = (A10 + A11) * B0[0,0] = (A[1,0] + A[1,1]) * B[0,2b]
        # M3 = A00 * (B0[0,1] - B0[1,1]) = A[0,0] * (B[0,2b+1] - B[1,2b+1])
        # M4 = A11 * (B0[1,0] - B0[0,0]) = A[1,1] * (B[1,2b] - B[0,2b])
        # M5 = (A00 + A01) * B0[1,1] = (A[0,0] + A[0,1]) * B[1,2b+1]
        # M6 = (A10 - A00) * (B0[0,0] + B0[1,0]) = (A[1,0] - A[0,0]) * (B[0,2b] + B[1,2b])
        # M7 = (A01 - A11) * (B0[0,1] + B0[1,1]) = (A[0,1] - A[1,1]) * (B[0,2b+1] + B[1,2b+1])
        
        # C[0, 2b] = M1 - M3 + M4 + M6
        # C[1, 2b] = M2 + M3 - M4 + M7  -- wait, let me recheck Strassen
        
        # Standard Strassen:
        # C00 = M1 - M3 + M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M5 + M7
        
        # So for C[0, 2b]: M1 - M3 + M4 + M6
        # For C[0, 2b+1]: M3 + M5
        # For C[1, 2b]: M2 + M4
        # For C[1, 2b+1]: M1 - M2 + M5 + M7
        
        # U matrix (4 x 42): rows are (i*m + j) = (i*2 + j) for i=0,1; j=0,1
        # V matrix (24 x 42): rows are (j*p + k) = (j*12 + k) for j=0,1; k=0..11
        # W matrix (24 x 42): rows are (k*n + i) = (k*2 + i) for k=0..11; i=0,1
        
        for s in range(7):
            col = c
            
            if s == 0:  # M1 = (A00 + A11) * (B[0,2b] + B[1,2b+1])
                U[0, col] = 1   # A00
                U[3, col] = 1   # A11
                V[0 * p + 2*b, col] = 1      # B[0, 2b]
                V[1 * p + (2*b+1), col] = 1  # B[1, 2b+1]
                # C[0, 2b] += M1: W[(2b)*2 + 0] += 1
                # C[1, 2b+1] += M1: W[(2b+1)*2 + 1] += 1
                W[(2*b) * n + 0, col] = 1
                W[(2*b+1) * n + 1, col] = 1
                
            elif s == 1:  # M2 = (A10 + A11) * B[0,2b]
                U[2, col] = 1   # A10
                U[3, col] = 1   # A11
                V[0 * p + 2*b, col] = 1      # B[0, 2b]
                # C[1, 2b] += M2: W[(2b)*2 + 1] += 1
                # C[1, 2b+1] -= M2: W[(2b+1)*2 + 1] -= 1
                W[(2*b) * n + 1, col] = 1
                W[(2*b+1) * n + 1, col] = -1
                
            elif s == 2:  # M3 = A00 * (B[0,2b+1] - B[1,2b+1])
                U[0, col] = 1   # A00
                V[0 * p + (2*b+1), col] = 1   # B[0, 2b+1]
                V[1 * p + (2*b+1), col] = -1  # B[1, 2b+1]
                # C[0, 2b] -= M3: W[(2b)*2 + 0] -= 1
                # C[0, 2b+1] += M3: W[(2b+1)*2 + 0] += 1
                W[(2*b) * n + 0, col] = -1
                W[(2*b+1) * n + 0, col] = 1
                
            elif s == 3:  # M4 = A11 * (B[1,2b] - B[0,2b])
                U[3, col] = 1   # A11
                V[1 * p + 2*b, col] = 1      # B[1, 2b]
                V[0 * p + 2*b, col] = -1     # B[0, 2b]
                # C[0, 2b] += M4: W[(2b)*2 + 0] += 1
                # C[1, 2b] += M4: W[(2b)*2 + 1] += 1
                W[(2*b) * n + 0, col] = 1
                W[(2*b) * n + 1, col] = 1
                
            elif s == 4:  # M5 = (A00 + A01) * B[1,2b+1]
                U[0, col] = 1   # A00
                U[1, col] = 1   # A01
                V[1 * p + (2*b+1), col] = 1  # B[1, 2b+1]
                # C[0, 2b+1] += M5: W[(2b+1)*2 + 0] += 1
                # C[1, 2b+1] += M5: W[(2b+1)*2 + 1] += 1
                W[(2*b+1) * n + 0, col] = 1
                W[(2*b+1) * n + 1, col] = 1
                
            elif s == 5:  # M6 = (A10 - A00) * (B[0,2b] + B[1,2b])
                U[2, col] = 1   # A10
                U[0, col] = -1  # A00
                V[0 * p + 2*b, col] = 1      # B[0, 2b]
                V[1 * p + 2*b, col] = 1      # B[1, 2b]
                # C[0, 2b] += M6: W[(2b)*2 + 0] += 1
                W[(2*b) * n + 0, col] = 1
                
            elif s == 6:  # M7 = (A01 - A11) * (B[0,2b+1] + B[1,2b+1])
                U[1, col] = 1   # A01
                U[3, col] = -1  # A11
                V[0 * p + (2*b+1), col] = 1   # B[0, 2b+1]
                V[1 * p + (2*b+1), col] = 1   # B[1, 2b+1]
                # C[1, 2b+1] += M7: W[(2b+1)*2 + 1] += 1
                W[(2*b+1) * n + 1, col] = 1
            
            c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # Rank is 6 * 7 = 42
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices for U (A matrix):
    # u[0] = A00, u[1] = A01, u[2] = A10, u[3] = A11
    
    # Indices for V (B matrix):
    # v[j*p + k]
    
    # Indices for W (C matrix):
    # w[k*n + i]
    
    # Strassen definitions for 2x2 block
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # For each of the 6 blocks (l=0..5)
    for l in range(6):
        # Column indices for this block's 7 products
        base_col = l * 7
        
        # B block indices
        # B00 -> v[0*12 + 2*l] = v[2*l]
        # B01 -> v[0*12 + 2*l + 1] = v[2*l + 1]
        # B10 -> v[1*12 + 2*l] = v[12 + 2*l]
        # B11 -> v[1*12 + 2*l + 1] = v[12 + 2*l + 1]
        
        b00 = 2*l
        b01 = 2*l + 1
        b10 = 12 + 2*l
        b11 = 12 + 2*l + 1
        
        # C block indices
        # C00 -> w[2*l*2 + 0] = w[4*l]
        # C01 -> w[(2*l+1)*2 + 0] = w[4*l + 2]
        # C10 -> w[2*l*2 + 1] = w[4*l + 1]
        # C11 -> w[(2*l+1)*2 + 1] = w[4*l + 3]
        
        c00 = 4*l
        c01 = 4*l + 2
        c10 = 4*l + 1
        c11 = 4*l + 3
        
        # Product 1: (A00+A11)(B00+B11)
        col = base_col + 0
        U[0, col] += 1 # A00
        U[3, col] += 1 # A11
        V[b00, col] += 1 # B00
        V[b11, col] += 1 # B11
        W[c00, col] += 1
        W[c11, col] += 1
        
        # Product 2: (A10+A11)B00
        col = base_col + 1
        U[2, col] += 1 # A10
        U[3, col] += 1 # A11
        V[b00, col] += 1 # B00
        W[c10, col] += 1
        W[c11, col] -= 1
        
        # Product 3: A00(B01-B11)
        col = base_col + 2
        U[0, col] += 1 # A00
        V[b01, col] += 1 # B01
        V[b11, col] -= 1 # B11
        W[c01, col] += 1
        W[c11, col] += 1
        
        # Product 4: A11(B10-B00)
        col = base_col + 3
        U[3, col] += 1 # A11
        V[b10, col] += 1 # B10
        V[b00, col] -= 1 # B00
        W[c00, col] += 1
        W[c10, col] += 1
        
        # Product 5: (A00+A01)B11
        col = base_col + 4
        U[0, col] += 1 # A00
        U[1, col] += 1 # A01
        V[b11, col] += 1 # B11
        W[c00, col] -= 1
        W[c01, col] += 1
        
        # Product 6: (A10-A00)(B00+B01)
        col = base_col + 5
        U[2, col] += 1 # A10
        U[0, col] -= 1 # A00
        V[b00, col] += 1 # B00
        V[b01, col] += 1 # B01
        W[c11, col] += 1
        
        # Product 7: (A01-A11)(B10+B11)
        col = base_col + 6
        U[1, col] += 1 # A01
        U[3, col] -= 1 # A11
        V[b10, col] += 1 # B10
        V[b11, col] += 1 # B11
        W[c00, col] += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

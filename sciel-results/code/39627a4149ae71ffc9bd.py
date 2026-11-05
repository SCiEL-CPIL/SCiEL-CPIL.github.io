import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a 3x3 block decomposition where each block is 2x2.
    # Each 2x2 matrix multiplication has rank 7 (Strassen).
    # Total rank = 9 blocks * 7 = 63? No.
    # The rank of the tensor M_{6x6x6} is the number of scalar multiplications.
    # In the block structure, we perform 27 block multiplications (3x3 grid).
    # Each block multiplication is a 2x2 matrix product, which takes 7 scalar multiplications.
    # Total rank = 27 * 7 = 189.
    
    rank = 189
    
    # Initialize U, V, W as numpy arrays for efficient construction
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Block indices: i (row block), s (col block of A / row block of B), j (col block of B)
    # i, s, j range from 0 to 2 (3 blocks each dimension)
    for i in range(3):
        for s in range(3):
            for j in range(3):
                # 7 Strassen products for this block multiplication
                for k in range(7):
                    # Determine global indices for A entries (a,b,c,d) in the 2x2 block at (i,s)
                    # Block (i,s) covers rows 2i, 2i+1 and cols 2s, 2s+1
                    # a: (2i, 2s), b: (2i, 2s+1), c: (2i+1, 2s), d: (2i+1, 2s+1)
                    idx_a = (2 * i) * 6 + (2 * s)
                    idx_b = (2 * i) * 6 + (2 * s + 1)
                    idx_c = (2 * i + 1) * 6 + (2 * s)
                    idx_d = (2 * i + 1) * 6 + (2 * s + 1)
                    
                    # Determine global indices for B entries (e,f,g,h) in the 2x2 block at (s,j)
                    # Block (s,j) covers rows 2s, 2s+1 and cols 2j, 2j+1
                    # e: (2s, 2j), f: (2s, 2j+1), g: (2s+1, 2j), h: (2s+1, 2j+1)
                    idx_e = (2 * s) * 6 + (2 * j)
                    idx_f = (2 * s) * 6 + (2 * j + 1)
                    idx_g = (2 * s + 1) * 6 + (2 * j)
                    idx_h = (2 * s + 1) * 6 + (2 * j + 1)
                    
                    # Determine global indices for C entries (c00, c01, c10, c11) in the 2x2 block at (i,j)
                    # W index is col * n + row. 
                    # C entry (row, col) -> W index col * 6 + row.
                    # Block (i,j) covers rows 2i, 2i+1 and cols 2j, 2j+1
                    # c00: row=2i, col=2j -> W idx: 2j * 6 + 2i
                    # c01: row=2i, col=2j+1 -> W idx: (2j+1) * 6 + 2i
                    # c10: row=2i+1, col=2j -> W idx: 2j * 6 + (2i+1)
                    # c11: row=2i+1, col=2j+1 -> W idx: (2j+1) * 6 + (2i+1)
                    idx_w00 = (2 * j) * 6 + (2 * i)
                    idx_w01 = (2 * j + 1) * 6 + (2 * i)
                    idx_w10 = (2 * j) * 6 + (2 * i + 1)
                    idx_w11 = (2 * j + 1) * 6 + (2 * i + 1)
                    
                    # Set U coefficients
                    if k == 0: # M1 = (a+d)(e+h)
                        U[idx_a, col] = 1
                        U[idx_d, col] = 1
                    elif k == 1: # M2 = (c+d)e
                        U[idx_c, col] = 1
                        U[idx_d, col] = 1
                    elif k == 2: # M3 = a(f-h)
                        U[idx_a, col] = 1
                    elif k == 3: # M4 = d(g-e)
                        U[idx_d, col] = 1
                    elif k == 4: # M5 = (a+b)h
                        U[idx_a, col] = 1
                        U[idx_b, col] = 1
                    elif k == 5: # M6 = (c-a)(e+f)
                        U[idx_c, col] = 1
                        U[idx_a, col] = -1
                    elif k == 6: # M7 = (b-d)(g+h)
                        U[idx_b, col] = 1
                        U[idx_d, col] = -1
                        
                    # Set V coefficients
                    if k == 0: # M1
                        V[idx_e, col] = 1
                        V[idx_h, col] = 1
                    elif k == 1: # M2
                        V[idx_e, col] = 1
                    elif k == 2: # M3
                        V[idx_f, col] = 1
                        V[idx_h, col] = -1
                    elif k == 3: # M4
                        V[idx_g, col] = 1
                        V[idx_e, col] = -1
                    elif k == 4: # M5
                        V[idx_h, col] = 1
                    elif k == 5: # M6
                        V[idx_e, col] = 1
                        V[idx_f, col] = 1
                    elif k == 6: # M7
                        V[idx_g, col] = 1
                        V[idx_h, col] = 1
                        
                    # Set W coefficients
                    if k == 0: # M1
                        W[idx_w00, col] = 1
                        W[idx_w11, col] = 1
                    elif k == 1: # M2
                        W[idx_w10, col] = 1
                        W[idx_w11, col] = -1
                    elif k == 2: # M3
                        W[idx_w01, col] = 1
                        W[idx_w11, col] = 1
                    elif k == 3: # M4
                        W[idx_w00, col] = 1
                        W[idx_w10, col] = 1
                    elif k == 4: # M5
                        W[idx_w00, col] = -1
                        W[idx_w01, col] = 1
                    elif k == 5: # M6
                        W[idx_w11, col] = 1
                    elif k == 6: # M7
                        W[idx_w00, col] = 1
                        
                    col += 1
                    
    # Convert back to lists if required, but numpy arrays are usually accepted
    # The task says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays is standard for such tasks.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

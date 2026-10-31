"""Matrix multiplication 4x4x4 rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 exact decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (rank 7 for 2x2x2 applied twice).
    
    Strategy: Represent 4x4 multiplication as 2x2x2 tensor over 2x2 matrices.
    Apply Strassen at both block level and element level.
    Total rank: 7 * 7 = 49.
    
    Index conventions:
    - U[i*4+j, r]: A entry (i,j)
    - V[j*4+k, r]: B entry (j,k)  
    - W[k*4+i, r]: C entry (i,k) with trace convention W[k*n+i]
    
    Block structure:
    - A[br*2+r, bc*2+c] for br, bc, r, c in {0, 1}
    - U index: (br*2+r)*4 + (bc*2+c)
    - V index: (br*2+r)*4 + (bc*2+c) 
    - W index: (bc*2+c)*4 + (br*2+r)
    """
    n, m, p = DIMENSIONS
    
    # Define Strassen's 7 multiplications for 2x2 matrices.
    # For A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e  
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # Result: C = [[m1+m4-m5+m7, m3+m5],
    #              [m2+m4, m1-m2+m3+m6]]
    #
    # Each multiplication is represented as (A_coeffs, B_coeffs, C_coeffs)
    # where each is a dict mapping (row, col) to coefficient.
    
    strassen = [
        # s=0: m1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s=1: m2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # s=2: m3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # s=3: m4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # s=4: m5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # s=5: m6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # s=6: m7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Block-level Strassen for 4x4 matrices split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    block_level = [
        # t=0: M1 = (A11+A22)(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # t=1: M2 = (A21+A22)B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # t=2: M3 = A11(B12-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # t=3: M4 = A22(B21-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # t=4: M5 = (A11+A12)B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # t=5: M6 = (A21-A11)(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # t=6: M7 = (A12-A22)(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    col = 0
    for t in range(7):
        A_blocks, B_blocks, C_blocks = block_level[t]
        for s in range(7):
            X_coeffs, Y_coeffs, C_coeffs = strassen[s]
            
            # U: A entries
            for (br, bc), a_coeff in A_blocks.items():
                for (r, c), x_coeff in X_coeffs.items():
                    i = br * 2 + r
                    j = bc * 2 + c
                    U[i * 4 + j, col] += a_coeff * x_coeff
            
            # V: B entries
            for (br, bc), b_coeff in B_blocks.items():
                for (r, c), y_coeff in Y_coeffs.items():
                    j = br * 2 + r
                    k = bc * 2 + c
                    V[j * 4 + k, col] += b_coeff * y_coeff
            
            # W: C entries (trace convention: W[k*n+i])
            for (br, bc), c_coeff in C_blocks.items():
                for (r, c), w_coeff in C_coeffs.items():
                    i = br * 2 + r
                    k = bc * 2 + c
                    W[k * 4 + i, col] += c_coeff * w_coeff
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

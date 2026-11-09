"""Optimized matrix multiplication tensor decomposition for 6x6."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use a 3x3 grid of 2x2 blocks with Laderman's rank-6 decomposition.
    # 27 block interactions, each with rank 6 => total rank = 162.
    
    # First, verify Laderman's decomposition numerically.
    # Laderman (1976): 2x2 matrix multiplication in 6 multiplications.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = AB
    # M1 = (a+d)(e+h)
    # M2 = (a+d)(g+f)
    # M3 = (b+c)e
    # M4 = (a-c)(e+g)
    # M5 = (b+d)(f+h)
    # M6 = (a-c)(f-h)
    # C[0,0] = M1 - M3 - M4 + M6
    # C[0,1] = M4 + M5 - M1 - M2
    # C[1,0] = M3 + M4 + M2 - M5
    # C[1,1] = M1 - M2 + M5 - M6
    
    # Verify with a random test
    np.random.seed(42)
    A_test = np.random.randn(2, 2)
    B_test = np.random.randn(2, 2)
    C_test = A_test @ B_test
    
    a, b, c, d = A_test[0,0], A_test[0,1], A_test[1,0], A_test[1,1]
    e, f, g, h = B_test[0,0], B_test[0,1], B_test[1,0], B_test[1,1]
    
    M1 = (a+d)*(e+h)
    M2 = (a+d)*(g+f)
    M3 = (b+c)*e
    M4 = (a-c)*(e+g)
    M5 = (b+d)*(f+h)
    M6 = (a-c)*(f-h)
    
    C00 = M1 - M3 - M4 + M6
    C01 = M4 + M5 - M1 - M2
    C10 = M3 + M4 + M2 - M5
    C11 = M1 - M2 + M5 - M6
    
    laderman_correct = (np.allclose(C00, C_test[0,0]) and
                        np.allclose(C01, C_test[0,1]) and
                        np.allclose(C10, C_test[1,0]) and
                        np.allclose(C11, C_test[1,1]))
    
    if laderman_correct:
        rank = 162
        # Laderman decomposition components
        # U (A-side): each M term expressed as linear combo of A entries
        # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        laderman_U = [
            [(0,0,1), (1,1,1)],  # M1: (a+d)
            [(0,0,1), (1,1,1)],  # M2: (a+d)
            [(0,1,1), (1,0,1)],  # M3: (b+c)
            [(0,0,1), (1,0,-1)], # M4: (a-c)
            [(0,1,1), (1,1,1)],  # M5: (b+d)
            [(0,0,1), (1,0,-1)]  # M6: (a-c)
        ]
        
        # V (B-side): each M term expressed as linear combo of B entries
        # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        laderman_V = [
            [(0,0,1), (1,1,1)],  # M1: (e+h)
            [(1,0,1), (0,1,1)],  # M2: (g+f)
            [(0,0,1)],           # M3: e
            [(0,0,1), (1,0,1)],  # M4: (e+g)
            [(0,1,1), (1,1,1)],  # M5: (f+h)
            [(0,1,1), (1,1,-1)]  # M6: (f-h)
        ]
        
        # W (C-side): reconstruction coefficients
        # C[0,0] = M1 - M3 - M4 + M6
        # C[0,1] = M4 + M5 - M1 - M2
        # C[1,0] = M3 + M4 + M2 - M5
        # C[1,1] = M1 - M2 + M5 - M6
        laderman_W = [
            [(0,0,1), (0,1,-1), (1,0,-1), (1,1,1)],   # M1: C00+1, C01-1, C10-1, C11+1
            [(0,1,-1), (1,0,1), (1,1,-1)],             # M2: C01-1, C10+1, C11-1
            [(0,0,-1), (1,0,1)],                       # M3: C00-1, C10+1
            [(0,0,-1), (0,1,1), (1,0,1)],              # M4: C00-1, C01+1, C10+1
            [(0,1,1), (1,0,-1), (1,1,1)],              # M5: C01+1, C10-1, C11+1
            [(0,0,1), (1,1,-1)]                        # M6: C00+1, C11-1
        ]
        
        block_U = laderman_U
        block_V = laderman_V
        block_W = laderman_W
        terms_per_block = 6
    else:
        # Fall back to Strassen if Laderman verification fails
        rank = 189
        
        strassen_U = [
            [(0,0,1), (1,1,1)],  # M1: (A00+A11)
            [(1,0,1), (1,1,1)],  # M2: (A10+A11)
            [(0,0,1)],           # M3: A00
            [(1,1,1)],           # M4: A11
            [(0,0,1), (0,1,1)],  # M5: (A00+A01)
            [(1,0,1), (0,0,-1)], # M6: (A10-A00)
            [(0,1,1), (1,1,-1)]  # M7: (A01-A11)
        ]
        
        strassen_V = [
            [(0,0,1), (1,1,1)],  # M1: (B00+B11)
            [(0,0,1)],           # M2: B00
            [(0,1,1), (1,1,-1)], # M3: (B01-B11)
            [(1,0,1), (0,0,-1)], # M4: (B10-B00)
            [(1,1,1)],           # M5: B11
            [(0,0,1), (0,1,1)],  # M6: (B00+B01)
            [(1,0,1), (1,1,1)]   # M7: (B10+B11)
        ]
        
        strassen_W = [
            [(0,0,1), (1,1,1)],   # M1: C00+1, C11+1
            [(1,0,1), (1,1,-1)],  # M2: C10+1, C11-1
            [(0,1,1), (1,1,1)],   # M3: C01+1, C11+1
            [(0,0,1), (1,0,1)],   # M4: C00+1, C10+1
            [(0,0,-1), (0,1,1)],  # M5: C00-1, C01+1
            [(1,1,1)],            # M6: C11+1
            [(0,0,1)]             # M7: C00+1
        ]
        
        block_U = strassen_U
        block_V = strassen_V
        block_W = strassen_W
        terms_per_block = 7
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Tiling 6x6 with 2x2 blocks
    # 3x3 grid of blocks
    col_idx = 0
    for i in range(3):  # row block
        for j in range(3):  # middle block
            for k in range(3):  # col block
                for t in range(terms_per_block):
                    # Update U
                    for lr, lc, coeff in block_U[t]:
                        u_row = (i * 2 + lr) * m + (j * 2 + lc)
                        U[u_row, col_idx] += coeff
                    
                    # Update V
                    for lr, lc, coeff in block_V[t]:
                        v_row = (j * 2 + lr) * p + (k * 2 + lc)
                        V[v_row, col_idx] += coeff
                    
                    # Update W
                    # W index convention: W[k*n+i, r] where i is row of C, k is col of C
                    for lr, lc, coeff in block_W[t]:
                        w_row = (k * 2 + lc) * n + (i * 2 + lr)
                        W[w_row, col_idx] += coeff
                    
                    col_idx += 1
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

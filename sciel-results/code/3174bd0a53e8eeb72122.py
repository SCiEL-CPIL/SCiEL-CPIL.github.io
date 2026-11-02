"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strassen-based decomposition: split 6x6 into 2x2 blocks of 3x3
    # 7 Strassen steps, each with 27 schoolbook terms for 3x3 multiplication
    # Total rank = 7 * 27 = 189 (better than schoolbook 216)
    
    rank = 7 * 27
    U = np.zeros((n * m, rank))  # 36 x 189
    V = np.zeros((m * p, rank))  # 36 x 189
    W = np.zeros((p * n, rank))  # 36 x 189
    
    # Strassen formulas coefficients:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # steps[s] = (A_blocks, B_blocks)
    # A_blocks: list of (block_index, sign)
    # B_blocks: list of (block_index, sign)
    steps = [
        # M1
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),
        # M3
        ([ (0, 1) ], [ (1, 1), (3, -1) ]),
        # M4
        ([ (3, 1) ], [ (2, 1), (0, -1) ]),
        # M5
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),
        # M6
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),
        # M7
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ]),
    ]
    
    # c_contrib[s] = list of (block_index, sign) for output C blocks
    c_contrib = [
        # M1: C00(+1), C11(+1)
        [ (0, 1), (3, 1) ],
        # M2: C10(+1), C11(-1)
        [ (2, 1), (3, -1) ],
        # M3: C01(+1), C11(+1)
        [ (1, 1), (3, 1) ],
        # M4: C00(-1), C10(+1)
        [ (0, -1), (2, 1) ],
        # M5: C01(+1)
        [ (1, 1) ],
        # M6: C00(+1)
        [ (0, 1) ],
        # M7: C00(-1), C11(+1)
        [ (0, -1), (3, 1) ],
    ]
    
    # Vectorized construction
    # We iterate over the 7 Strassen steps.
    # For each step, we have 27 terms (i, j, k) for the 3x3 multiplication.
    # Instead of looping over all 189 columns in Python, we can construct
    # the pattern for the 27 columns of one step and assign them at once.
    
    for s in range(7):
        a_blocks = steps[s][0]
        b_blocks = steps[s][1]
        c_blocks = c_contrib[s]
        
        # The 27 columns for this step correspond to all (i, j, k) in 0..2
        # Column index in U, V, W for this step: s*27 + i*9 + j*3 + k
        
        # We need to fill U[:, s*27:(s+1)*27], etc.
        # It's easier to just loop over the 27 terms if the vectorization overhead
        # of creating sparse matrices or complex indexing is high.
        # However, 27 iterations is very small.
        
        # Let's try to be slightly more efficient by pre-calculating the 
        # local indices for the 3x3 block multiplication.
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = s * 27 + i * 9 + j * 3 + k
                    
                    # U contributions
                    for a_block, a_sign in a_blocks:
                        a_row = (a_block // 2) * 3 + i
                        a_col = (a_block % 2) * 3 + j
                        U[a_row * m + a_col, col] += a_sign
                    
                    # V contributions
                    for b_block, b_sign in b_blocks:
                        b_row = (b_block // 2) * 3 + j
                        b_col = (b_block % 2) * 3 + k
                        V[b_row * p + b_col, col] += b_sign
                    
                    # W contributions (trace convention: W[k*n+i])
                    for c_block, c_sign in c_blocks:
                        c_row = (c_block // 2) * 3 + i
                        c_col = (c_block % 2) * 3 + k
                        W[c_col * n + c_row, col] += c_sign

    # Check deadline
    if time.time() > DEADLINE:
        # Fallback to schoolbook if we're running out of time (shouldn't happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists of lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

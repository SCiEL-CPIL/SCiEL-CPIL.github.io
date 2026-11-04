import numpy as np
import time
import sys

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 7
    
    # Build the 4x4x7 Strassen decomposition with rank 93
    # We use the same structure as the current program but written differently
    
    total_rank = 93
    
    # Define Strassen 2x2x2 terms
    # Each term: (x_coeffs, y_coeffs, c_coeffs)
    # x_coeffs: dict mapping (i,j) -> coeff for A
    # y_coeffs: dict mapping (i,j) -> coeff for B
    # c_coeffs: dict mapping (i,j) -> coeff for C
    
    s22 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Strassen 4x4x4 outer terms
    # Each: (x_block_def, y_block_def) where block_def maps block_index -> coeff
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    s44_outer = [
        ({0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}),
        ({0:1}, {1:1, 3:-1}),
        ({3:1}, {2:1, 0:-1}),
        ({0:1, 1:1}, {3:1}),
        ({2:1, 0:-1}, {0:1, 1:1}),
        ({1:1, 3:-1}, {2:1, 3:1}),
    ]
    
    # C-block coefficients for each of the 7 outer Strassen 4x4 terms
    # c_block maps (row_block, col_block) -> coeff
    c44_blocks = [
        {(0,0):1.0, (1,1):1.0},
        {(1,0):1.0, (1,1):-1.0},
        {(0,1):1.0, (1,1):1.0},
        {(0,0):1.0, (1,0):1.0},
        {(0,0):-1.0, (0,1):1.0},
        {(1,1):1.0},
        {(0,0):1.0},
    ]
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    rank = 0
    
    # Part 1: 4x4x4 Strassen for columns 0-3 (rank 49)
    for outer_idx in range(7):
        x_def, y_def = s44_outer[outer_idx]
        c_block = c44_blocks[outer_idx]
        
        for inner_idx in range(7):
            r = rank
            rank += 1
            x_sub, y_sub, c_sub = s22[inner_idx]
            
            # Fill U
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2*bi + li
                    j = 2*bj + lj
                    U[i*m + j, r] += cx * cb
            
            # Fill V for columns 0-3
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2*bi + li
                    j = 2*bj + lj
                    V[i*p + j, r] += cy * cb
            
            # Fill W for columns 0-3
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2*br + li
                    k_final = 2*bc + lj
                    W[k_final*n + i_final, r] += cc * cb_c
    
    # Part 2: 4x4x3 for columns 4-6
    # Decompose into four 2x2x3 blocks
    # Each 2x2x3 uses Strassen 2x2x2 (rank 7) + 4 schoolbook = rank 11
    # Total: 4 * 11 = 44
    
    def add_block(ar, ac, br, cr):
        """Add a 2x2x3 block at A[2ar:2ar+2, 2ac:2ac+2] * B[2br:2br+2, 4:7] -> C[2cr:2cr+2, 4:7]"""
        nonlocal rank
        
        # Strassen 2x2x2 for B columns 4,5 (local cols 0,1)
        for t in range(7):
            r = rank
            rank += 1
            x_sub, y_sub, c_sub = s22[t]
            
            for (li, lj), cx in x_sub.items():
                i = 2*ar + li
                j = 2*ac + lj
                U[i*m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2*br + li
                j = 4 + lj
                V[i*p + j, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2*cr + li
                k_final = 4 + lj
                W[k_final*n + i_final, r] += cc
        
        # Schoolbook for B column 6 (local col 2)
        # C[i][6] = A[i][0]*B[0][6] + A[i][1]*B[1][6]
        
        # A[0,0] * B[0,2]
        r = rank; rank += 1
        U[(2*ar)*m + (2*ac), r] = 1.0
        V[(2*br)*p + 6, r] = 1.0
        W[6*n + 2*cr, r] = 1.0
        
        # A[0,1] * B[1,2]
        r = rank; rank += 1
        U[(2*ar)*m + (2*ac+1), r] = 1.0
        V[(2*br+1)*p + 6, r] = 1.0
        W[6*n + 2*cr, r] = 1.0
        
        # A[1,0] * B[0,2]
        r = rank; rank += 1
        U[(2*ar+1)*m + (2*ac), r] = 1.0
        V[(2*br)*p + 6, r] = 1.0
        W[6*n + 2*cr+1, r] = 1.0
        
        # A[1,1] * B[1,2]
        r = rank; rank += 1
        U[(2*ar+1)*m + (2*ac+1), r] = 1.0
        V[(2*br+1)*p + 6, r] = 1.0
        W[6*n + 2*cr+1, r] = 1.0
    
    # Four blocks:
    # Top-left A * Top B -> Top C
    add_block(0, 0, 0, 0)
    # Top-right A * Bottom B -> Top C
    add_block(0, 1, 1, 0)
    # Bottom-left A * Top B -> Bottom C
    add_block(1, 0, 0, 1)
    # Bottom-right A * Bottom B -> Bottom C
    add_block(1, 1, 1, 1)
    
    assert rank == 93, f"Expected rank 93, got {rank}"
    
    U = U[:, :rank].tolist()
    V = V[:, :rank].tolist()
    W = W[:, :rank].tolist()
    
    return {"U": U, "V": V, "W": W}

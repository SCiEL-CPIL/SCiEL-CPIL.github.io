import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    n = 4
    m = 4
    p = 4
    
    # Rank 28 decomposition: 7 Strassen steps, each using rank-4 schoolbook for 2x2 blocks
    rank = 28
    
    # Initialize U, V, W as numpy arrays for efficient construction
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 block coefficients
    # r: 0..6
    # left_coeffs: [c00, c01, c10, c11] for A blocks
    # right_coeffs: [c00, c01, c10, c11] for B blocks
    # out_coeffs: dict { (block_row, block_col): coeff }
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    strassen_params = [
        # r=0 (M1)
        {
            'left': [1, 0, 0, 1],   # A00 + A11
            'right': [1, 0, 0, 1],  # B00 + B11
            'out': {(0,0): 1, (1,1): 1} # C00 += M1, C11 += M1
        },
        # r=1 (M2)
        {
            'left': [0, 0, 1, 1],   # A10 + A11
            'right': [1, 0, 0, 0],  # B00
            'out': {(1,0): 1, (1,1): -1} # C10 += M2, C11 -= M2
        },
        # r=2 (M3)
        {
            'left': [1, 0, 0, 0],   # A00
            'right': [0, 1, 0, -1], # B01 - B11
            'out': {(0,1): 1, (1,1): 1} # C01 += M3, C11 += M3
        },
        # r=3 (M4)
        {
            'left': [0, 0, 0, 1],   # A11
            'right': [0, 0, 1, -1], # B10 - B00
            'out': {(0,0): 1, (1,0): 1} # C00 += M4, C10 += M4
        },
        # r=4 (M5)
        {
            'left': [1, 1, 0, 0],   # A00 + A01
            'right': [0, 0, 0, 1],  # B11
            'out': {(0,0): -1, (0,1): 1} # C00 -= M5, C01 += M5
        },
        # r=5 (M6)
        {
            'left': [0, 0, 1, -1],  # A10 - A00
            'right': [1, 1, 0, 0],  # B00 + B01
            'out': {(1,1): 1} # C11 += M6
        },
        # r=6 (M7)
        {
            'left': [0, 1, 0, -1],  # A01 - A11
            'right': [0, 0, 1, 1],  # B10 + B11
            'out': {(0,0): 1} # C00 += M7
        }
    ]
    
    # Schoolbook 2x2 terms: (i, k, j)
    # s=0: (0,0,0) -> A00*B00
    # s=1: (0,1,1) -> A01*B11
    # s=2: (1,0,0) -> A10*B00
    # s=3: (1,1,1) -> A11*B11
    schoolbook_terms = [(0,0,0), (0,1,1), (1,0,0), (1,1,1)]
    
    # Block offsets for 4x4 matrix partitioned into 2x2 blocks
    # Block (br, bc) has top-left at (2*br, 2*bc)
    block_offsets = {
        (0,0): (0, 0),
        (0,1): (0, 2),
        (1,0): (2, 0),
        (1,1): (2, 2)
    }
    
    for r, params in enumerate(strassen_params):
        left_coeffs = params['left']
        right_coeffs = params['right']
        out_coeffs = params['out']
        
        for s, (i, k, j) in enumerate(schoolbook_terms):
            col = r * 4 + s
            
            # U: Input A
            # L_r[i, k] = sum_b left_coeffs[b] * A_b[i, k]
            for b_idx, (br, bc) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
                coeff = left_coeffs[b_idx]
                if coeff != 0:
                    row_off, col_off = block_offsets[(br, bc)]
                    row_global = (row_off + i) * 4 + (col_off + k)
                    U[row_global, col] += coeff
            
            # V: Input B
            # R_r[k, j] = sum_b right_coeffs[b] * B_b[k, j]
            for b_idx, (br, bc) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
                coeff = right_coeffs[b_idx]
                if coeff != 0:
                    row_off, col_off = block_offsets[(br, bc)]
                    row_global = (row_off + k) * 4 + (col_off + j)
                    V[row_global, col] += coeff
            
            # W: Output C
            # M_r[i, j] contributes to C blocks
            for (obr, ocb), w_coeff in out_coeffs.items():
                if w_coeff != 0:
                    row_off, col_off = block_offsets[(obr, ocb)]
                    row_global = (row_off + i) * 4 + (col_off + j)
                    W[row_global, col] += w_coeff
    
    # Verify
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to rank 49 Kronecker
        U2 = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=int)
        V2 = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ], dtype=int)
        W2 = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=int)
        
        U4 = np.kron(U2, U2)
        V4 = np.kron(V2, V2)
        W4 = np.kron(W2, W2)
        
        perm = np.zeros(16, dtype=int)
        for S in range(16):
            i = S // 4
            j = S % 4
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            R = (i_hi * 2 + j_hi) * 4 + (i_lo * 2 + j_lo)
            perm[S] = R
            
        U = U4[perm, :]
        V = V4[perm, :]
        W = W4[perm, :]
        
        if not verify_decomposition(U, V, W, n, m, p):
             # Last resort schoolbook 64
             rank = 64
             U = np.zeros((16, rank), dtype=int)
             V = np.zeros((16, rank), dtype=int)
             W = np.zeros((16, rank), dtype=int)
             col = 0
             for i in range(4):
                 for j in range(4):
                     for k in range(4):
                         U[i*4+j, col] = 1
                         V[j*4+k, col] = 1
                         W[k*4+i, col] = 1
                         col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

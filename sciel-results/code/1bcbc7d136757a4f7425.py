"""Matrix multiplication 4x4x8 tensor decomposition - rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Find a low-rank exact decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Strassen's algorithm recursively to construct a rank-98 decomposition:
    - Split 4x8 into two 4x4 matrices
    - Each 4x4x4 product uses Strassen with rank 49
    - Total rank: 49 + 49 = 98
    All coefficients are integers.
    """
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2x2
    # Index mapping: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    
    # X_s: linear comb of A entries
    X_strassen = [
        [1, 0, 0, 1],   # X0 = A00 + A11
        [0, 0, 1, 1],   # X1 = A10 + A11
        [1, 0, 0, 0],   # X2 = A00
        [0, 0, 0, 1],   # X3 = A11
        [1, 1, 0, 0],   # X4 = A00 + A01
        [0, 0, 1, -1],  # X5 = A10 - A00
        [0, 1, 0, -1]   # X6 = A01 - A11
    ]
    
    # Y_s: linear comb of B entries
    Y_strassen = [
        [1, 0, 0, 1],   # Y0 = B00 + B11
        [1, 0, 0, 0],   # Y1 = B00
        [0, 1, 0, -1],  # Y2 = B01 - B11
        [0, 0, 1, -1],  # Y3 = B10 - B00
        [0, 0, 0, 1],   # Y4 = B11
        [1, 1, 0, 0],   # Y5 = B00 + B01
        [0, 0, 1, 1]    # Y6 = B10 + B11
    ]
    
    # Z_s: contribution to C entries, as list of (row, col, coeff)
    Z_strassen = [
        [(0,0,1), (1,1,1)],   # Z0: C00 += Z0, C11 += Z0
        [(1,0,1), (1,1,-1)],  # Z1: C10 += Z1, C11 -= Z1
        [(0,1,1), (1,1,1)],   # Z2: C01 += Z2, C11 += Z2
        [(0,0,1), (1,0,1)],   # Z3: C00 += Z3, C10 += Z3
        [(0,0,-1), (0,1,1)],  # Z4: C00 -= Z4, C01 += Z4
        [(1,1,1)],            # Z5: C11 += Z5
        [(0,0,1)]             # Z6: C00 += Z6
    ]
    
    # Convert Z_strassen to dict format for easier iteration
    Z_strassen_dict = []
    for z in Z_strassen:
        d = {}
        for (i, j, coeff) in z:
            d[(i, j)] = coeff
        Z_strassen_dict.append(d)
    
    # Outer Strassen block combinations for 4x4x4
    # X_outer_blocks[t]: { (br, bc): coeff } for A blocks
    X_outer_blocks = [
        { (0,0): 1, (1,1): 1 },   # X0 = A00 + A11
        { (1,0): 1, (1,1): 1 },   # X1 = A10 + A11
        { (0,0): 1 },              # X2 = A00
        { (1,1): 1 },              # X3 = A11
        { (0,0): 1, (0,1): 1 },   # X4 = A00 + A01
        { (1,0): 1, (0,0): -1 },  # X5 = A10 - A00
        { (0,1): 1, (1,1): -1 }   # X6 = A01 - A11
    ]
    
    Y_outer_blocks = [
        { (0,0): 1, (1,1): 1 },   # Y0 = B00 + B11
        { (0,0): 1 },              # Y1 = B00
        { (0,1): 1, (1,1): -1 },  # Y2 = B01 - B11
        { (1,0): 1, (0,0): -1 },  # Y3 = B10 - B00
        { (1,1): 1 },              # Y4 = B11
        { (0,0): 1, (0,1): 1 },   # Y5 = B00 + B01
        { (1,0): 1, (1,1): 1 }    # Y6 = B10 + B11
    ]
    
    Z_outer_blocks = [
        { (0,0): 1, (1,1): 1 },   # C00 += M0, C11 += M0
        { (1,0): 1, (1,1): -1 },  # C10 += M1, C11 -= M1
        { (0,1): 1, (1,1): 1 },   # C01 += M2, C11 += M2
        { (0,0): 1, (1,0): 1 },   # C00 += M3, C10 += M3
        { (0,0): -1, (0,1): 1 },  # C00 -= M4, C01 += M4
        { (1,1): 1 },              # C11 += M5
        { (0,0): 1 }               # C00 += M6
    ]
    
    def build_4x4x4_strassen():
        """Build the rank-49 Strassen decomposition for 4x4x4."""
        rank = 49
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((16, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)
        
        def a_idx(br, bc, lr, lc):
            """Global index in A (4x4) for block (br,bc), local (lr,lc)."""
            i = br * 2 + lr
            j = bc * 2 + lc
            return i * 4 + j
        
        def b_idx(br, bc, lr, lc):
            """Global index in B (4x4) for block (br,bc), local (lr,lc)."""
            j = br * 2 + lr
            k = bc * 2 + lc
            return j * 4 + k
        
        def c_idx(br, bc, lr, lc):
            """Global index in C (4x4) for block (br,bc), local (lr,lc)."""
            k = br * 2 + lr
            i = bc * 2 + lc
            return k * 4 + i
        
        col = 0
        for t_outer in range(7):
            for t_inner in range(7):
                # U: linear combination of A entries
                for (br_A, bc_A), coeff_outer in X_outer_blocks[t_outer].items():
                    for lr in range(2):
                        for lc in range(2):
                            coeff_inner = X_strassen[t_inner][lr * 2 + lc]
                            if coeff_inner != 0:
                                a_index = a_idx(br_A, bc_A, lr, lc)
                                U[a_index, col] += coeff_outer * coeff_inner
                
                # V: linear combination of B entries
                for (br_B, bc_B), coeff_outer in Y_outer_blocks[t_outer].items():
                    for lr in range(2):
                        for lc in range(2):
                            coeff_inner = Y_strassen[t_inner][lr * 2 + lc]
                            if coeff_inner != 0:
                                b_index = b_idx(br_B, bc_B, lr, lc)
                                V[b_index, col] += coeff_outer * coeff_inner
                
                # W: contribution to C entries
                for (lr_M, lc_M), coeff_z_inner in Z_strassen_dict[t_inner].items():
                    if coeff_z_inner != 0:
                        for (br_C, bc_C), coeff_z_outer in Z_outer_blocks[t_outer].items():
                            c_index = c_idx(br_C, bc_C, lr_M, lc_M)
                            W[c_index, col] += coeff_z_outer * coeff_z_inner
                
                col += 1
        
        return U, V, W, rank
    
    # Build 4x4x4 decomposition
    U_444, V_444, W_444, rank_444 = build_4x4x4_strassen()
    
    # For 4x4x8: split B into B1 (4x4) and B2 (4x4)
    # A * [B1 | B2] = [A*B1 | A*B2]
    rank_total = 2 * rank_444  # 98
    U = np.zeros((16, rank_total), dtype=int)
    V = np.zeros((32, rank_total), dtype=int)  # m*p = 4*8 = 32
    W = np.zeros((32, rank_total), dtype=int)  # p*n = 8*4 = 32
    
    # Copy U for both halves
    U[:, 0:rank_444] = U_444
    U[:, rank_444:rank_total] = U_444
    
    # Fill in V:
    # B1: left half of B (columns 0-3)
    # B2: right half of B (columns 4-7)
    for col in range(rank_444):
        for j in range(4):
            for k in range(4):
                val = V_444[j*4 + k, col]
                V[j*8 + k, col] = val  # B1: column k in 0..3
                V[j*8 + k + 4, col + rank_444] = val  # B2: column k+4 in 4..7
    
    # Fill in W:
    # C1: left half of C (columns 0-3)
    # C2: right half of C (columns 4-7)
    # W uses trace convention: W[k*n+i, r] where k is column index in C, i is row index
    for col in range(rank_444):
        for k in range(4):
            for i in range(4):
                val = W_444[k*4 + i, col]
                W[k*4 + i, col] = val  # C1: column k in 0..3
                W[(k+4)*4 + i, col + rank_444] = val  # C2: column k+4 in 4..7
    
    # Convert to lists
    U_list = [[float(U[i, c]) for c in range(rank_total)] for i in range(16)]
    V_list = [[float(V[i, c]) for c in range(rank_total)] for i in range(32)]
    W_list = [[float(W[i, c]) for c in range(rank_total)] for i in range(32)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

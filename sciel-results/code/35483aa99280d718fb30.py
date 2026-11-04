import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Rank 32 decomposition:
    # 28 for 4x4x4 (Strassen blocks + Schoolbook scalars)
    # 4 for 4x4x1 (Schoolbook)
    
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 2: 4x4x1 (Rank 4) ---
    # C[:, 4] = A @ B[:, 4]
    # C_i,4 = sum_j A_i,j B_j,4
    for j in range(4):
        c = 28 + j
        # U: A_i,j
        for i in range(4):
            U[i * 4 + j, c] = 1.0
        # V: B_j,4
        V[j * 5 + 4, c] = 1.0
        # W: C_4,i
        for i in range(4):
            W[4 * 4 + i, c] = 1.0
            
    # --- Part 1: 4x4x4 (Rank 28) ---
    # Strassen's 7 terms
    # Each term is a 2x2 matrix product X_s Y_s
    # Decompose each 2x2 product using Schoolbook (4 terms)
    
    # Define Strassen terms:
    # s, X_blocks, Y_blocks, C_blocks (list of (block_idx, sign))
    # Block indices: 00, 01, 10, 11
    # X_s is sum of A_blocks with coeffs
    # Y_s is sum of B_blocks with coeffs
    
    strassen_terms = [
        # M1
        (
            {0: 1, 3: 1},  # A00 + A11
            {0: 1, 3: 1},  # B00 + B11
            [(0, 1), (3, 1)] # C00 += M1, C11 += M1
        ),
        # M2
        (
            {1: 1, 3: 1},  # A10 + A11
            {0: 1},        # B00
            [(2, 1)]       # C10 += M2
        ),
        # M3
        (
            {0: 1},        # A00
            {1: 1, 3: -1}, # B01 - B11
            [(1, 1)]       # C01 += M3
        ),
        # M4
        (
            {3: 1},        # A11
            {2: 1, 0: -1}, # B10 - B00
            [(0, 1), (2, 1)] # C00 += M4, C10 += M4
        ),
        # M5
        (
            {0: 1, 1: 1},  # A00 + A01
            {3: 1},        # B11
            [(0, -1), (1, 1)] # C00 -= M5, C01 += M5
        ),
        # M6
        (
            {1: 1, 0: -1}, # A10 - A00
            {0: 1, 1: 1},  # B00 + B01
            [(3, 1)]       # C11 += M6
        ),
        # M7
        (
            {1: 1, 3: -1}, # A01 - A11
            {2: 1, 3: 1},  # B10 + B11
            [(0, 1)]       # C00 += M7
        ),
    ]
    
    # Helper to get block indices for A and B
    # A blocks:
    # 0: 00 (rows 0-1, cols 0-1)
    # 1: 01 (rows 0-1, cols 2-3)
    # 2: 10 (rows 2-3, cols 0-1)
    # 3: 11 (rows 2-3, cols 2-3)
    
    def get_block_indices(block_id):
        if block_id == 0:
            return [(0,0), (0,1), (1,0), (1,1)]
        elif block_id == 1:
            return [(0,2), (0,3), (1,2), (1,3)]
        elif block_id == 2:
            return [(2,0), (2,1), (3,0), (3,1)]
        elif block_id == 3:
            return [(2,2), (2,3), (3,2), (3,3)]

    col = 0
    for s_idx in range(7):
        X_coeffs, Y_coeffs, C_targets = strassen_terms[s_idx]
        
        # For each of the 4 Schoolbook terms in the 2x2 multiplication
        # u,v,w in 0..1
        # Terms:
        # 1. u=0, v=0, w=0 -> C[0,0]
        # 2. u=0, v=1, w=0 -> C[0,0]
        # 3. u=1, v=0, w=1 -> C[1,0]
        # 4. u=1, v=1, w=1 -> C[1,1]
        
        sb_terms = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 1),
            (1, 1, 1)
        ]
        
        for u, v, w in sb_terms:
            c = col
            col += 1
            
            # Construct U vector
            # X_s[u,v] is sum of A_i,j
            # X_s is formed by X_coeffs
            # X_s[u,v] corresponds to the entry (u,v) of the 2x2 block sum
            # The block sum is sum_{b} coeff_b * A_b
            # The entry (u,v) of A_b is A_{r_b+u, c_b+v}
            
            # We need the coefficient of each A_i,j in X_s[u,v]
            # X_s[u,v] = sum_{b} X_coeffs[b] * (A_b)[u,v]
            # (A_b)[u,v] is A_{row_base_b + u, col_base_b + v}
            
            U_vec = np.zeros(16)
            for b, coeff in X_coeffs.items():
                if coeff == 0: continue
                # Get base row/col for block b
                if b == 0:
                    r_base, c_base = 0, 0
                elif b == 1:
                    r_base, c_base = 0, 2
                elif b == 2:
                    r_base, c_base = 2, 0
                elif b == 3:
                    r_base, c_base = 2, 2
                
                i = r_base + u
                j = c_base + v
                U_vec[i * 4 + j] += coeff
            
            # Construct V vector
            # Y_s[v,w]
            V_vec = np.zeros(20)
            for b, coeff in Y_coeffs.items():
                if coeff == 0: continue
                if b == 0:
                    r_base, c_base = 0, 0
                elif b == 1:
                    r_base, c_base = 0, 2
                elif b == 2:
                    r_base, c_base = 2, 0
                elif b == 3:
                    r_base, c_base = 2, 2
                
                j = r_base + v
                k = c_base + w
                # k is 0..3 for B
                V_vec[j * 5 + k] += coeff
            
            # Construct W vector
            # Output C position
            # The 2x2 result M_s[u,w] is added to C blocks in C_targets
            # For each (block_idx, sign) in C_targets:
            # The entry M_s[u,w] maps to C_{r_base_block + u, c_base_block + w}
            
            W_vec = np.zeros(20)
            for b, sign in C_targets:
                if b == 0:
                    r_base, c_base = 0, 0
                elif b == 1:
                    r_base, c_base = 0, 2
                elif b == 2:
                    r_base, c_base = 2, 0
                elif b == 3:
                    r_base, c_base = 2, 2
                
                i = r_base + u
                k = c_base + w
                # W index is k*n + i
                W_vec[k * 4 + i] += sign
            
            U[:, c] = U_vec
            V[:, c] = V_vec
            W[:, c] = W_vec
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

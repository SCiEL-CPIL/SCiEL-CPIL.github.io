import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We construct a rank-36 decomposition using Strassen's algorithm for 2x2x2 blocks
    # and standard multiplication for the remaining 2x2x1 blocks.
    # 
    # Structure:
    # A is 2x4, B is 4x5, C is 2x5.
    # We split the middle dimension (j) into two blocks of size 2: j in {0,1} and j in {2,3}.
    # For each block, we multiply a 2x2 matrix (A_block) by a 2x5 matrix (B_block).
    # The result is a 2x5 matrix.
    # 
    # For the 2x2 by 2x5 multiplication:
    # We can split the 5 columns of B into two 2-column blocks and one 1-column block.
    # Block 1 (cols 0,1): 2x2 x 2x2 -> Use Strassen (rank 7)
    # Block 2 (cols 2,3): 2x2 x 2x2 -> Use Strassen (rank 7)
    # Block 3 (col 4):    2x2 x 2x1 -> Use Standard (rank 4)
    # Total rank per A-block: 7 + 7 + 4 = 18.
    # Two A-blocks: 18 * 2 = 36.
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    
    # Strassen terms for 2x2x2
    # Local indices:
    # A: (i, j) with i,j in {0,1}
    # B: (j, k) with j,k in {0,1}
    # C: (i, k) with i,k in {0,1}
    # 
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
    # M2 = (A10 + A11)B00        -> C10 += M2, C11 -= M2
    # M3 = A00(B01 - B11)        -> C01 += M3, C11 += M3
    # M4 = A11(B10 - B00)        -> C00 += M4, C10 += M4
    # M5 = (A00 + A01)B11        -> C00 -= M5, C01 += M5
    # M6 = (A10 - A00)(B00 + B01)-> C11 += M6
    # M7 = (A01 - A11)(B10 + B11)-> C00 += M7
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # Keys are (i,j), (j,k), (i,k) respectively
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        )
    ]
    
    # Standard terms for 2x2x1
    # A is 2x2, B is 2x1 (column vector), C is 2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 
    # Term 1: A00 * B0 -> C0
    # Term 2: A01 * B1 -> C0
    # Term 3: A10 * B0 -> C1
    # Term 4: A11 * B1 -> C1
    
    standard_terms_2x2x1 = [
        (
            {(0,0): 1},
            {(0,0): 1},
            {(0,0): 1}
        ),
        (
            {(0,1): 1},
            {(1,0): 1},
            {(0,0): 1}
        ),
        (
            {(1,0): 1},
            {(0,0): 1},
            {(1,0): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1},
            {(1,0): 1}
        )
    ]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, idx):
        """
        u_coeffs: dict {(i, j): val} where i in 0..n-1, j in 0..m-1
        v_coeffs: dict {(j, k): val} where j in 0..m-1, k in 0..p-1
        w_coeffs: dict {(k, i): val} where k in 0..p-1, i in 0..n-1
        """
        for (i, j), val in u_coeffs.items():
            U[i * m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i, idx] += val

    # We iterate over the two blocks of A (and corresponding rows of B)
    # Block 0: A rows 0,1; cols 0,1. B rows 0,1.
    # Block 1: A rows 0,1; cols 2,3. B rows 2,3.
    
    for block_idx in range(2):
        j_offset = block_idx * 2
        
        # Sub-block 0: B cols 0,1 (k_offset 0)
        # Uses Strassen
        for u_c, v_c, w_c in strassen_terms:
            # Map local indices to global
            u_g = {}
            v_g = {}
            w_g = {}
            
            for (i, j_l), val in u_c.items():
                j_g = j_offset + j_l
                u_g[(i, j_g)] = val
            
            for (j_l, k_l), val in v_c.items():
                j_g = j_offset + j_l
                k_g = 0 + k_l
                v_g[(j_g, k_g)] = val
            
            for (i, k_l), val in w_c.items():
                k_g = 0 + k_l
                w_g[(k_g, i)] = val
            
            add_term(u_g, v_g, w_g, col_idx)
            col_idx += 1
            
        # Sub-block 1: B cols 2,3 (k_offset 2)
        # Uses Strassen
        for u_c, v_c, w_c in strassen_terms:
            u_g = {}
            v_g = {}
            w_g = {}
            
            for (i, j_l), val in u_c.items():
                j_g = j_offset + j_l
                u_g[(i, j_g)] = val
            
            for (j_l, k_l), val in v_c.items():
                j_g = j_offset + j_l
                k_g = 2 + k_l
                v_g[(j_g, k_g)] = val
            
            for (i, k_l), val in w_c.items():
                k_g = 2 + k_l
                w_g[(k_g, i)] = val
            
            add_term(u_g, v_g, w_g, col_idx)
            col_idx += 1
            
        # Sub-block 2: B col 4 (k_offset 4)
        # Uses Standard 2x2x1
        for u_c, v_c, w_c in standard_terms_2x2x1:
            u_g = {}
            v_g = {}
            w_g = {}
            
            for (i, j_l), val in u_c.items():
                j_g = j_offset + j_l
                u_g[(i, j_g)] = val
            
            # B is 2x1 in this block, so j_l in {0,1}, k_l is 0 (maps to global 4)
            for (j_l, _), val in v_c.items():
                j_g = j_offset + j_l
                k_g = 4
                v_g[(j_g, k_g)] = val
            
            # C is 2x1 in this block, so i in {0,1}, k_l is 0 (maps to global 4)
            for (i, _), val in w_c.items():
                k_g = 4
                w_g[(k_g, i)] = val
            
            add_term(u_g, v_g, w_g, col_idx)
            col_idx += 1

    # Convert numpy arrays to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

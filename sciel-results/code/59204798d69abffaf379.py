"""
Matrix multiplication 6x6x6 rank-minimal exact tensor decomposition.
Uses Strassen's algorithm on 3x3 blocks, with schoolbook multiplication for the 3x3 blocks.
Total rank = 7 (Strassen) * 27 (Schoolbook 3x3) = 189.
"""

import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank = 7 * 27 = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper function to add a schoolbook 3x3 multiplication to the decomposition
    # left_coeffs: dict mapping (i, j) local index to list of (global_idx, coeff)
    # right_coeffs: dict mapping (i, j) local index to list of (global_idx, coeff)
    # output_map: dict mapping (i, j) local index to list of (global_idx, coeff)
    def add_3x3_schoolbook(left_coeffs, right_coeffs, output_map):
        nonlocal col
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Left factor term: left_coeffs[i][k]
                    # Right factor term: right_coeffs[k][j]
                    # Output term: output_map[i][j]
                    
                    # For each term in the linear combination of the left matrix element
                    for u_idx, u_coeff in left_coeffs[i][k]:
                        # For each term in the linear combination of the right matrix element
                        for v_idx, v_coeff in right_coeffs[k][j]:
                            # For each term in the linear combination of the output matrix element
                            for w_idx, w_coeff in output_map[i][j]:
                                # Add to U, V, W
                                U[u_idx][col] += u_coeff
                                V[v_idx][col] += v_coeff
                                W[w_idx][col] += w_coeff
                    col += 1
                    if col >= rank:
                        raise ValueError("Rank exceeded")

    # Define global indices for 3x3 blocks in 6x6 matrix
    # Block (bi, bj) where bi, bj in {0, 1}
    # Local indices li, lj in {0, 1, 2}
    # Global index = (bi * 3 + li) * 6 + (bj * 3 + lj)
    
    def get_block_coeffs(block_i, block_j, sign=1):
        """
        Returns a 3x3 list of lists of tuples (global_idx, coeff)
        representing the linear combination of the input matrix for the block (block_i, block_j).
        Here we assume the block is taken directly from the matrix A or B.
        """
        coeffs = [[[] for _ in range(3)] for _ in range(3)]
        for li in range(3):
            for lj in range(3):
                gi = block_i * 3 + li
                gj = block_j * 3 + lj
                global_idx = gi * 6 + gj
                coeffs[li][lj].append((global_idx, sign))
        return coeffs

    # Strassen Steps
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Output Assembly
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to define the output mapping for each Mi.
    # output_map[i][j] should contain the global indices and weights for the contribution of Mi[i][j].
    
    def get_output_map(m_idx):
        """
        Returns a 3x3 list of lists of tuples (global_idx, coeff)
        for the output block corresponding to Mi.
        """
        out = [[[] for _ in range(3)] for _ in range(3)]
        
        # Determine which global blocks Mi contributes to and with what weight
        # m_idx 0 -> M1, 1 -> M2, etc.
        
        contributions = {
            0: [(0, 0, 1), (1, 1, 1)],      # M1 -> C11 (+1), C22 (+1)
            1: [(1, 0, 1), (1, 1, -1)],     # M2 -> C21 (+1), C22 (-1)
            2: [(0, 1, 1), (1, 1, 1)],      # M3 -> C12 (+1), C22 (+1)
            3: [(0, 0, 1), (1, 0, 1)],      # M4 -> C11 (+1), C21 (+1)
            4: [(0, 0, -1), (0, 1, 1)],     # M5 -> C11 (-1), C12 (+1)
            5: [(1, 1, 1)],                 # M6 -> C22 (+1)
            6: [(0, 0, 1)]                  # M7 -> C11 (+1)
        }
        
        for li in range(3):
            for lj in range(3):
                for block_i, block_j, weight in contributions[m_idx]:
                    gi = block_i * 3 + li
                    gj = block_j * 3 + lj
                    global_idx = gi * 6 + gj
                    out[li][lj].append((global_idx, weight))
        
        return out

    # Define the linear combinations for the inputs for each Mi
    
    # M1: (A11 + A22) (B11 + B22)
    left_1 = [[[] for _ in range(3)] for _ in range(3)]
    right_1 = [[[] for _ in range(3)] for _ in range(3)]
    for li in range(3):
        for lj in range(3):
            gi = lj # local col
            gi_row = li # local row
            # A11
            global_idx_a11 = (0 * 3 + li) * 6 + (0 * 3 + lj)
            # A22
            global_idx_a22 = (1 * 3 + li) * 6 + (1 * 3 + lj)
            left_1[li][lj].append((global_idx_a11, 1))
            left_1[li][lj].append((global_idx_a22, 1))
            
            # B11
            global_idx_b11 = (0 * 3 + li) * 6 + (0 * 3 + lj)
            # B22
            global_idx_b22 = (1 * 3 + li) * 6 + (1 * 3 + lj)
            right_1[li][lj].append((global_idx_b11, 1))
            right_1[li][lj].append((global_idx_b22, 1))
            
    add_3x3_schoolbook(left_1, right_1, get_output_map(0))

    # M2: (A21 + A22) B11
    left_2 = [[[] for _ in range(3)] for _ in range(3)]
    right_2 = [[[] for _ in range(3)] for _ in range(3)]
    for li in range(3):
        for lj in range(3):
            # A21
            global_idx_a21 = (1 * 3 + li) * 6 + (0 * 3 + lj)
            # A22
            global_idx_a22 = (1 * 3 + li) * 6 + (1 * 3 + lj)
            left_2[li][lj].append((global_idx_a21, 1))
            left_2[li][lj].append((global_idx_a22, 1))
            
            # B11
            global_idx_b11 = (0 * 3 + li) * 6 + (0 * 3 + lj)
            right_2[li][lj].append((global_idx_b11, 1))
            
    add_3x3_schoolbook(left_2, right_2, get_output_map(1))

    # M3: A11 (B12 - B22)
    left_3 = [[[] for _ in range(3)] for _ in range(3)]
    right_3 = [[[] for _ in range(3)] for _ in range(3)]
    for li in range(3):
        for lj in range(3):
            # A11
            global_idx_a11 = (0 * 3 + li) * 6 + (0 * 3 + lj)
            left_3[li][lj].append((global_idx_a11, 1))
            
            # B12
            global_idx_b12 = (0 * 3 + li) * 6 + (1 * 3 + lj)
            # B22
            global_idx_b22 = (1 * 3 + li) * 6 + (1 * 3 + lj)
            right_3[li][lj].append((global_idx_b12, 1))
            right_3[li][lj].append((global_idx_b22, -1))
            
    add_3x3_schoolbook(left_3, right_3, get_output_map(2))

    # M4: A22 (B21 - B11)
    left_4 = [[[] for _ in range(3)] for _ in range(3)]
    right_4 = [[[] for _ in range(3)] for _ in range(3)]
    for li in range(3):
        for lj in range(3):
            # A22
            global_idx_a22 = (1 * 3 + li) * 6 + (1 * 3 + lj)
            left_4[li][lj].append((global_idx_a22, 1))
            
            # B21
            global_idx_b21 = (1 * 3 + li) * 6 + (0 * 3 + lj)
            # B11
            global_idx_b11 = (0 * 3 + li) * 6 + (0 * 3 + lj)
            right_4[li][lj].append((global_idx_b21, 1))
            right_4[li][lj].append((global_idx_b11, -1))
            
    add_3x3_schoolbook(left_4, right_4, get_output_map(3))

    # M5: (A11 + A12) B22
    left_5 = [[[] for _ in range(3)] for _ in range(3)]
    right_5 = [[[] for _ in range(3)] for _ in range(3)]
    for li in range(3):
        for lj in range(3):
            # A11
            global_idx_a11 = (0 * 3 + li) * 6 + (0 * 3 + lj)
            # A12
            global_idx_a12 = (0 * 3 + li) * 6 + (1 * 3 + lj)
            left_5[li][lj].append((global_idx_a11, 1))
            left_5[li][lj].append((global_idx_a12, 1))
            
            # B22
            global_idx_b22 = (1 * 3 + li) * 6 + (1 * 3 + lj)
            right_5[li][lj].append((global_idx_b22, 1))
            
    add_3x3_schoolbook(left_5, right_5, get_output_map(4))

    # M6: (A21 - A11) (B11 + B12)
    left_6 = [[[] for _ in range(3)] for _ in range(3)]
    right_6 = [[[] for _ in range(3)] for _ in range(3)]
    for li in range(3):
        for lj in range(3):
            # A21
            global_idx_a21 = (1 * 3 + li) * 6 + (0 * 3 + lj)
            # A11
            global_idx_a11 = (0 * 3 + li) * 6 + (0 * 3 + lj)
            left_6[li][lj].append((global_idx_a21, 1))
            left_6[li][lj].append((global_idx_a11, -1))
            
            # B11
            global_idx_b11 = (0 * 3 + li) * 6 + (0 * 3 + lj)
            # B12
            global_idx_b12 = (0 * 3 + li) * 6 + (1 * 3 + lj)
            right_6[li][lj].append((global_idx_b11, 1))
            right_6[li][lj].append((global_idx_b12, 1))
            
    add_3x3_schoolbook(left_6, right_6, get_output_map(5))

    # M7: (A12 - A22) (B21 + B22)
    left_7 = [[[] for _ in range(3)] for _ in range(3)]
    right_7 = [[[] for _ in range(3)] for _ in range(3)]
    for li in range(3):
        for lj in range(3):
            # A12
            global_idx_a12 = (0 * 3 + li) * 6 + (1 * 3 + lj)
            # A22
            global_idx_a22 = (1 * 3 + li) * 6 + (1 * 3 + lj)
            left_7[li][lj].append((global_idx_a12, 1))
            left_7[li][lj].append((global_idx_a22, -1))
            
            # B21
            global_idx_b21 = (1 * 3 + li) * 6 + (0 * 3 + lj)
            # B22
            global_idx_b22 = (1 * 3 + li) * 6 + (1 * 3 + lj)
            right_7[li][lj].append((global_idx_b21, 1))
            right_7[li][lj].append((global_idx_b22, 1))
            
    add_3x3_schoolbook(left_7, right_7, get_output_map(6))

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

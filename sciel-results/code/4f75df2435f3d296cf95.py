"""Program to find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    def strassen_decomposition_2x2():
        """Return the 7 Strassen coefficient sets for 2x2x2 matrix multiplication."""
        coeff_sets = []
        
        # P1 = (a+d)(e+h)
        coeff_sets.append(([1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1]))
        # P2 = (c+d)e
        coeff_sets.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # P3 = a(f-h)
        coeff_sets.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # P4 = d(g-e)
        coeff_sets.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
        # P5 = (a+b)h
        coeff_sets.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # P6 = (c-a)(e+f)
        coeff_sets.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # P7 = (b-d)(g+h)
        coeff_sets.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        return coeff_sets
    
    def expand_strassen_4x4():
        """Expand Strassen coefficients for 4x4x4 matrix multiplication (rank 49)."""
        top_level_coeffs = strassen_decomposition_2x2()
        bottom_level_coeffs = strassen_decomposition_2x2()
        
        # Mapping from 2x2 block index to 4x4 indices
        # Block 0 (top-left): [0, 1, 4, 5]
        # Block 1 (top-right): [2, 3, 6, 7]
        # Block 2 (bottom-left): [8, 9, 12, 13]
        # Block 3 (bottom-right): [10, 11, 14, 15]
        block_indices = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
        
        expanded = []
        for top_u, top_v, top_w in top_level_coeffs:
            for bot_u, bot_v, bot_w in bottom_level_coeffs:
                full_u = [0.0] * 16
                full_v = [0.0] * 16
                full_w = [0.0] * 16
                
                for block in range(4):
                    for sub in range(4):
                        idx = block_indices[block][sub]
                        full_u[idx] += top_u[block] * bot_u[sub]
                        full_v[idx] += top_v[block] * bot_v[sub]
                        full_w[idx] += top_w[block] * bot_w[sub]
                
                expanded.append((full_u, full_v, full_w))
        
        return expanded
    
    strassen_4x4 = expand_strassen_4x4()
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill first 49 terms for 4x4x4 part
    for c in range(49):
        u_coeffs, v_coeffs, w_coeffs = strassen_4x4[c]
        for i in range(16):
            U[i][c] = u_coeffs[i]
        for j in range(16):
            j_row = j // 4
            k_col = j % 4
            V[j_row * p + k_col][c] = v_coeffs[j]
        for i in range(16):
            i_row = i // 4
            k_col = i % 4
            W[k_col * n + i_row][c] = w_coeffs[i]
    
    # Fill next 16 terms for the 5th column
    for i in range(n):
        for j in range(m):
            c = 49 + i * m + j
            U[i * m + j][c] = 1
            V[j * p + 4][c] = 1
            W[4 * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

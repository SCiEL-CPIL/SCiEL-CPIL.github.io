"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen 2x2x2 blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Local indices: U: 2i+j, V: 2j+k, W: 2k+i
    # i,j,k in {0,1}
    
    strassen_terms = [
        # U, V, W
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, -1, -1, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
    ]
    
    # Blocks:
    # 0: A1 B11 (A col off 0, B row off 0, B col off 0)
    # 1: A2 B21 (A col off 2, B row off 2, B col off 0)
    # 2: A1 B12 (A col off 0, B row off 0, B col off 2)
    # 3: A2 B22 (A col off 2, B row off 2, B col off 2)
    
    blocks = [
        (0, 0, 0),
        (2, 2, 0),
        (0, 0, 2),
        (2, 2, 2)
    ]
    
    col = 0
    for a_off, b_row_off, b_col_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Map local indices to global
            # U local idx: 2i + j
            # V local idx: 2j + k
            # W local idx: 2k + i
            
            for local_idx in range(4):
                # Extract i, j, k from local_idx
                # U: 2i + j
                i = local_idx // 2
                j = local_idx % 2
                
                # Global U index
                u_global = 4 * i + (j + a_off)
                U[u_global][col] += u_coeffs[local_idx]
                
                # V: 2j + k -> j is same as U's j? No.
                # In Strassen, the sum is over j.
                # U term is sum over i,j of u_ij a_ij.
                # V term is sum over j,k of v_jk b_jk.
                # The index j must match.
                # My local_idx for U is 2i+j.
                # My local_idx for V is 2j+k.
                # This implies the "local_idx" variable is reused for different structures.
                # I need to iterate over all i,j,k combinations or handle the coefficients correctly.
                
                # The coefficients lists are length 4.
                # u_coeffs[local_idx] corresponds to basis element local_idx.
                # For U, basis is (i,j).
                # For V, basis is (j,k).
                # For W, basis is (k,i).
                
                # So I should just add the coefficient to the corresponding global index.
                pass # The loop above is wrong because local_idx maps differently for U, V, W.

            # Correct approach:
            # Iterate over the 4 basis elements for each matrix.
            # But the coefficients are given as a flat list of 4.
            # For U: index 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # For V: index 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # For W: index 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            
            for idx in range(4):
                # U
                i = idx // 2
                j = idx % 2
                u_global = 4 * i + (j + a_off)
                U[u_global][col] += u_coeffs[idx]
                
                # V
                j_v = idx // 2
                k_v = idx % 2
                v_global = 4 * (j_v + b_row_off) + (k_v + b_col_off)
                V[v_global][col] += v_coeffs[idx]
                
                # W
                k_w = idx // 2
                i_w = idx % 2
                w_global = 2 * (k_w + b_col_off) + i_w
                W[w_global][col] += w_coeffs[idx]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

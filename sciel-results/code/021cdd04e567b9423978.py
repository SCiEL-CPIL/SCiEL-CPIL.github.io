import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Returns a rank-65 decomposition for 4x4 @ 4x5 matrix multiplication.
    Uses Strassen's algorithm for the 4x4x4 part (rank 49) and schoolbook for the 4x4x1 part (rank 16).
    """
    n, m, p = DIMENSIONS
    # 4x4x4 part: rank 49
    # 4x4x1 part: rank 16
    total_rank = 49 + 16
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2x2 terms
    # U: A indices (0,1,2,3) -> (00, 01, 10, 11)
    # V: B indices (0,1,2,3) -> (00, 01, 10, 11)
    # W: C indices (0,1,2,3) -> (00, 01, 10, 11)
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [1, 0, 0, 0], # M4
        [0, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Build 4x4x4 terms
    # A blocks: 11, 12, 21, 22
    # B blocks: 11, 12, 21, 22
    # C blocks: 11, 12, 21, 22
    
    # For each Strassen term k (0..6):
    # M_k is a product of (linear combo of A blocks) and (linear combo of B blocks)
    # The result is added to C blocks according to strassen_W[k]
    
    # We need to expand each M_k into 7 rank-1 terms for 2x2x2
    
    col = 0
    for k in range(7):
        # strassen_U[k] gives coefficients for A blocks
        # strassen_V[k] gives coefficients for B blocks
        # strassen_W[k] gives coefficients for C blocks
        
        # For each of the 7 sub-terms in the 2x2x2 multiplication:
        for sub_k in range(7):
            # The sub-term is:
            # (sum_i cA_i * A_block_i) * (sum_j cB_j * B_block_j) * (sum_l cC_l * C_block_l)
            # where cA, cB, cC are from strassen_U[k], strassen_V[k], strassen_W[k]
            # Wait, the Strassen formulas are:
            # M_k = (Linear A) * (Linear B)
            # C = Linear(M_k)
            
            # So for a fixed k, we have 7 rank-1 terms for the product of two 2x2 matrices.
            # Let the 7 rank-1 terms for 2x2x2 be T_0..T_6.
            # Each T_s is: U_s * V_s * W_s
            # where U_s, V_s, W_s are vectors of length 4.
            
            # The total term is:
            # (sum_i strassen_U[k][i] * A_block_i) * (sum_j strassen_V[k][j] * B_block_j) * (sum_l strassen_W[k][l] * C_block_l)
            
            # This is not quite right. The Strassen algorithm computes 7 products M_k.
            # Each M_k is a 2x2 matrix.
            # So M_k = A_k * B_k, where A_k and B_k are 2x2 matrices.
            # A_k is a linear combination of A blocks.
            # B_k is a linear combination of B blocks.
            
            # So for a fixed k, we need to decompose the product of two 2x2 matrices (A_k * B_k) into 7 rank-1 terms.
            # Let the 7 rank-1 terms for 2x2x2 be defined by vectors u_s, v_s, w_s (length 4).
            # Then M_k = sum_s u_s^T A_k v_s * w_s^T C_k? No.
            # M_k = sum_s (u_s^T A_k) * (v_s^T B_k) * w_s? No.
            # The tensor is T = sum_s U_s \otimes V_s \otimes W_s.
            # For 2x2x2, T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}.
            # Rank 7 decomposition:
            # T = sum_{s=1}^7 U_s \otimes V_s \otimes W_s.
            
            # So for a fixed k, the contribution to the global tensor is:
            # sum_s (U_s \otimes V_s \otimes W_s) with indices mapped to blocks.
            
            # U_s has entries for the 4 elements of A_k.
            # A_k = sum_i strassen_U[k][i] * A_block_i.
            # So the coefficient for A_block_i in the U part is strassen_U[k][i] * U_s[local_index].
            
            # Similarly for V and W.
            
            # Let's define the 7 rank-1 terms for 2x2x2 explicitly.
            # We already have strassen_U, strassen_V, strassen_W for the 7 terms of 2x2x2?
            # No, those were the 7 terms for the 2x2x2 multiplication itself.
            # So we can just use them directly.
            
            # For a fixed k, the 7 terms are:
            # U_global = sum_i strassen_U[k][i] * (U_s expanded to 16 dims)
            # V_global = sum_j strassen_V[k][j] * (V_s expanded to 16 dims)
            # W_global = sum_l strassen_W[k][l] * (W_s expanded to 16 dims)
            
            # Wait, the Strassen formulas for 4x4x4 are:
            # M_k = A_k * B_k
            # C = sum_k c_k * M_k
            # So the tensor is sum_k c_k * (A_k * B_k).
            # And A_k * B_k is decomposed into 7 rank-1 terms.
            
            # So for each k, we add 7 terms.
            # For each of these 7 terms (s), we have:
            # U_s, V_s, W_s (length 4) from the 2x2x2 decomposition.
            
            # The global U vector for this term is:
            # For each block i (0..3), the entries are strassen_U[k][i] * U_s[local_indices].
            
            # Let's implement this.
            
            for s in range(7):
                # U_s, V_s, W_s are from the 2x2x2 decomposition (the strassen_U/V/W lists above)
                # Wait, the lists above are for the 7 terms of 2x2x2.
                # So U_s = strassen_U[s], etc.
                
                u_s = strassen_U[s]
                v_s = strassen_V[s]
                w_s = strassen_W[s]
                
                # Expand to 16 dims for U, V, W
                # U: 4 blocks, each 2x2 = 4 elements. Total 16.
                # Block i corresponds to rows 2i, 2i+1 and cols 2j, 2j+1?
                # No, block i is a 2x2 matrix.
                # Block 0: rows 0,1 cols 0,1
                # Block 1: rows 0,1 cols 2,3
                # Block 2: rows 2,3 cols 0,1
                # Block 3: rows 2,3 cols 2,3
                
                # Local indices in u_s: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                
                for i in range(4):
                    coeff_u = strassen_U[k][i]
                    if coeff_u == 0: continue
                    # Map local indices to global
                    # Block i:
                    # Local 0 -> Global (2*i//2, 2*i%2)? No.
                    # Block 0: (0,0), (0,1), (1,0), (1,1) -> Global 0,1,4,5
                    # Block 1: (0,2), (0,3), (1,2), (1,3) -> Global 2,3,6,7
                    # Block 2: (2,0), (2,1), (3,0), (3,1) -> Global 8,9,12,13
                    # Block 3: (2,2), (2,3), (3,2), (3,3) -> Global 10,11,14,15
                    
                    # Local 0 -> (r, c) = (2*(i//2), 2*(i%2))
                    # Local 1 -> (r, c+1)
                    # Local 2 -> (r+1, c)
                    # Local 3 -> (r+1, c+1)
                    
                    r_base = 2 * (i // 2)
                    c_base = 2 * (i % 2)
                    
                    # Global index = r * m + c = r * 4 + c
                    global_indices = [
                        r_base * 4 + c_base,
                        r_base * 4 + c_base + 1,
                        (r_base + 1) * 4 + c_base,
                        (r_base + 1) * 4 + c_base + 1
                    ]
                    
                    for local_idx, val in enumerate(u_s):
                        if val != 0:
                            U[global_indices[local_idx]][col] += coeff_u * val
                
                # Similarly for V
                for j in range(4):
                    coeff_v = strassen_V[k][j]
                    if coeff_v == 0: continue
                    r_base = 2 * (j // 2)
                    c_base = 2 * (j % 2)
                    # V is m x p = 4 x 5.
                    # Block j is 2x2.
                    # Global indices in V: r * 5 + c
                    global_indices = [
                        r_base * 5 + c_base,
                        r_base * 5 + c_base + 1,
                        (r_base + 1) * 5 + c_base,
                        (r_base + 1) * 5 + c_base + 1
                    ]
                    for local_idx, val in enumerate(v_s):
                        if val != 0:
                            V[global_indices[local_idx]][col] += coeff_v * val
                
                # Similarly for W
                for l in range(4):
                    coeff_w = strassen_W[k][l]
                    if coeff_w == 0: continue
                    r_base = 2 * (l // 2)
                    c_base = 2 * (l % 2)
                    # W is p x n = 5 x 4.
                    # Block l is 2x2.
                    # Global indices in W: r * 4 + c
                    global_indices = [
                        r_base * 4 + c_base,
                        r_base * 4 + c_base + 1,
                        (r_base + 1) * 4 + c_base,
                        (r_base + 1) * 4 + c_base + 1
                    ]
                    for local_idx, val in enumerate(w_s):
                        if val != 0:
                            W[global_indices[local_idx]][col] += coeff_w * val
                
                col += 1
    
    # Now add the 16 terms for the 5th column
    # C[:, 5] = A * B[:, 5]
    # B[:, 5] is a 4x1 vector.
    # C[:, 5] is a 4x1 vector.
    # C_i = sum_j A_{i,j} B_{j,5}
    # This is 16 multiplications.
    
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

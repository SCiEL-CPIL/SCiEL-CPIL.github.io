"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: Use a block decomposition.
    # Partition 6x6 into 2x2 blocks of size 3x3.
    # Use Strassen's algorithm for the 2x2 block structure.
    # Strassen uses 7 multiplications of 3x3 matrices.
    # Each 3x3 matrix multiplication is decomposed using the schoolbook method (rank 27).
    # Total rank = 7 * 27 = 189.
    # This is an improvement over the full schoolbook rank of 6*6*6 = 216.
    
    # We need to construct U, V, W matrices of shapes:
    # U: (n*m, rank) = (36, 189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    
    rank = 7 * 27  # 189
    
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)
    
    # Strassen's algorithm for 2x2 block matrices:
    # Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]]
    # The 7 products are:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A22 - A12) * (B21 + B22)
    
    # The result C = [[C11, C12], [C21, C22]] is:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Define the 7 Strassen factors as (U_block, V_block, W_block) where:
    # U_block is a 2x2 matrix indicating how to combine A blocks
    # V_block is a 2x2 matrix indicating how to combine B blocks
    # W_block is a 2x2 matrix indicating how to combine C blocks (result)
    
    # For each Strassen factor s (0..6), we have:
    # U_s: 2x2 matrix of coefficients for A blocks
    # V_s: 2x2 matrix of coefficients for B blocks
    # The result C_block_ij gets contribution from W_s[i][j] * M_s
    
    # Let's define the U and V patterns for the 7 Strassen multiplications.
    # Each multiplication is of the form (linear combo of A blocks) * (linear combo of B blocks).
    
    strassen_U = [
        [[1, 0], [0, 1]],  # M1: (A11 + A22)
        [[0, 0], [1, 1]],  # M2: (A21 + A22)
        [[1, 0], [0, 0]],  # M3: A11
        [[0, 0], [0, 1]],  # M4: A22
        [[1, 1], [0, 0]],  # M5: (A11 + A12)
        [[-1, 0], [1, 0]], # M6: (A21 - A11)
        [[0, -1], [0, 1]], # M7: (A22 - A12)
    ]
    
    strassen_V = [
        [[1, 0], [0, 1]],  # M1: (B11 + B22)
        [[1, 0], [0, 0]],  # M2: B11
        [[0, 1], [0, -1]], # M3: (B12 - B22)
        [[-1, 0], [1, 0]], # M4: (B21 - B11)
        [[0, 0], [0, 1]],  # M5: B22
        [[1, 1], [0, 0]],  # M6: (B11 + B12)
        [[0, 0], [1, 1]],  # M7: (B21 + B22)
    ]
    
    strassen_W = [
        [[1, 0], [0, 1]],  # M1 contributes to C11 and C22
        [[0, 0], [1, 0]],  # M2 contributes to C21
        [[0, 1], [0, 0]],  # M3 contributes to C12
        [[1, 0], [0, 1]],  # M4 contributes to C11 and C22
        [[-1, 0], [0, 1]], # M5 contributes to C11 and C22
        [[0, 0], [0, 1]],  # M6 contributes to C22
        [[1, 0], [0, 0]],  # M7 contributes to C11
    ]
    
    # Now, for each Strassen factor s, we need to expand the 3x3 matrix multiplication.
    # A 3x3 matrix multiplication has 27 terms in the schoolbook method.
    # For each term (i, j, k) in the 3x3 multiplication, we add a column to U, V, W.
    
    block_size = 3
    
    for s in range(7):
        U_s = strassen_U[s]
        V_s = strassen_V[s]
        W_s = strassen_W[s]
        
        # The Strassen multiplication s computes:
        # M_s = (sum_{a,b} U_s[a][b] * A_ab) * (sum_{c,d} V_s[c][d] * B_cd)
        
        # Each term in the 3x3 multiplication of M_s is:
        # (U_s[a][b] * A_ab[i][j]) * (V_s[c][d] * B_cd[j][k])
        # This contributes to C_ad[i][k] with weight W_s[a][c] * (U_s[a][b] * V_s[c][d])
        
        # Wait, the Strassen algorithm says:
        # C_ad = sum_s W_s[a][c] * M_s
        # where M_s = (U_s A) (V_s B)
        
        # So, M_s[i][k] = sum_{j} (U_s A)[i][j] * (V_s B)[j][k]
        # (U_s A)[i][j] = sum_b U_s[0][b] * A_0b[i][j] if i in block 0, etc.
        
        # Let's be more precise.
        # Let A be partitioned into 2x2 blocks A_00, A_01, A_10, A_11.
        # Let B be partitioned into 2x2 blocks B_00, B_01, B_10, B_11.
        # Let C be partitioned into 2x2 blocks C_00, C_01, C_10, C_11.
        
        # M_s = (U_s * A) * (V_s * B)
        # where U_s and V_s are 2x2 matrices of scalars.
        
        # (U_s * A)[i][j] = sum_b U_s[0][b] * A_0b[i_mod][j_mod] if i in block 0
        #                   = sum_b U_s[1][b] * A_1b[i_mod][j_mod] if i in block 1
        
        # (V_s * B)[j][k] = sum_d V_s[0][d] * B_0d[j_mod][k_mod] if j in block 0
        #                   = sum_d V_s[1][d] * B_1d[j_mod][k_mod] if j in block 1
        
        # C[i][k] = sum_s W_s[i_block][k_block] * M_s[i_mod][k_mod]
        
        # For a specific term in the 3x3 multiplication of M_s:
        # M_s[i_mod][k_mod] includes the term:
        # (U_s[i_block][b] * A_i_block, b[i_mod][j_mod]) * (V_s[j_block][d] * B_j_block, d[j_mod][k_mod])
        
        # This contributes to C[i_block, k_block][i_mod][k_mod] with weight:
        # W_s[i_block][k_block] * U_s[i_block][b] * V_s[j_block][d]
        
        # So, for each s, i_block, j_block, k_block, b, d, i_mod, j_mod, k_mod:
        # if U_s[i_block][b] != 0 and V_s[j_block][d] != 0 and W_s[i_block][k_block] != 0:
        #   Add a term to the tensor decomposition.
        
        # The term corresponds to:
        # U[i, j] = U_s[i_block][b]
        # V[j, k] = V_s[j_block][d]
        # W[k, i] = W_s[i_block][k_block] * U_s[i_block][b] * V_s[j_block][d]
        
        # Wait, the tensor is T[i][j][k] = C[i][k] / (A[i][j] * B[j][k])? No.
        # The matrix multiplication tensor T is defined by:
        # C[i][k] = sum_j A[i][j] B[j][k]
        
        # In the decomposition:
        # C[i][k] = sum_r U[i, r] * V[j, r] * W[k, r]  (This is not the standard form)
        
        # The standard form is:
        # C[i][k] = sum_r U[i, r] * V[k, r] * W[r, j] ? No.
        
        # Let's use the definition from the task:
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # C[i][k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So, the term for a specific (i, j, k, r) is:
        # U[i*m+j, r] = a
        # V[j*p+k, r] = b
        # W[k*n+i, r] = c
        # contributes a * b * c to C[i][k] via the path j.
        
        # In our block decomposition:
        # i = i_block * 3 + i_mod
        # j = j_block * 3 + j_mod
        # k = k_block * 3 + k_mod
        
        # The contribution is:
        # U_s[i_block][b] * V_s[j_block][d] * W_s[i_block][k_block]
        
        # But we need to sum over b and d.
        # M_s[i_mod][k_mod] = sum_{j_block, b, d} (U_s[i_block][b] * A_i_block, b[i_mod][j_mod]) * (V_s[j_block][d] * B_j_block, d[j_mod][k_mod])
        
        # Wait, the indices are mixed.
        
        # Let's define the column index r for each term.
        # For each s, i_block, j_block, b, d, i_mod, j_mod, k_mod:
        # if U_s[i_block][b] != 0 and V_s[j_block][d] != 0:
        #   r = current_rank
        #   U[i*m+j, r] = U_s[i_block][b]
        #   V[j*p+k, r] = V_s[j_block][d]
        #   W[k*n+i, r] = W_s[i_block][k_block] * U_s[i_block][b] * V_s[j_block][d]
        
        # This will work.
        
        current_rank = 0
        for s in range(7):
            U_s = strassen_U[s]
            V_s = strassen_V[s]
            W_s = strassen_W[s]
            
            for i_block in range(2):
                for j_block in range(2):
                    for b in range(2):
                        for d in range(2):
                            for i_mod in range(3):
                                for j_mod in range(3):
                                    for k_mod in range(3):
                                        u_val = U_s[i_block][b]
                                        v_val = V_s[j_block][d]
                                        
                                        if u_val == 0 or v_val == 0:
                                            continue
                                        
                                        # Find the corresponding k_block such that W_s[i_block][k_block] != 0
                                        # Actually, W_s[i_block][k_block] is a scalar.
                                        # We need to iterate over k_block too.
                                        # But in Strassen, each M_s contributes to specific C blocks.
                                        
                                        # Let's iterate over k_block as well.
                                        pass
        
        # Let's restructure the loop.
        
        current_rank = 0
        for s in range(7):
            U_s = strassen_U[s]
            V_s = strassen_V[s]
            W_s = strassen_W[s]
            
            for i_block in range(2):
                for k_block in range(2):
                    w_val = W_s[i_block][k_block]
                    if w_val == 0:
                        continue
                    
                    for j_block in range(2):
                        for b in range(2):
                            u_val = U_s[i_block][b]
                            if u_val == 0:
                                continue
                            
                            for d in range(2):
                                v_val = V_s[j_block][d]
                                if v_val == 0:
                                    continue
                                
                                for i_mod in range(3):
                                    for j_mod in range(3):
                                        for k_mod in range(3):
                                            i = i_block * 3 + i_mod
                                            j = j_block * 3 + j_mod
                                            k = k_block * 3 + k_mod
                                            
                                            U[i*m + j, current_rank] = u_val
                                            V[j*p + k, current_rank] = v_val
                                            W[k*n + i, current_rank] = w_val * u_val * v_val
                                            
                                            current_rank += 1
                            
    # The total rank should be 189.
    # Let's verify the rank.
    # If current_rank != 189, there might be an error.
    
    # Convert to lists for the return format.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

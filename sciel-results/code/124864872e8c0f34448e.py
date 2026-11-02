"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-49 tensor decomposition for 4x4 matrix multiplication
    using the recursive Strassen algorithm.
    """
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # W_scalar[m][k] is the coefficient of M_m in C_k for 2x2
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_scalar = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # U_scalar[m][idx] coefficient for A_idx in M_m
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    U_scalar = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    # V_scalar[m][idx] coefficient for B_idx in M_m
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    V_scalar = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    # Block level Strassen
    # M_block_1 = (A00 + A11)(B00 + B11)
    # M_block_2 = (A10 + A11)B00
    # M_block_3 = A00(B01 - B11)
    # M_block_4 = A11(B10 - B00)
    # M_block_5 = (A00 + A01)B11
    # M_block_6 = (A10 - A00)(B00 + B01)
    # M_block_7 = (A01 - A11)(B10 + B11)
    
    # W_block[m][k] coefficient of M_block_m in C_block_k
    W_block = [
        [1, 0, 0, 1, -1, 0, 1], # C_block_00
        [0, 0, 1, 0, 1, 0, 0],  # C_block_01
        [0, 1, 0, 1, 0, 0, 0],  # C_block_10
        [1, -1, 1, 0, 0, 1, 0]  # C_block_11
    ]
    
    # U_block[m][block_idx] coefficient for A_block in M_block_m
    # Block indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    U_block = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    # V_block[m][block_idx] coefficient for B_block in M_block_m
    V_block = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    # Map block indices to scalar indices
    # A_block_00 corresponds to A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,4,5
    # A_block_01 corresponds to A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2,3,6,7
    # A_block_10 corresponds to A[2,0], A[2,1], A[3,0], A[3,1] -> indices 8,9,12,13
    # A_block_11 corresponds to A[2,2], A[2,3], A[3,2], A[3,3] -> indices 10,11,14,15
    
    # B_block_00 corresponds to B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,4,5
    # B_block_01 corresponds to B[0,2], B[0,3], B[1,2], B[1,3] -> indices 2,3,6,7
    # B_block_10 corresponds to B[2,0], B[2,1], B[3,0], B[3,1] -> indices 8,9,12,13
    # B_block_11 corresponds to B[2,2], B[2,3], B[3,2], B[3,3] -> indices 10,11,14,15
    
    # C_block_00 corresponds to C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,4,5
    # C_block_01 corresponds to C[0,2], C[0,3], C[1,2], C[1,3] -> indices 2,3,6,7
    # C_block_10 corresponds to C[2,0], C[2,1], C[3,0], C[3,1] -> indices 8,9,12,13
    # C_block_11 corresponds to C[2,2], C[2,3], C[3,2], C[3,3] -> indices 10,11,14,15
    
    def get_block_indices(block_idx):
        # block_idx 0: 00, 1: 01, 2: 10, 3: 11
        # Returns list of 4 scalar indices
        r = block_idx // 2
        c = block_idx % 2
        base = r * 8 + c * 2
        return [base, base + 1, base + 4, base + 5]
    
    col = 0
    for mb in range(7):
        for ms in range(7):
            # This scalar term ms contributes to block term mb
            # The coefficient of this scalar term in the block term is 1 (since we expand the block term into 7 scalar terms)
            # Wait, the block term M_block_mb is a matrix.
            # M_block_mb = U_block_mb (linear comb of A blocks) * V_block_mb (linear comb of B blocks)
            # This product is computed using Strassen, so it is a sum of 7 scalar products.
            # The scalar products are M_scalar_ms.
            # The coefficients of M_scalar_ms in M_block_mb are determined by the Strassen structure.
            # Actually, M_block_mb is the result of multiplying two 2x2 matrices.
            # Let X = U_block_mb applied to A. Y = V_block_mb applied to B.
            # M_block_mb = X * Y.
            # X and Y are 2x2 matrices.
            # We compute X*Y using Strassen.
            # The 7 terms of Strassen for X*Y are:
            # S1 = (X00+X11)(Y00+Y11)
            # ...
            # Each S_k is a scalar product of linear combinations of X entries and Y entries.
            # X entries are linear combinations of A entries.
            # Y entries are linear combinations of B entries.
            
            # So for each scalar term ms in the Strassen expansion of X*Y:
            # S_ms = (sum u_x X_ij) * (sum v_y Y_ij)
            # We need to expand X_ij and Y_ij in terms of A and B.
            
            # Let's precompute the coefficients of A and B for each X_ij and Y_ij.
            # X is a 2x2 matrix. X_ij = sum_{block} U_block[mb][block] * A_block
            # A_block is 2x2.
            
            # This is getting complex to implement generically.
            # Let's simplify:
            # We iterate over the 49 scalar multiplications.
            # Each scalar multiplication is defined by:
            # - The block term mb (0..6)
            # - The scalar term ms (0..6) within the Strassen expansion of the block term.
            
            # The block term mb corresponds to:
            # A_part = sum_{b} U_block[mb][b] A_b
            # B_part = sum_{b} V_block[mb][b] B_b
            # M_block_mb = A_part * B_part
            
            # We compute A_part * B_part using Strassen.
            # The Strassen terms for A_part * B_part are:
            # S0 = (A_part00 + A_part11)(B_part00 + B_part11)
            # ...
            
            # A_part00 = sum_{b} U_block[mb][b] * (A_b)00
            # (A_b)00 is A[2*b_r, 2*b_c]
            
            # So A_part00 is a linear combination of A entries.
            # We can compute the coefficients of A entries in A_part00, etc.
            
            # Let's do this step by step.
            
            # 1. Compute coefficients of A entries in A_part (2x2 matrix)
            # A_part is 2x2.
            # A_part[i][j] = sum_{b} U_block[mb][b] * A_b[i][j]
            # A_b[i][j] is A[2*(b//2) + i][2*(b%2) + j]
            
            coeff_A_part = [[0]*4 for _ in range(4)] # 2x2 matrix of coefficients for A indices? No.
            # Let's store coefficients for each of the 16 A entries.
            # coeff_A_part[i][j] is a list of 16 coefficients?
            # No, A_part[i][j] is a linear combination of A entries.
            # Let's store it as a vector of 16 coefficients.
            
            # Actually, we can just compute the final U, V, W contributions directly.
            
            # For a given scalar term ms in the Strassen expansion of A_part * B_part:
            # S_ms = (sum_{i,j} u_A[ms][i,j] A_part[i,j]) * (sum_{i,j} v_B[ms][i,j] B_part[i,j])
            # Where u_A[ms][i,j] and v_B[ms][i,j] are the Strassen coefficients for the 2x2 matrices.
            
            # We know the Strassen coefficients for 2x2 matrices (U_scalar, V_scalar).
            # U_scalar[ms][k] is the coefficient of A_k in the first factor of S_ms.
            # Here A_k are the entries of A_part.
            
            # So S_ms = (sum_{k} U_scalar[ms][k] A_part[k]) * (sum_{k} V_scalar[ms][k] B_part[k])
            
            # A_part[k] is a linear combination of A entries.
            # A_part[k] = sum_{a_idx} coeff_A_part[k][a_idx] A[a_idx]
            
            # So S_ms = (sum_{a_idx} (sum_{k} U_scalar[ms][k] coeff_A_part[k][a_idx]) A[a_idx]) * 
            #            (sum_{b_idx} (sum_{k} V_scalar[ms][k] coeff_B_part[k][b_idx]) B[b_idx])
            
            # We can compute the combined coefficients for A and B.
            
            # coeff_A_part[k] for k in 0..3 (entries of A_part)
            # A_part is 2x2.
            # A_part[0,0] = sum_{b} U_block[mb][b] * A_b[0,0]
            # A_b[0,0] is A[2*(b//2), 2*(b%2)]
            
            coeff_A_part = [[0]*16 for _ in range(4)]
            coeff_B_part = [[0]*16 for _ in range(4)]
            
            for b in range(4):
                u_b = U_block[mb][b]
                v_b = V_block[mb][b]
                
                # A_b is 2x2 block b
                # Indices in A:
                # b=0 (00): 0,1,4,5
                # b=1 (01): 2,3,6,7
                # b=2 (10): 8,9,12,13
                # b=3 (11): 10,11,14,15
                
                # A_b[0,0] -> A[2*(b//2), 2*(b%2)]
                # A_b[0,1] -> A[2*(b//2), 2*(b%2)+1]
                # A_b[1,0] -> A[2*(b//2)+1, 2*(b%2)]
                # A_b[1,1] -> A[2*(b//2)+1, 2*(b%2)+1]
                
                r = b // 2
                c = b % 2
                base = r * 8 + c * 2
                
                # A_part[0,0] (k=0)
                coeff_A_part[0][base] += u_b
                # A_part[0,1] (k=1)
                coeff_A_part[1][base + 1] += u_b
                # A_part[1,0] (k=2)
                coeff_A_part[2][base + 4] += u_b
                # A_part[1,1] (k=3)
                coeff_A_part[3][base + 5] += u_b
                
                # B_part[0,0] (k=0)
                coeff_B_part[0][base] += v_b
                # B_part[0,1] (k=1)
                coeff_B_part[1][base + 1] += v_b
                # B_part[1,0] (k=2)
                coeff_B_part[2][base + 4] += v_b
                # B_part[1,1] (k=3)
                coeff_B_part[3][base + 5] += v_b
            
            # Now compute the final coefficients for A and B for this scalar term ms
            # final_U[a_idx] = sum_{k} U_scalar[ms][k] * coeff_A_part[k][a_idx]
            # final_V[b_idx] = sum_{k} V_scalar[ms][k] * coeff_B_part[k][b_idx]
            
            final_U = [0]*16
            final_V = [0]*16
            
            for k in range(4):
                u_k = U_scalar[ms][k]
                v_k = V_scalar[ms][k]
                for a_idx in range(16):
                    final_U[a_idx] += u_k * coeff_A_part[k][a_idx]
                for b_idx in range(16):
                    final_V[b_idx] += v_k * coeff_B_part[k][b_idx]
            
            # Now we have the linear combinations for A and B for this scalar multiplication.
            # We add them to U and V.
            for a_idx in range(16):
                if final_U[a_idx] != 0:
                    U[a_idx][col] = final_U[a_idx]
            for b_idx in range(16):
                if final_V[b_idx] != 0:
                    V[b_idx][col] = final_V[b_idx]
            
            # Now we need to update W.
            # This scalar term ms contributes to M_block_mb.
            # M_block_mb contributes to C_block_k with coefficient W_block[k][mb].
            # C_block_k is a 2x2 matrix.
            # C_block_k[i,j] gets contribution W_block[k][mb] * (M_block_mb)[i,j].
            # (M_block_mb)[i,j] is computed using Strassen.
            # (M_block_mb)[i,j] = sum_{ms} W_scalar[i*2+j][ms] * S_ms.
            # So C_block_k[i,j] gets contribution W_block[k][mb] * W_scalar[i*2+j][ms] * S_ms.
            
            # So for each C entry in C_block_k, we add the coefficient.
            # C_block_k corresponds to C indices.
            # k=0 (00): C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,4,5
            # k=1 (01): C[0,2], C[0,3], C[1,2], C[1,3] -> 2,3,6,7
            # k=2 (10): C[2,0], C[2,1], C[3,0], C[3,1] -> 8,9,12,13
            # k=3 (11): C[2,2], C[2,3], C[3,2], C[3,3] -> 10,11,14,15
            
            for k in range(4):
                w_block = W_block[k][mb]
                if w_block == 0:
                    continue
                r = k // 2
                c = k % 2
                base = r * 8 + c * 2
                
                # C_block_k[0,0] -> C[base]
                # C_block_k[0,1] -> C[base+1]
                # C_block_k[1,0] -> C[base+4]
                # C_block_k[1,1] -> C[base+5]
                
                # W_scalar[i*2+j][ms]
                # i,j in 0..1 for C_block_k entries
                
                # C[base] (i=0, j=0)
                W[base][col] += w_block * W_scalar[0][ms]
                # C[base+1] (i=0, j=1)
                W[base + 1][col] += w_block * W_scalar[1][ms]
                # C[base+4] (i=1, j=0)
                W[base + 4][col] += w_block * W_scalar[2][ms]
                # C[base+5] (i=1, j=1)
                W[base + 5][col] += w_block * W_scalar[3][ms]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

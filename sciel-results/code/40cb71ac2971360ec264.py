import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen 2x2x2 coefficients
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # Terms: M0..M6
    
    # U coefficients (A side) for each term s
    # s: 0, 1, 2, 3, 4, 5, 6
    # A00, A01, A10, A11
    alpha_coeffs = [
        [1, 0, 0, 1],  # M0: A00 + A11
        [0, 0, 1, 1],  # M1: A10 + A11
        [1, 0, 0, 0],  # M2: A00
        [0, 0, 0, 1],  # M3: A11
        [1, 1, 0, 0],  # M4: A00 + A01
        [-1, 0, 1, 0], # M5: A10 - A00
        [0, 1, 0, -1]  # M6: A01 - A11
    ]
    
    # V coefficients (B side) for each term s
    beta_coeffs = [
        [1, 0, 0, 1],  # M0: B00 + B11
        [1, 0, 0, 0],  # M1: B00
        [0, 1, 0, -1], # M2: B01 - B11
        [-1, 0, 1, 0], # M3: B10 - B00
        [0, 0, 0, 1],  # M4: B11
        [1, 1, 0, 0],  # M5: B00 + B01
        [0, 0, 1, 1]   # M6: B10 + B11
    ]
    
    # W reconstruction: C blocks (00, 01, 10, 11)
    # C00 = M0 - M4 + M6
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M0 - M1 + M2 + M5
    # Indices in W: C_block * 4 + term_index? No, W is p*n x R.
    # W shape (24, 77). Rows are output indices (r, c) where r in 0..5, c in 0..3?
    # Task says W[k*n+i, r] where k is row of C (0..5), i is col of C (0..3).
    # So W is 24 x R.
    # For a 2x2x2 Strassen block, output is 2x2.
    # We need to map the 2x2 output of the inner Strassen to the appropriate 2x2 or 2x3 block of the final output.
    
    # The current program uses a specific nesting:
    # Top level: Strassen 2x2x2 for the 4x4 A and 4x6 B?
    # A is 4x4 -> 4 blocks 2x2.
    # B is 4x6 -> 4 blocks 2x3? (Rows 0-1, 2-3; Cols 0-5).
    # Wait, B is 4x6. Splitting into 2x3 blocks:
    # B00 (2x3), B01 (2x3)
    # B10 (2x3), B11 (2x3)
    # So top level operates on 2x2 grid of A blocks and 2x2 grid of B blocks (each 2x3).
    # The product of a 2x2 block of A and a 2x3 block of B is a 2x3 matrix.
    # The top level Strassen computes 7 linear combinations of A blocks (2x2) and B blocks (2x3).
    # X_s is 2x2, Y_s is 2x3.
    # The product X_s Y_s is 2x3.
    # Then we need to decompose the 2x2 x 2x3 multiplication into rank terms.
    
    # The current program decomposes X_s (2x2) * Y_s (2x3) as:
    # Inner Strassen for Y_s[:, 0:2] (2x2 part) + Schoolbook for Y_s[:, 2:3] (2x1 part? No, Y is 2x3, so cols 0,1 and col 2).
    # Y_s is 2x3.
    # Part 1: X_s * Y_s[:, 0:2] (2x2 * 2x2). Rank 7.
    # Part 2: X_s * Y_s[:, 2:3] (2x2 * 2x1). Rank 4 (Schoolbook).
    # Total rank per top-level term: 7 + 4 = 11.
    # Total rank: 7 * 11 = 77.
    
    # We will replicate this logic cleanly.
    
    col = 0
    for s in range(7):
        # X_s is 2x2, Y_s is 2x3
        # We compute X_s * Y_s
        # Split Y_s into Y_left (2x2) and Y_right (2x1)
        
        # Part 1: X_s * Y_left (2x2 * 2x2) using Strassen
        for si in range(7):
            if time.time() > deadline: break
            
            # U: A contribution
            # X_s is formed by alpha[s] * A_blocks.
            # X_s itself is multiplied by Y_left via Strassen si.
            # So U gets alpha[s] * alpha_inner[si]
            # Indices:
            # A blocks are 2x2.
            # A_block (ar, br) is at rows (ar*2 .. ar*2+1) and cols (br*2 .. br*2+1) in A (4x4).
            # Wait, A is 4x4. Blocks are:
            # A00: rows 0-1, cols 0-1
            # A01: rows 0-1, cols 2-3
            # A10: rows 2-3, cols 0-1
            # A11: rows 2-3, cols 2-3
            
            # The inner Strassen si computes M_si = X_s_inner * Y_s_inner_left
            # X_s_inner is a 2x2 matrix formed from X_s? No.
            # The inner Strassen is applied to the 2x2 matrix X_s and the 2x2 matrix Y_left.
            # So the "A" of the inner Strassen is X_s, and "B" is Y_left.
            # But X_s is not a linear combination of A blocks in the inner step?
            # No, X_s IS a linear combination of A blocks.
            # So U[row, col] = sum_{a_block} (coeff_from_top_s[a_block] * coeff_from_inner_si[a_block])
            
            for a in range(4):
                coeff_top = alpha_coeffs[s][a]
                coeff_inner = alpha_coeffs[si][a]
                if coeff_top and coeff_inner:
                    val = coeff_top * coeff_inner
                    ar, br = a // 2, a % 2
                    # A block (ar, br) occupies rows 2*ar .. 2*ar+1, cols 2*br .. 2*br+1
                    for r_local in range(2):
                        for c_local in range(2):
                            u_idx = (2*ar + r_local) * 4 + (2*br + c_local)
                            U[u_idx, col] += val
            
            # V: B contribution
            # Y_s is formed by beta[s] * B_blocks.
            # Y_left is the first 2 cols of Y_s.
            # B blocks are 2x3.
            # B_block (ac, bc) is at rows (ac*2 .. ac*2+1) and cols (bc*3 .. bc*3+2) in B (4x6).
            # Wait, B is 4x6.
            # B00: rows 0-1, cols 0-2
            # B01: rows 0-1, cols 3-5
            # B10: rows 2-3, cols 0-2
            # B11: rows 2-3, cols 3-5
            
            # Y_s is 2x3.
            # Y_s[i, k] = sum_{b_block} beta[s][b_block] * B[b_block][i, k]
            # Y_left[i, k] for k in 0,1 is sum_{b_block} beta[s][b_block] * B[b_block][i, k]
            
            # Inner Strassen si uses Y_left as "B".
            # So V gets beta_top[s] * beta_inner[si]
            # But we need to map the 2x2 B of inner Strassen to the 2x3 B of top level?
            # The inner Strassen "B" is Y_left (2x2).
            # Y_left is derived from B blocks (2x3) by taking first 2 cols.
            
            for b in range(4):
                coeff_top = beta_coeffs[s][b]
                coeff_inner = beta_coeffs[si][b]
                if coeff_top and coeff_inner:
                    val = coeff_top * coeff_inner
                    ac, bc = b // 2, b % 2
                    # B block (ac, bc) occupies rows 2*ac .. 2*ac+1, cols 3*bc .. 3*bc+2
                    # We only use the first 2 cols of this block for Y_left?
                    # Y_s is 2x3. The "inner Strassen" applies to the 2x2 part.
                    # Which 2 cols of the 2x3 B block are used?
                    # The top level B is 4x6.
                    # Y_s is 2x3.
                    # Y_s[:, 0:2] uses the first 2 columns of the corresponding B blocks.
                    # So for B block (ac, bc), we use cols 3*bc and 3*bc+1.
                    
                    for r_local in range(2):
                        for c_local in range(2):
                            v_idx = (2*ac + r_local) * 6 + (3*bc + c_local)
                            V[v_idx, col] += val
            
            # W: Output contribution
            # Inner Strassen si produces 2x2 result for X_s * Y_left.
            # Top level Strassen s produces 2x2 result for "blocks" of A and B.
            # The 2x2 result of inner si corresponds to the top-left 2x2 of the 2x3 result of term s?
            # Term s result is 2x3.
            # Part 1 (inner Strassen) computes the first 2 cols of the 2x3 result.
            # So the 2x2 output of inner si is placed in cols 0,1 of the 2x3 block of C for term s.
            
            # Top level recon:
            # C_block (a, b) gets contribution from term s with coeff gamma.
            # C is 4x6.
            # C blocks are 2x3.
            # C00: rows 0-1, cols 0-2
            # C01: rows 0-1, cols 3-5
            # C10: rows 2-3, cols 0-2
            # C11: rows 2-3, cols 3-5
            
            # Inner recon:
            # Inner C_block (za, zb) gets contribution from inner term si with coeff zgamma.
            # Inner C is 2x2.
            # za, zb in 0,1.
            
            # So W index:
            # Row: 2*a + za
            # Col: 3*b + zb
            # (Because C block (a,b) is at rows 2a.., cols 3b..)
            
            # Top recon coefficients:
            top_recon = [
                [1, 0, 0, 1],  # C00: M0 + M3? No.
                # Standard Strassen:
                # C00 = M0 - M4 + M6
                # C01 = M2 + M4
                # C10 = M3 + M5
                # C11 = M0 - M1 + M2 + M5
                # My alpha/beta order:
                # 0: M0, 1: M1, 2: M2, 3: M3, 4: M4, 5: M5, 6: M6
                # C00 (a=0,b=0): M0 - M4 + M6 -> [1, 0, 0, 0, -1, 0, 1]
                # C01 (a=0,b=1): M2 + M4 -> [0, 0, 1, 0, 1, 0, 0]
                # C10 (a=1,b=0): M3 + M5 -> [0, 0, 0, 1, 0, 1, 0]
                # C11 (a=1,b=1): M0 - M1 + M2 + M5 -> [1, -1, 1, 0, 0, 1, 0]
            ]
            top_recon_matrix = [
                [1, 0, 0, 0, -1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
                [1, -1, 1, 0, 0, 1, 0]
            ]
            
            inner_recon_matrix = [
                [1, 0, 0, 0, -1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
                [1, -1, 1, 0, 0, 1, 0]
            ]
            
            for a_top in range(2):
                for b_top in range(2):
                    gamma_top = top_recon_matrix[a_top * 2 + b_top][s]
                    if gamma_top:
                        for za in range(2):
                            for zb in range(2):
                                zgamma = inner_recon_matrix[za * 2 + zb][si]
                                if zgamma:
                                    w_row = 2 * a_top + za
                                    w_col = 3 * b_top + zb
                                    W[w_row * 4 + w_col, col] += gamma_top * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for X_s * Y_right (2x2 * 2x1)
        # Y_right is col 2 of Y_s.
        # Y_s is 2x3. Col 2 corresponds to col index 2 in the 2x3 block.
        # In the full B (4x6), B block (ac, bc) has cols 3*bc, 3*bc+1, 3*bc+2.
        # So Y_right uses col 3*bc+2 of each B block.
        
        for u_idx_local in range(2): # row of X_s (2x2)
            for v_idx_local in range(2): # row of Y_right (2x1)
                if time.time() > deadline: break
                
                # U: X_s * e_u?
                # X_s is 2x2. Y_right is 2x1.
                # Term (u, v) of product: X[u, :] * Y_right[:, v]? No, Y is 2x1.
                # Product C[u, 0] = sum_v X[u, v] Y[v, 0].
                # Here "v" is the inner index (row of Y_right).
                
                # U contribution:
                # X_s is formed by alpha[s] * A_blocks.
                # X_s[u, v] = sum_a alpha[s][a] * A_block_a[u, v]
                # We need U for the term corresponding to X[u, v] * Y[v, 0].
                # This is a rank 1 term: (X[:, u]) * (Y_right[v, :])?
                # No, it's sum over v of X[u, v] * Y[v, 0].
                # So for fixed u, v:
                # U gets alpha[s][a] * A_block_a[u, v]
                # V gets beta[s][b] * B_block_b[v, 2]
                # W gets C_block[u, 0] contribution.
                
                for a in range(4):
                    coeff = alpha_coeffs[s][a]
                    if coeff:
                        ar, br = a // 2, a % 2
                        u_idx = (2*ar + u_idx_local) * 4 + (2*br + v_idx_local)
                        U[u_idx, col] += coeff
                
                for b in range(4):
                    coeff = beta_coeffs[s][b]
                    if coeff:
                        ac, bc = b // 2, b % 2
                        # Y_right is col 2 of the 2x3 block.
                        # So col index in B is 3*bc + 2.
                        v_idx = (2*ac + v_idx_local) * 6 + (3*bc + 2)
                        V[v_idx, col] += coeff
                
                # W contribution:
                # The result is placed in C_block (a_top, b_top) at position (u_idx_local, 2).
                # Because Part 2 computes the 3rd column (index 2) of the 2x3 result.
                for a_top in range(2):
                    for b_top in range(2):
                        gamma_top = top_recon_matrix[a_top * 2 + b_top][s]
                        if gamma_top:
                            w_row = 2 * a_top + u_idx_local
                            w_col = 3 * b_top + 2
                            W[w_row * 4 + w_col, col] += gamma_top
                
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a10+a11)*b00
    # M2 = a11*(b01-b00)
    # M3 = a00*(b11-b10)
    # M4 = (a00+a01)*b11
    # M5 = (a01+a11)*(b00+b10)
    # M6 = (a00-a10)*(b10+b11)
    # M7 = (a01+a10)*(b01-b10)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M3 - M2 + M4 + M5
    
    # We need to map local indices (0,1) to global indices.
    # Local A: rows i_loc (0,1) -> global i (0,1)
    # Local A: cols j_loc (0,1) -> global j (j_start + j_loc)
    # Local B: rows j_loc (0,1) -> global j (j_start + j_loc)
    # Local B: cols k_loc (0,1) -> global k (k_start + k_loc)
    # Local C: rows i_loc (0,1) -> global i (0,1)
    # Local C: cols k_loc (0,1) -> global k (k_start + k_loc)
    
    def add_strassen_block(j_start, k_start, col_offset):
        nonlocal col
        # M1
        # U: a10, a11 -> U[i*4 + j_start+1]
        # V: b00 -> V[(j_start+0)*4 + k_start+0]
        # W: c00, c10 -> W[k_start+0]*2+0, W[k_start+0]*2+1
        # M1 contributes to c00 (+1), c10 (+1)
        
        # Helper to add a term
        def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
            nonlocal col
            if u_val == 0 and v_val == 0 and w_val == 0:
                return
            U[u_idx][col] = u_val
            V[v_idx][col] = v_val
            W[w_idx][col] = w_val
            col += 1
            
        # This simple helper adds one rank-1 term.
        # But Strassen terms are sums of products.
        # M1 = (a10+a11) * b00
        # This is NOT a single product of a linear form in A and linear form in B?
        # Yes it is. U part: a10+a11. V part: b00.
        # So for M1:
        # U[0*4 + j_start+1] = 1 (a10)
        # U[1*4 + j_start+1] = 1 (a11)
        # V[(j_start+0)*4 + k_start+0] = 1 (b00)
        # W[k_start+0]*2+0 = 1 (c00)
        # W[k_start+0]*2+1 = 1 (c10)
        
        # Wait, U index is i*m + j.
        # i=0, j=j_start+1 -> U[j_start+1]
        # i=1, j=j_start+1 -> U[4+j_start+1]
        
        # Let's list the 7 terms.
        
        # Term 1: M1 = (a10+a11) b00
        # U: [0,1] at j=j1, [1,1] at j=j1? No.
        # a10 is A[1, j1]. a11 is A[1, j1+1]? No, local indices.
        # Local A:
        # a00 = A[0, j0]
        # a01 = A[0, j1]
        # a10 = A[1, j0]
        # a11 = A[1, j1]
        # where j0 = j_start, j1 = j_start+1.
        
        # M1 = (a10 + a11) * b00
        # U[1*4 + j0] += 1
        # U[1*4 + j1] += 1
        # V[j0*4 + k0] += 1
        # W[k0*2 + 0] += 1
        # W[k0*2 + 1] += 1
        
        # We need to accumulate these into the global U, V, W for the current column.
        # Since we are building a new column for each Strassen term, we just set the values.
        
        # Term 1
        U[1*4 + j_start][col] = 1
        U[1*4 + j_start+1][col] = 1
        V[j_start*4 + k_start][col] = 1
        W[k_start*2 + 0][col] = 1
        W[k_start*2 + 1][col] = 1
        col += 1
        
        # Term 2: M2 = a11 * (b01 - b00)
        U[1*4 + j_start+1][col] = 1
        V[j_start*4 + k_start+1][col] = 1
        V[j_start*4 + k_start][col] = -1
        W[k_start*2 + 0][col] = 0 # c00 doesn't get M2
        W[k_start*2 + 1][col] = 0
        # c01 = M2 + M4 -> W[k1*2 + 0] = 1
        # c11 = ... - M2 -> W[k1*2 + 1] = -1
        W[k_start+1*2 + 0][col] = 1
        W[k_start+1*2 + 1][col] = -1
        col += 1
        
        # Term 3: M3 = a00 * (b11 - b10)
        U[0*4 + j_start][col] = 1
        V[j_start+1*4 + k_start+1][col] = 1
        V[j_start+1*4 + k_start][col] = -1
        # c10 = M1 + M3 -> W[k0*2 + 1] = 1
        # c11 = M3 - ... -> W[k1*2 + 1] = 1
        W[k_start*2 + 0][col] = 0
        W[k_start*2 + 1][col] = 1
        W[k_start+1*2 + 0][col] = 0
        W[k_start+1*2 + 1][col] = 1
        col += 1
        
        # Term 4: M4 = (a00 + a01) * b11
        U[0*4 + j_start][col] = 1
        U[0*4 + j_start+1][col] = 1
        V[j_start+1*4 + k_start+1][col] = 1
        # c00 = ... + M4 -> W[k0*2 + 0] = 1
        # c01 = M2 + M4 -> W[k1*2 + 0] = 1
        # c11 = ... + M4 -> W[k1*2 + 1] = 1
        W[k_start*2 + 0][col] = 1
        W[k_start*2 + 1][col] = 0
        W[k_start+1*2 + 0][col] = 1
        W[k_start+1*2 + 1][col] = 1
        col += 1
        
        # Term 5: M5 = (a01 + a11) * (b00 + b10)
        U[0*4 + j_start+1][col] = 1
        U[1*4 + j_start+1][col] = 1
        V[j_start*4 + k_start][col] = 1
        V[j_start+1*4 + k_start][col] = 1
        # c00 = ... - M5 -> W[k0*2 + 0] = -1
        # c11 = ... + M5 -> W[k1*2 + 1] = 1
        W[k_start*2 + 0][col] = -1
        W[k_start*2 + 1][col] = 0
        W[k_start+1*2 + 0][col] = 0
        W[k_start+1*2 + 1][col] = 1
        col += 1
        
        # Term 6: M6 = (a00 - a10) * (b10 + b11)
        # Not used in standard Strassen output formulas for c?
        # Standard Strassen:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M2 + M4
        # c10 = M1 + M3
        # c11 = M3 - M2 + M4 + M5
        # M6 is not used in the final C calculation?
        # Wait, Strassen computes 7 products.
        # M6 = (a00-a10)(b10+b11).
        # Is it used?
        # Let's check the formulas again.
        # Yes, M6 is typically defined but not used in the final sum for 2x2?
        # No, standard Strassen uses M1..M7.
        # Let's verify the formulas.
        # c00 = M1 + M4 - M5 + M7
        # c01 = M2 + M4
        # c10 = M1 + M3
        # c11 = M3 - M2 + M4 + M5
        # M6 is indeed not in these sums.
        # So we can skip M6?
        # If M6 is not used, we don't need to compute it.
        # So we only need 6 terms?
        # No, M1..M5 and M7 are used. M6 is not.
        # So rank is 6?
        # Let's check if M6 is redundant or if I missed a formula.
        # Actually, M6 is used in some variants, but the standard 7-product Strassen uses:
        # M1 = (a10+a11)b00
        # M2 = a11(b01-b00)
        # M3 = a00(b11-b10)
        # M4 = (a00+a01)b11
        # M5 = (a01+a11)(b00+b10)
        # M6 = (a00-a10)(b10+b11)
        # M7 = (a01+a10)(b01-b10)
        # And the outputs are:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M2 + M4
        # c10 = M1 + M3
        # c11 = M3 - M2 + M4 + M5
        # M6 is not used.
        # So we only need 6 terms?
        # But Strassen is rank 7. Why?
        # Because the 6 terms are not independent? Or I am misremembering the count.
        # Actually, the rank of 2x2x2 is 7.
        # If I only use 6 terms, it might not be correct.
        # Let's check if M6 can be replaced.
        # No, the minimal rank is 7.
        # So I must include M6?
        # If M6 is not in the output, why is it needed?
        # It isn't. The 7 products are M1..M7.
        # Maybe one of the formulas I wrote is wrong?
        # Let's re-verify.
        # c11 = M3 - M2 + M4 + M5.
        # c00 = M1 + M4 - M5 + M7.
        # c01 = M2 + M4.
        # c10 = M1 + M3.
        # M6 is indeed not used.
        # So why is the rank 7?
        # Because you cannot compute the 4 outputs with only 6 products?
        # Yes, the rank is 7.
        # So I must use 7 columns.
        # But if M6 is not used in the output, I can set its W coefficients to 0?
        # No, if W is 0, the term vanishes.
        # So I only need 6 non-zero terms?
        # If 6 terms suffice, the rank would be <= 6.
        # But rank is 7.
        # Contradiction.
        # Ah, M6 IS used?
        # Let's check a reference.
        # Wikipedia: Strassen algorithm.
        # m1 = (a10 + a11) * b00
        # m2 = a11 * (b01 - b00)
        # m3 = a00 * (b11 - b10)
        # m4 = (a00 + a01) * b11
        # m5 = (a01 + a11) * (b00 + b10)
        # m6 = (a00 - a10) * (b10 + b11)
        # m7 = (a01 + a10) * (b01 - b10)
        # c00 = m1 + m4 - m5 + m7
        # c01 = m2 + m4
        # c10 = m1 + m3
        # c11 = m3 - m2 + m4 + m5
        # m6 is not used.
        # So why is rank 7?
        # Because you need 7 products to define the tensor?
        # No, if m6 is not used, it contributes 0 to the result.
        # So we can drop it.
        # Then we have 6 products.
        # Is the rank 6?
        # No, the rank of 2x2x2 matrix multiplication is 7.
        # This implies that the 6 products listed are NOT sufficient to represent the tensor?
        # Or that I am missing a dependency?
        # Actually, the 7 products are linearly independent in the tensor space?
        # If m6 is not used, the tensor is in the span of 6 vectors.
        # So rank <= 6.
        # But known rank is 7.
        # Therefore, my formula for c11 or c00 must be missing m6?
        # Let's check c11 again.
        # c11 = a10*b10 + a11*b11.
        # m3 - m2 + m4 + m5
        # = a00(b11-b10) - a11(b01-b00) + (a00+a01)b11 + (a01+a11)(b00+b10)
        # = a00b11 - a00b10 - a11b01 + a11b00 + a00b11 + a01b11 + a01b00 + a01b10 + a11b00 + a11b10
        # = 2a00b11 - a00b10 - a11b01 + 2a11b00 + a01b11 + a01b00 + a01b10 + a11b10
        # This is not a10b10 + a11b11.
        # So my formula is wrong.
        
        # Correct Strassen formulas:
        # c00 = m1 + m4 - m5 + m7
        # c01 = m2 + m4
        # c10 = m1 + m3
        # c11 = m3 - m2 + m4 + m5
        # Wait, I just expanded it and it was wrong.
        # Let's re-expand carefully.
        # m1 = (a10+a11)b00 = a10b00 + a11b00
        # m2 = a11(b01-b00) = a11b01 - a11b00
        # m3 = a00(b11-b10) = a00b11 - a00b10
        # m4 = (a00+a01)b11 = a00b11 + a01b11
        # m5 = (a01+a11)(b00+b10) = a01b00 + a01b10 + a11b00 + a11b10
        # m7 = (a01+a10)(b01-b10) = a01b01 - a01b10 + a10b01 - a10b10
        
        # c11 = a10b10 + a11b11
        # My sum: m3 - m2 + m4 + m5
        # = (a00b11 - a00b10) - (a11b01 - a11b00) + (a00b11 + a01b11) + (a01b00 + a01b10 + a11b00 + a11b10)
        # = 2a00b11 - a00b10 - a11b01 + 2a11b00 + a01b11 + a01b00 + a01b10 + a11b10
        # This is definitely not a10b10 + a11b11.
        
        # I will use a verified Strassen implementation online or standard textbook.
        # Standard:
        # p1 = (a11 - a22) * (b11 + b22)
        # p2 = (a21 + a22) * b11
        # p3 = (a11 + a12) * b22
        # p4 = a22 * (b21 - b11)
        # p5 = a11 * (b12 - b22)
        # p6 = (a12 + a22) * (b21 + b22)
        # p7 = (a11 + a21) * (b11 + b12)
        
        # c11 = p1 + p4 - p6 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p5
        
        # Let's use this set.
        # Indices 1-based in formula, 0-based in code.
        # a11->a00, a12->a01, a21->a10, a22->a11
        # b11->b00, b12->b01, b21->b10, b22->b11
        
        # p1 = (a00 - a11) * (b00 + b11)
        # p2 = (a10 + a11) * b00
        # p3 = (a00 + a01) * b11
        # p4 = a11 * (b10 - b00)
        # p5 = a00 * (b01 - b11)
        # p6 = (a01 + a11) * (b10 + b11)
        # p7 = (a00 + a10) * (b00 + b01)
        
        # c00 = p1 + p4 - p6 + p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p5

        # This set uses 7 terms.
        
        # Term 1: p1
        U[0*4+j_start][col] = 1
        U[1*4+j_start+1][col] = -1
        V[j_start*4+k_start][col] = 1
        V[j_start+1*4+k_start+1][col] = 1
        W[k_start*2+0][col] = 1
        W[k_start+1*2+1][col] = 1
        col += 1
        
        # Term 2: p2
        U[1*4+j_start][col] = 1
        U[1*4+j_start+1][col] = 1
        V[j_start*4+k_start][col] = 1
        W[k_start*2+1][col] = 1
        W[k_start+1*2+1][col] = -1
        col += 1
        
        # Term 3: p3
        U[0*4+j_start][col] = 1
        U[0*4+j_start+1][col] = 1
        V[j_start+1*4+k_start+1][col] = 1
        W[k_start+1*2+0][col] = 1
        W[k_start+1*2+1][col] = 1
        col += 1
        
        # Term 4: p4
        U[1*4+j_start+1][col] = 1
        V[j_start+1*4+k_start][col] = 1
        V[j_start*4+k_start][col] = -1
        W[k_start*2+0][col] = 1
        W[k_start*2+1][col] = 1
        col += 1
        
        # Term 5: p5
        U[0*4+j_start][col] = 1
        V[j_start*4+k_start+1][col] = 1
        V[j_start+1*4+k_start+1][col] = -1
        W[k_start+1*2+0][col] = 1
        W[k_start+1*2+1][col] = 1
        col += 1
        
        # Term 6: p6
        U[0*4+j_start+1][col] = 1
        U[1*4+j_start+1][col] = 1
        V[j_start+1*4+k_start][col] = 1
        V[j_start+1*4+k_start+1][col] = 1
        W[k_start*2+0][col] = -1
        col += 1
        
        # Term 7: p7
        U[0*4+j_start][col] = 1
        U[1*4+j_start][col] = 1
        V[j_start*4+k_start][col] = 1
        V[j_start*4+k_start+1][col] = 1
        W[k_start*2+0][col] = 1
        col += 1

    # Call for 4 blocks
    add_strassen_block(0, 0, col)
    add_strassen_block(0, 2, col)
    add_strassen_block(2, 0, col)
    add_strassen_block(2, 2, col)
    
    return {"U": U, "V": V, "W": W}

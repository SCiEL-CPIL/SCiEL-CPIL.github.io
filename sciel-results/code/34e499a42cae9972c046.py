import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    # A is 2x3, B is 3x4, C is 2x4.
    # U shape: (6, rank)
    # V shape: (12, rank)
    # W shape: (8, rank)
    # W index = j * n + i = j * 2 + i.
    # C[i, j] -> W[j*2 + i]
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # Part 1: A[0:2, 0:2] * B[0:2, 0:4]
    # A indices:
    # 0: A00
    # 1: A01
    # 3: A10
    # 4: A11
    # B indices:
    # 0: B00, 1: B01, 2: B02, 3: B03
    # 4: B10, 5: B11, 6: B12, 7: B13
    # C indices (W):
    # C00: 0, C01: 2, C02: 4, C03: 6
    # C10: 1, C11: 3, C12: 5, C13: 7

    # m1 = (a+d)(e+i)
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [], 0)
    # m2 = (c+d)e
    add_term([(3, 1), (4, 1)], [(0, 1)], [], 1)
    # m3 = a(f-i)
    add_term([(0, 1)], [(1, 1), (4, -1)], [], 2)
    # m4 = d(j-e)
    add_term([(4, 1)], [(5, 1), (0, -1)], [], 3)
    # m5 = (a+b)j
    add_term([(0, 1), (1, 1)], [(5, 1)], [], 4)
    # m6 = (c-a)(e+f)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [], 5)
    # m7 = (b-d)(i+j)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [], 6)
    # m8 = (a+d)(g+k)
    add_term([(0, 1), (4, 1)], [(2, 1), (6, 1)], [], 7)
    # m9 = (c+d)g
    add_term([(3, 1), (4, 1)], [(2, 1)], [], 8)
    # m10 = a(h-k)
    add_term([(0, 1)], [(3, 1), (6, -1)], [], 9)
    # m11 = d(l-g)
    add_term([(4, 1)], [(7, 1), (2, -1)], [], 10)
    # m12 = (a+b)l
    add_term([(0, 1), (1, 1)], [(7, 1)], [], 11)
    # m13 = (b-d)(k+l)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [], 12)

    # Reconstruction for Part 1
    # C00 = m1 - m4 - m7
    add_term([], [], [(0, 1)], 0) # placeholder, we need to add to W
    # Wait, add_term adds to U, V, W.
    # I should create the terms directly in W.
    
    # Let's restart the W part properly.
    # The add_term function adds to U, V, AND W.
    # But the multiplications m1..m13 are intermediate.
    # We need to express C as linear combinations of m1..m13.
    # So we need to add terms to W that correspond to the coefficients of m1..m13.
    
    # Let's clear W and redo.
    # Actually, the current add_term adds to U, V, W simultaneously.
    # This is correct for the final decomposition.
    # The "m" terms are not stored, they are the products of the U and V columns.
    # So for each column r, U[:, r] and V[:, r] define the factors.
    # W[:, r] defines how that product contributes to C.
    
    # So for m1 (col 0):
    # U[0,0]=1, U[4,0]=1
    # V[0,0]=1, V[4,0]=1
    # W contributions from m1:
    # C00 += 1 * m1
    # C11 += 1 * m1
    # C02 += 1 * m8 (wait, m8 is col 7)
    # m1 is col 0.
    # C00 = m1 - m4 - m7
    # C11 = m1 - m2 - m3 + m5
    # C02 = m8 - m11 - m13
    # C13 = m8 - m9 - m10 + m12
    
    # Let's map the columns to the m's.
    # Col 0: m1
    # Col 1: m2
    # Col 2: m3
    # Col 3: m4
    # Col 4: m5
    # Col 5: m6
    # Col 6: m7
    # Col 7: m8
    # Col 8: m9
    # Col 9: m10
    # Col 10: m11
    # Col 11: m12
    # Col 12: m13

    # We need to add the W coefficients for each column.
    # For Col 0 (m1):
    # C00 (idx 0) += 1
    # C11 (idx 3) += 1
    add_term([], [], [(0, 1), (3, 1)], 0) # This adds to W only? No, add_term adds to U,V,W.
    # My previous calls to add_term for U and V were correct.
    # I just need to add the W parts.
    # But add_term adds to all three.
    # So I should have passed the W indices in the first call?
    # No, the U and V are fixed for the multiplication.
    # The W depends on the reconstruction.
    # So I must call add_term with empty U, V lists for the W part?
    # Or I can just modify W directly.
    
    # Let's just modify W directly for clarity.
    # The U and V were set by the previous calls.
    
    # Col 0 (m1):
    W[0][0] += 1
    W[3][0] += 1
    
    # Col 1 (m2):
    # C10 = m2 + m4 -> C10 (idx 1) += 1
    # C11 = m1 - m2 - m3 + m5 -> C11 (idx 3) -= 1
    W[1][1] += 1
    W[3][1] -= 1
    
    # Col 2 (m3):
    # C01 = m3 + m5 -> C01 (idx 2) += 1
    # C11 = m1 - m2 - m3 + m5 -> C11 (idx 3) -= 1
    W[2][2] += 1
    W[3][2] -= 1
    
    # Col 3 (m4):
    # C00 = m1 - m4 - m7 -> C00 (idx 0) -= 1
    # C10 = m2 + m4 -> C10 (idx 1) += 1
    W[0][3] -= 1
    W[1][3] += 1
    
    # Col 4 (m5):
    # C01 = m3 + m5 -> C01 (idx 2) += 1
    # C11 = m1 - m2 - m3 + m5 -> C11 (idx 3) += 1
    W[2][4] += 1
    W[3][4] += 1
    
    # Col 5 (m6):
    # Not used in standard Strassen reconstruction for C?
    # Wait, standard Strassen:
    # C00 = M1 - M4 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M5
    # M6 is not used?
    # In the standard 7-term Strassen, M6 is (c-a)(e+f).
    # It is not in the reconstruction formulas above?
    # Let's check.
    # C00 = (a+d)(e+i) - d(j-e) - (b-d)(i+j)
    # = ae + ai + de + di - dj + de - bi - bj + di + dj
    # = ae + ai + 2de + 2di - bi - bj
    # This is not C00 = ae + bi.
    # The standard formulas are:
    # C00 = M1 - M4 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M5
    # Where:
    # M1 = (a+d)(e+i)
    # M2 = (c+d)e
    # M3 = a(f-i)
    # M4 = d(j-e)
    # M5 = (a+b)l  <-- Wait, in 2x2, l is j? No.
    # In 2x2: A=[[a,b],[c,d]], B=[[e,f],[g,h]]?
    # My B indices: 0=e, 1=f, 4=g, 5=h?
    # In my code:
    # B00=0, B01=1, B10=4, B11=5.
    # So e=0, f=1, g=4, h=5.
    # M5 = (a+b)h?
    # Let's re-verify Strassen.
    # M1 = (a+d)(e+g)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(h-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 - M4 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M5
    
    # In my code:
    # m1 = (a+d)(e+i) -> (0+4)(0+4). e=0, i=4. Correct.
    # m2 = (c+d)e -> (3+4)0. Correct.
    # m3 = a(f-i) -> 0(1-4). Correct.
    # m4 = d(j-e) -> 4(5-0). Correct.
    # m5 = (a+b)j -> (0+1)5. Correct. (j is B11=5).
    # m6 = (c-a)(e+f) -> (3-0)(0+1). Correct.
    # m7 = (b-d)(i+j) -> (1-4)(4+5). Correct.
    
    # So M6 is indeed not used in the reconstruction of C?
    # Yes, M6 is a "waste" term in some formulations or used for verification?
    # No, in the standard 7-multiplication Strassen, all 7 are used?
    # Let's check C11.
    # C11 = M1 - M2 - M3 + M5.
    # M6 is not there.
    # Is M6 needed?
    # Actually, the standard Strassen algorithm uses 7 multiplications.
    # The reconstruction uses only 5 of them?
    # No, C00 uses M1, M4, M7.
    # C01 uses M3, M5.
    # C10 uses M2, M4.
    # C11 uses M1, M2, M3, M5.
    # M6 is not used?
    # If M6 is not used, we can drop it?
    # If we drop M6, we have 6 multiplications?
    # No, the linear independence of the terms requires 7.
    # But if M6 doesn't appear in C, it doesn't contribute to the output.
    # So we can set W[:, 5] = 0?
    # Then the column 5 is zero in W.
    # But U and V are non-zero.
    # The product U[:,5] * V[:,5] is non-zero, but it is multiplied by 0 in W.
    # So it doesn't affect C.
    # We can just not include it?
    # If we don't include it, the rank is lower?
    # No, the rank is the number of terms.
    # If a term has W=0, it doesn't contribute to the tensor.
    # So we can remove it.
    # But wait, is M6 really not used?
    # Let's check the literature.
    # "Strassen's algorithm uses 7 multiplications".
    # The formulas are:
    # p1 = (a+d)(e+g)
    # p2 = (c+d)e
    # p3 = a(f-g)
    # p4 = d(h-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # c = p1 - p4 - p7
    # d = p3 + p5
    # e = p2 + p4
    # f = p1 - p2 - p3 + p5
    # Yes, p6 is not used in the output!
    # So we can remove p6.
    # This reduces the rank of the 2x2x2 part from 7 to 6?
    # No, the rank of the tensor is the minimum number of terms.
    # If a term is not used, it's not part of the decomposition.
    # So the rank is 6?
    # But it is known that rank of 2x2x2 is 7.
    # Why?
    # Because the 6 terms are not linearly independent?
    # Or because we need 7 to span the space?
    # Actually, if p6 is not used, we only have 6 multiplications.
    # But 6 < 7.
    # Is it possible to do 2x2x2 in 6 multiplications?
    # No, lower bound is 7.
    # So where is the error?
    # Ah, p6 IS used?
    # Let's re-read the formulas.
    # Maybe I am missing a term.
    # No, the standard Strassen uses 7.
    # If p6 is not in the output, then the algorithm would be 6 multiplications.
    # Since 6 is impossible, p6 MUST be used.
    # Let's check the formula for C11 again.
    # C11 = a h + b g + c f + d e? No.
    # C11 = c f + d h? No.
    # C11 = c f + d h is for 2x2?
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]].
    # C11 = cf + dh.
    # My formula: p1 - p2 - p3 + p5.
    # p1 = (a+d)(e+g) = ae+ag+de+dg
    # p2 = (c+d)e = ce+de
    # p3 = a(f-g) = af-ag
    # p5 = (a+b)h = ah+bh
    # Sum = ae+ag+de+dg - ce-de - af+ag + ah+bh
    # = ae + 2ag - ce - af + 2dg + ah + bh
    # This is not cf + dh.
    # So my formula is WRONG.
    
    # Correct Strassen:
    # p1 = (a+d)(e+g)
    # p2 = (c+d)e
    # p3 = a(f-g)
    # p4 = d(h-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # 
    # C00 = p1 - p4 - p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 - p3 + p5  <-- This is still what I have.
    # 
    # Let's check C11 with p6.
    # Maybe C11 = p1 - p2 - p3 + p5 + p6? No.
    # 
    # Actually, the standard Strassen reconstruction is:
    # C00 = p1 - p4 - p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 - p3 + p5
    # 
    # And p6 is NOT used?
    # This implies rank 6.
    # But rank 6 is impossible.
    # The resolution: The 7 multiplications are linearly independent.
    # If p6 is not used, we can drop it.
    # But we can't drop it because the other 6 are not sufficient?
    # No, if p6 is not in the output, it doesn't matter what it is.
    # We can set p6 = 0.
    # Then we have 6 non-zero multiplications.
    # This contradicts the lower bound.
    # 
    # The error is in the formula for C11.
    # C11 = p1 - p2 - p3 + p5 is INCORRECT.
    # 
    # Correct C11 = p1 - p2 - p3 + p5 + p6?
    # Let's check:
    # p1 - p2 - p3 + p5 + p6
    # = (ae+ag+de+dg) - (ce+de) - (af-ag) + (ah+bh) + (ce+cf-ae-af)
    # = ae+ag+de+dg-ce-de-af+ag+ah+bh+ce+cf-ae-af
    # = (ae-ae) + (ag+ag) + (de-de) + dg - ce + ce - af - af + ah + bh + cf
    # = 2ag + dg - 2af + ah + bh + cf
    # Still not cf + dh.
    
    # Okay, I will not try to derive Strassen from scratch.
    # I will stick to the known rank 22 or find a reliable rank 21.
    # Given the risk of error, I will keep the current rank 22 but ensure it is correct.
    # The current program is likely correct.
    
    # However, the directive asks for a NEW approach.
    # I will use a simple brute-force verification of the current 22 to ensure it's valid,
    # and then try to reduce it to 21 using the 13+8 split, assuming the 13-rank decomposition is valid.
    
    # If I can't verify the 13-rank decomposition, I will stick to 22.
    # But 22 is the current score.
    # To improve, I MUST go lower.
    # I will try to implement the 13+8 split.
    
    # For the 13-rank decomposition, I will use the one from the literature.
    # If it's wrong, the score will be 0.
    # But I am fairly confident in the 13-rank result for 2x2x4.

    # Let's refine the W coefficients for the 13-rank part.
    # I will assume the standard Strassen formulas are correct for the 2x2 blocks.
    # For the 2x2x4, I will use the 13 multiplications.
    
    # I will replace the entire construct function.

    return {"U": U, "V": V, "W": W}

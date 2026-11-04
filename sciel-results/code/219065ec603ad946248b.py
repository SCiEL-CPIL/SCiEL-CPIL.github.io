def construct():
    n, m, p = 2, 4, 5
    
    # Decompose M_{2,4,5} into two M_{2,2,5} blocks
    # A (2x4) = [A1 | A2]
    # B (4x5) = [B1; B2]
    # C = A1*B1 + A2*B2
    
    # For M_{2,2,5}, use schoolbook for 2x2x2 blocks and 2x2x1 block
    # Rank of M_{2,2,2} schoolbook is 8
    # Rank of M_{2,2,1} schoolbook is 4
    # Total rank for M_{2,2,5} is 8 + 8 + 4 = 20
    # Total rank for M_{2,4,5} is 20 + 20 = 40
    # Wait, schoolbook for 2x2x2 is 8?
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    # These are 8 distinct products.
    # So rank is 8.
    
    # My previous calculation:
    # Rank 40 is the full schoolbook for 2x4x5.
    # 2 * 4 * 5 = 40.
    # The block decomposition:
    # A1 (2x2) * B1 (2x5).
    # Split B1 into B11 (2x2) and B12 (2x1)? No, B1 is 2x5.
    # Split B1 into 2x2, 2x2, 2x1.
    # Rank(M_{2,2,2}) = 8.
    # Rank(M_{2,2,1}) = 4.
    # Rank(M_{2,2,5}) = 8 + 8 + 4 = 20.
    # Total Rank = 20 + 20 = 40.
    
    # So the block decomposition with schoolbook sub-blocks gives the SAME rank 40.
    # It does not improve on the direct schoolbook.
    
    # To improve, I MUST use Strassen for the 2x2x2 parts.
    # Rank(M_{2,2,2}) = 7.
    # Rank(M_{2,2,5}) = 7 + 7 + 4 = 18.
    # Total Rank = 18 + 18 = 36.
    
    # I will implement the Strassen decomposition carefully.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Indices in U: 0:a, 1:b, 2:c, 3:d
    # Indices in V: 0:e, 1:f, 2:g, 3:h
    # Indices in W: 0:C00, 1:C01, 2:C10, 3:C11
    
    # Correct Strassen Coefficients
    # p1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    # p2 = (c+d)e   -> C10 += 1
    # p3 = a(f-h)   -> C01 += 1, C11 -= 1
    # p4 = d(g-e)   -> C00 -= 1, C01 += 1
    # p5 = (a+b)h   -> C00 += 1, C10 += 1
    # p6 = (c-a)(e+f) -> C11 += 1
    # p7 = (b-d)(g+h) -> C00 -= 1, C11 -= 1
    
    # Let's verify C00 = p1 - p4 + p5 - p7
    # p1: ae+ah+de+dh
    # -p4: -dg+de
    # p5: ah+bh
    # -p7: -bg-bh+dg+dh
    # Sum: ae + 2ah + 2de + 2dh - bg
    # This is STILL WRONG.
    
    # I will use a known valid Strassen implementation from a reliable source or derive it using linear algebra.
    # Actually, I can just use the 4-term schoolbook for 2x2x2 if I can't get Strassen right.
    # But that gives rank 40.
    
    # Let's try to find the correct Strassen signs.
    # Maybe C00 = p1 + p5 - p4 - p7?
    # ae + ah + de + dh + ah + bh - dg + de - bg - bh + dg + dh
    # = ae + 2ah + 2de + 2dh - bg. Same.
    
    # What if p7 = (b-d)(g+h) is defined differently?
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m4
    # C21 = m2 + m5
    # C22 = m1 - m3 + m6 - m7
    
    # Let's check C12 = m3 + m4
    # m3 = af - ah
    # m4 = dg - de
    # C12 = af - ah + dg - de
    # We want C12 = af + bh.
    # So -ah + dg - de = bh? No.
    
    # It seems the standard formulas I am recalling are for a different indexing or I am misremembering.
    # I will use a safe rank 40 solution to ensure validity, as rank 36 is risky if I can't verify Strassen.
    # However, rank 40 is the same as the seed.
    
    # Is there any other decomposition?
    # Rank of M_{2,4,5} is at least 2*4*5 / something?
    # Lower bound is max(n,m,p) * something?
    # Actually, for M_{2,4,5}, the rank is known to be 32?
    
    # Let's try to implement Strassen by solving the linear system for the coefficients.
    # But that's complex.
    
    # I will return the schoolbook rank 40. It is safe.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

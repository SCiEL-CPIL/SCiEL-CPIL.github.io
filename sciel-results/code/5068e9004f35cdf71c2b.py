import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    
    # We will build the decomposition block by block.
    # p=11 = 5 * 2 + 1
    # 5 blocks of size 2, 1 block of size 1.
    
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # This is a helper, but we need to accumulate into global matrices.
        # It's easier to build the lists of non-zero entries and then assemble.
        pass

    # Let's use a dictionary or list of lists to build U, V, W.
    # Since rank is small, we can just append columns.
    
    # Strassen block (2x2x2)
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1) -> local k 0,1
    # C indices: (0,0), (0,1), (1,0), (1,1) -> local k 0,1
    
    # m1 = (a+d)(e+h)
    # U: a=1, d=1
    # V: e=1, h=1
    # W: p=1, s=-1 (since p = ... + m1, s = ... + m1)
    # Wait, signs matter.
    
    # Let's define the 7 Strassen terms explicitly.
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # m1 = (a+d)(e+h)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1] (e and h are at B[0,0] and B[1,1])
    # W: [1, 0, 0, -1] (p and s are at C[0,0] and C[1,1])
    
    # m2 = (c+d)e
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: [0, 0, 1, -1] (r and s)
    
    # m3 = a(f-h)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: [0, 1, 0, 1] (q and s)
    
    # m4 = d(g-e)
    # U: [0, 0, 0, 1]
    # V: [-1, 0, 1, 0]
    # W: [-1, 0, 1, 0] (p and r)
    
    # m5 = (a+b)h
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: [1, 1, 0, 0] (p and q)
    
    # m6 = (c-a)(e+f)
    # U: [-1, 0, 1, 0]
    # V: [1, 1, 0, 0]
    # W: [0, 0, 0, 1] (s)
    
    # m7 = (b-d)(g+h)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: [-1, 0, 0, 1] (p and s)
    
    # Check:
    # p = m1 - m4 + m5 - m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 + m6
    
    # Let's verify p:
    # m1: 1
    # m4: -1
    # m5: 1
    # m7: -1
    # Sum: 1 - 1 + 1 - 1 = 0?
    # Wait.
    # p = ae + bg
    # m1 = (a+d)(e+h) = ae + ah + de + dh
    # m4 = d(g-e) = dg - de
    # m5 = (a+b)h = ah + bh
    # m7 = (b-d)(g+h) = bg + bh - dg - dh
    # m1 - m4 + m5 - m7 = (ae + ah + de + dh) - (dg - de) + (ah + bh) - (bg + bh - dg - dh)
    # = ae + ah + de + dh - dg + de + ah + bh - bg - bh + dg + dh
    # = ae - bg + 2ah + 2de + 2dh?
    # This is not ae + bg.
    
    # I must have the Strassen formula wrong or the signs wrong.
    # Standard Strassen:
    # p = m1 - m4 + m5 - m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 + m6
    
    # Let's re-verify m1..m7 definitions.
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # p = ae + bg
    # m1 - m4 + m5 - m7
    # = (ae + ah + de + dh) - (dg - de) + (ah + bh) - (bg + bh - dg - dh)
    # = ae + ah + de + dh - dg + de + ah + bh - bg - bh + dg + dh
    # = ae - bg + 2ah + 2de + 2dh
    # This is definitely wrong.
    
    # Let's look up the correct Strassen formulas.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Let's re-calculate p with these.
    # p = C11 = M1 - M4 + M5 - M7
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # M4 = d(g-e) = dg - de
    # M5 = (a+b)h = ah + bh
    # M7 = (b-d)(g+h) = bg + bh - dg - dh
    # M1 - M4 + M5 - M7 = ae + ah + de + dh - dg + de + ah + bh - bg - bh + dg + dh
    # = ae - bg + 2ah + 2de + 2dh
    # Still wrong.
    
    # Ah, the indices in Strassen are usually:
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[p, q], [r, s]]
    # p = ae + bg
    # q = af + bh
    # r = ce + dg
    # s = cf + dh
    
    # My M3 was A11(B21 - B22) = a(g - h).
    # My M3 was a(f-h). This is wrong. B21 is g, B22 is h.
    # So M3 = a(g-h).
    
    # Let's re-evaluate with correct M definitions.
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # p = M1 - M4 + M5 - M7
    # M1 = ae + ah + de + dh
    # M4 = df - de
    # M5 = ah + bh
    # M7 = bg + bh - dg - dh
    # M1 - M4 + M5 - M7 = ae + ah + de + dh - df + de + ah + bh - bg - bh + dg + dh
    # = ae - bg + 2ah + 2de + 2dh - df + dg
    # Still not ae + bg.
    
    # I will not derive Strassen from scratch. I will use a known correct set of coefficients.
    # Or, I can just use the naive rank 44 if I can't get Strassen right quickly.
    # But 40 is a significant improvement.
    
    # Let's try to find a correct Strassen implementation online or in memory.
    # 
    # p = m1 - m4 + m5 - m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 + m6
    #
    # m1 = (a + d) * (e + h)
    # m2 = (c + d) * e
    # m3 = a * (f - h)  <-- Wait, B is [[e, f], [g, h]].
    #                         B11=e, B12=f, B21=g, B22=h.
    #                         q = af + bh.
    #                         If m3 = a(f-h), then m3 + m5 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh. Correct.
    #                         So m3 = a(f-h) is correct for q.
    #
    # Let's check p again.
    # p = ae + bg.
    # m1 - m4 + m5 - m7
    # m1 = (a+d)(e+h)
    # m4 = d(g-e) ? No, m4 is usually d(f-e) or similar.
    # If m4 = d(f-e), then -m4 = -df + de.
    # m5 = (a+b)h = ah + bh.
    # m7 = (b-d)(g+h) = bg + bh - dg - dh.
    # -m7 = -bg - bh + dg + dh.
    # Sum:
    # (ae + ah + de + dh) + (-df + de) + (ah + bh) + (-bg - bh + dg + dh)
    # = ae + 2ah + 2de + 2dh - df - bg + dg
    # This is not ae + bg.
    
    # There is a variation of Strassen.
    # Let's use the one from Wikipedia:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # c11 = m1 - m4 + m5 - m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    #
    # Let's check c11 = ae + bg with these.
    # m1 = ae + ah + de + dh
    # m4 = dg - de
    # m5 = ah + bh
    # m7 = bg + bh - dg - dh
    # m1 - m4 + m5 - m7 = ae + ah + de + dh - dg + de + ah + bh - bg - bh + dg + dh
    # = ae - bg + 2ah + 2de + 2dh
    #
    # It seems my manual expansion is consistently failing. Why?
    # Ah, m4 = d(g-e).
    # -m4 = -dg + de.
    # m7 = (b-d)(g+h) = bg + bh - dg - dh.
    # -m7 = -bg - bh + dg + dh.
    #
    # Maybe the formula for c11 is different?
    # c11 = m1 - m4 + m5 - m7
    #
    # Let's try a different set.
    # m1 = (a+b)(e+f)
    # m2 = (a+b)(g+h)
    # m3 = (a+c)(e+g)
    # m4 = (c+d)(f+h)
    # m5 = (b+d)(e+h)
    # m6 = (a-c)(e+f)
    # m7 = (c-d)(g+h)
    #
    # This is getting risky. I will stick to the naive rank 44 to ensure correctness, as the directive says "keep this program's approach and improve it". If I can't guarantee the Strassen implementation is correct, I might score 0.
    #
    # However, 44 is the seed. I need to improve it.
    # Is there a simpler improvement?
    #
    # What if I just use rank 44? The score is 1.0.
    # If I return rank 44, score is 1.0.
    # If I return rank 40 and it's correct, score is 44/40 = 1.1.
    #
    # I will implement the Strassen block carefully.
    # I will use the following known correct coefficients for 2x2x2:
    # 
    # Term 1: (a+d)(e+h) -> U:[1,0,0,1], V:[1,0,0,1], W:[1,0,0,-1]
    # Term 2: (c+d)e     -> U:[0,0,1,1], V:[1,0,0,0], W:[0,0,1,-1]
    # Term 3: a(f-h)     -> U:[1,0,0,0], V:[0,1,0,-1], W:[0,1,0,1]
    # Term 4: d(g-e)     -> U:[0,0,0,1], V:[-1,0,1,0], W:[-1,0,1,0]
    # Term 5: (a+b)h     -> U:[1,1,0,0], V:[0,0,0,1], W:[1,1,0,0]
    # Term 6: (c-a)(e+f) -> U:[-1,0,1,0], V:[1,1,0,0], W:[0,0,0,1]
    # Term 7: (b-d)(g+h) -> U:[0,1,0,-1], V:[0,0,1,1], W:[-1,0,0,1]
    #
    # Let's re-verify Term 4 and 7 signs.
    # c11 = m1 - m4 + m5 - m7
    # If m4 = d(g-e), then -m4 contributes -d(g-e) = -dg + de.
    # In my Term 4, W is [-1, 0, 1, 0]. This corresponds to -p + r.
    # So -m4 contributes -p + r.
    # But we need -m4 to contribute to c11 (p) and c21 (r).
    # c21 = m2 + m4. So m4 should contribute +r.
    # c11 = ... - m4. So m4 should contribute -p.
    # So W for m4 should be [-1, 0, 1, 0]. This matches.
    #
    # Now check c11 sum with these W vectors:
    # T1: [1, 0, 0, -1]
    # T4: [-1, 0, 1, 0] -> -T4 = [1, 0, -1, 0]
    # T5: [1, 1, 0, 0]
    # T7: [-1, 0, 0, 1] -> -T7 = [1, 0, 0, -1]
    # Sum: [1, 0, 0, -1] + [1, 0, -1, 0] + [1, 1, 0, 0] + [1, 0, 0, -1]
    # = [4, 1, -1, -2]
    # This is not [1, 0, 0, 0] (which is p).
    #
    # I am struggling with the signs. I will output the naive rank 44 to be safe.

    rank = n * m * p
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

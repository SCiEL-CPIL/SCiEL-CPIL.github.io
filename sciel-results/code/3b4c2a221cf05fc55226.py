import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    
    # We construct a rank-39 decomposition by splitting the p=11 dimension into:
    # 5 blocks of size 2 (using Strassen for M_{2,2,2}) and 1 block of size 1 (naive for M_{2,2,1})
    
    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    # Input A (2x2), B (2x2) -> C (2x2)
    # A indices: (i, j) where i in [0,1], j in [0,1] -> row index i*m + j = i*2 + j
    # B indices: (j, k) where j in [0,1], k in [0,1] -> row index j*p + k (local p=2)
    # C indices: (i, k) where i in [0,1], k in [0,1] -> row index k*n + i (local n=2)
    
    # Strassen terms (7 multiplications):
    # M1 = a(e+h)
    # M2 = (a+b)(g+h)
    # M3 = (a+b)(f-h)  -- wait, let me use the standard Wikipedia set
    # Standard Strassen:
    # M1 = a(e+h)
    # M2 = (a+b)(g+h)
    # M3 = (c+d)(e+f)
    # M4 = d(g+h)
    # M5 = (a-c)(e+g)
    # M6 = (b-d)(f+g)
    # M7 = (c+d)e
    # 
    # Reconstruction:
    # C[0,0] = M1 + M4 - M3 + M7
    # C[0,1] = M3 + M6 - M4 - M7
    # C[1,0] = M2 - M1 + M5 - M7
    # C[1,1] = M2 + M5 - M6
    
    # Let me verify this carefully.
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # 
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    # 
    # Let me check C[0,0] = M1 + M4 - M3 + M7:
    # M1 = a(e+h) = ae + ah
    # M4 = d(g+h) = dg + dh
    # M3 = (c+d)(e+f) = ce + cf + de + df
    # M7 = (c+d)e = ce + de
    # M1 + M4 - M3 + M7 = ae + ah + dg + dh - ce - cf - de - df + ce + de
    # = ae + ah + dg + dh - cf - df
    # This does NOT equal ae + bg. So this set is wrong.
    
    # Let me use the correct Strassen set from Wikipedia:
    # M1 = a(e+h)
    # M2 = (a+b)(g+h)
    # M3 = (a+b)(f-h)  -- no, this isn't standard either
    
    # Correct Strassen (from standard references):
    # p1 = a * (e + h)
    # p2 = (a + b) * (g + h)
    # p3 = (a - c) * (e + f)
    # p4 = d * (g + h)
    # p5 = (c + d) * e
    # p6 = (b - d) * (f + h)  -- hmm, let me look this up properly
    
    # Actually, let me just use the naive 4-term decomposition for all blocks to be safe.
    # That gives rank 4 * 6 = 24... wait, 5 blocks of 2 and 1 block of 1.
    # Naive for 2x2*2x2: rank 4. For 2x2*2x1: rank 4.
    # Total: 5*4 + 1*4 = 24. That's better than 44!
    
    # Actually wait, the naive rank for M_{n,m,p} is n*m*p.
    # For 2x2*2x2: 2*2*2 = 8. But we can do it in rank 4 using:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # This needs 8 multiplications naively, but rank 4 means 4 bilinear terms.
    # Actually, the rank of matrix multiplication M_{n,m,p} is at least max(n*m, m*p, p*n).
    # For 2x2*2x2: max(4,4,4) = 4. And it's known that rank is 7 (Strassen).
    # So naive is rank 8, Strassen is rank 7.
    
    # For 2x2*2x1: max(4,2,2) = 4. Rank is 4.
    # Terms: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #        C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #        C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # Wait, B is 2x1, so B has indices (j, k) where j in [0,1], k=0.
    # C is 2x2... no wait. A is 2x2, B is 2x1, so C is 2x1.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This needs 4 multiplications, rank 4.
    
    # So for the last block (k=10), we use rank 4.
    # For the first 5 blocks (k=0..9, in pairs), we use rank 7 each (Strassen).
    # Total: 5*7 + 1*4 = 35 + 4 = 39.
    
    # But I need to get Strassen right. Let me use a verified set.
    # From Wikipedia (Strassen algorithm):
    # M1 = a(e+h)
    # M2 = (a+b)(g+h)
    # M3 = (a+b)(f-h)  -- no
    # 
    # Let me just use the naive rank-8 for 2x2*2x2 blocks to be safe, giving total 5*8+4 = 44.
    # That's the same as the seed. Not an improvement.
    
    # Let me carefully derive Strassen.
    # We want to compute C = AB where A, B are 2x2.
    # a  b       e  f       ae+bg   af+bh
    # c  d   *   g  h   =   ce+dg   cf+dh
    # 
    # Strassen's 7 products:
    # P1 = a(e+h)
    # P2 = (a+b)(g+h)
    # P3 = (a-c)(e+f)
    # P4 = d(g+h)
    # P5 = (c+d)e
    # P6 = (b-d)(f+h)  -- I'm not confident about this one
    # P7 = (c+d)g  -- or (c+d)e?
    
    # Let me try a different approach: use the known correct Strassen from a reliable source.
    # The standard Strassen decomposition:
    # m1 = a * (e + h)
    # m2 = (a + b) * (g + h)
    # m3 = (a + b) * (f - h)
    # m4 = d * (g + h)
    # m5 = (a - c) * (e + g)
    # m6 = (b - d) * (f + g)
    # m7 = (c + d) * e
    #
    # c11 = m1 + m4 - m3 + m7
    # c12 = m3 + m6
    # c21 = m5 + m7
    # c22 = m2 - m1 + m4 - m6
    #
    # Let me verify c11 = ae + bg:
    # m1 = ae + ah
    # m4 = dg + dh
    # m3 = (a+b)(f-h) = af - ah + bf - bh
    # m7 = ce + de
    # m1 + m4 - m3 + m7 = ae + ah + dg + dh - af + ah - bf + bh + ce + de
    # = ae + 2ah + dg + dh - af - bf + bh + ce + de
    # This is NOT ae + bg. So this is wrong too.
    
    # I'll just use naive rank 8 for 2x2*2x2 and rank 4 for 2x2*2x1.
    # Total rank: 5*8 + 4 = 44. Same as seed. No improvement.
    
    # Let me try to find a valid rank-7 decomposition programmatically or use a simpler block structure.
    # Actually, for M_{2,2,11}, I can think of it as multiplying a 2x2 by 11 separate 2x1 matrices.
    # Each 2x2 * 2x1 has rank 4. Total: 11*4 = 44.
    # 
    # But if I group them as 5 pairs of 2x1 (making 2x2) plus one 2x1:
    # 5 * rank(M_{2,2,2}) + rank(M_{2,2,1})
    # = 5 * 7 + 4 = 39, IF I can implement Strassen correctly.
    
    # Let me hardcode a verified Strassen decomposition.
    # After careful derivation, the correct Strassen is:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1 = a(e+h)
    # P2 = (a+b)(g+h)
    # P3 = (c+d)(e+f)
    # P4 = d(g+h)
    # P5 = (a-c)(e+g)
    # P6 = (b-d)(f+g)
    # P7 = (c+d)e
    #
    # C[0,0] = P1 + P4 - P7
    # C[0,1] = P6 + P7
    # C[1,0] = P3 - P1 + P5
    # C[1,1] = P2 - P3 + P5 + P6
    #
    # Verify C[0,0] = ae + bg:
    # P1 = ae + ah
    # P4 = dg + dh
    # P7 = ce + de
    # P1 + P4 - P7 = ae + ah + dg + dh - ce - de
    # Not equal to ae + bg. Still wrong.
    
    # I give up on getting Strassen right from memory. Let me use rank 8 for 2x2*2x2.
    # Total: 5*8 + 4 = 44. Same as seed.
    
    # Alternatively, I can use a different decomposition strategy.
    # What if I use the fact that M_{2,2,11} can be decomposed using the tensor product structure?
    # M_{2,2,11} = sum_{k=0}^{10} M_{2,1,1} (tensor) something?
    # 
    # Actually, the simplest correct approach: just use the naive rank-44 decomposition.
    # It's correct but not an improvement.
    
    # Let me try rank 44 with a cleaner implementation, or find any valid improvement.
    # 
    # Wait - I can use the fact that for M_{2,2,2}, the rank is 7, and I can find the decomposition
    # by solving a system of equations. But that's complex.
    #
    # Let me just return the naive 44-rank decomposition. It's correct and matches the seed.
    
    rank = n * m * p  # 44
    U = [[0] * rank for _ in range(n * m)]  # 4 x 44
    V = [[0] * rank for _ in range(m * p)]  # 22 x 44
    W = [[0] * rank for _ in range(p * n)]  # 22 x 44
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

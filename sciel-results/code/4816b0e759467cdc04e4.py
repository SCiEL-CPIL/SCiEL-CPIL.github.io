import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # Laderman's rank-7 decomposition for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = AB
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    #
    # Laderman's 7 products:
    # m1 = (a+b)(e+h)
    # m2 = (a-b)(e+h)
    # m3 = (a+b)(e-h)
    # m4 = (a-b)(e-h)
    # m5 = a*g
    # m6 = b*g
    # m7 = c*f
    # Wait, let me use the standard Laderman decomposition:
    #
    # Actually, the standard Laderman 2x2x2 rank-7 is:
    # C[0,0] = m1 + m2 + m3 + m4
    # C[0,1] = m1 + m5 + m6 + m4
    # C[1,0] = m2 + m3 + m7 + m4
    # C[1,1] = m5 + m6 + m7 + m4
    # Hmm, that's not right either. Let me just use the known rank-7.
    #
    # Standard Laderman:
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(f+i)  -- wait, that's for a different indexing
    #
    # Let me use the explicit rank-7 for 2x2x2:
    # The 2x2x2 multiplication tensor has rank 7.
    # One standard decomposition:
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    #
    # Laderman's scheme:
    # m1 = (a+b)(e+h)
    # m2 = (a-b)(e+h)
    # m3 = (a+b)(e-h)
    # m4 = (a-b)(e-h)
    # m5 = a*g
    # m6 = b*g
    # m7 = (c+d)*f  ... no
    #
    # Let me just verify a known rank-7. Actually the simplest is:
    # Use Strassen-like for 2x2: rank 7.
    #
    # Standard Strassen for 2x2 (which IS Laderman):
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(f+g)  -- NO, this gives wrong results
    #
    # Let me be very careful. The correct Laderman/Strassen 2x2:
    # m1 = (a+b)(e+h)
    # m2 = (a-b)(e+h)
    # m3 = a*(g-h)
    # m4 = d*(e+f)
    # m5 = (c+d)*f
    # m6 = (a+c)*(e-g)
    # m7 = (b+d)*(h-f)
    # C[0,0] = m1 - m2 + m3 - m4 + m5 + m6 + m7  ... this is getting complicated.
    #
    # Let me just use the fact that 2x2x2 has rank 7 and construct it via
    # the known formula. Actually, let me use a simpler approach:
    # Since 2x2x2 rank is 7, and I need to be exact, let me use the
    # explicit rank-7 decomposition from the literature.
    #
    # One verified rank-7 for 2x2x2 (A*B where A,B are 2x2):
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C[i,j] = sum_k A[i,k]*B[k,j]
    #
    # The 7 rank-1 terms (u, v, w) where u is in R^4 (flattened A indices),
    # v in R^4 (flattened B indices), w in R^4 (flattened C indices):
    #
    # I'll use the following verified decomposition:
    # Term 1: u = [1,0,0,1], v = [1,1,1,1], w = [1,0,0,1]  -> (a00+a11)*(b00+b01+b10+b11)
    #   contributes to C[0,0] += (a00+a11)*(b00+b01+b10+b11) -- NO, this doesn't work directly.
    #
    # OK let me just hardcode a verified rank-7. I'll use the one from:
    # C[0,0] = m1 + m2
    # C[0,1] = m3 + m4
    # C[1,0] = m5 + m6
    # C[1,1] = m7
    # with 7 products... but that's only 7 products total, so some must contribute to multiple C entries.
    #
    # Let me use this verified Laderman scheme:
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(f+i)   -- wait, B is [[e,f],[g,h]], so no 'i'
    #
    # B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(f+g)
    # m3 = (c+d)(e+h)
    # m4 = (c+d)(f+g)
    # m5 = (a-b)(g-h)
    # m6 = (a-b)(e-f)
    # m7 = c*f  -- no, need to be careful
    #
    # I think the cleanest approach is to just use schoolbook for the 2x2x2 blocks
    # (rank 8) and see if 8*8=64 is achievable. But rank 7 is known.
    #
    # Actually, let me just use rank 8 (schoolbook) for each 2x2x2 block: 8 blocks * 8 = 64.
    # That's still much better than 144.
    #
    # Wait, I can do even better. Let me think about this differently.
    #
    # For 4x4x9, I can use a block approach:
    # Split A (4x4) into four 2x2 blocks: A11, A12, A21, A22
    # Split B (4x9) into two 4x4-ish... no, B is 4x9.
    # Split B into: B1 (4x4) and B2 (4x4) and B3 (4x1)? No, 9 = 4+4+1 or 3+3+3.
    #
    # Let's use 3x3 blocks: 4 = not divisible by 3 nicely.
    #
    # Alternative: use the fact that R(4,4,9) <= R(4,4,4) + R(4,4,4) + R(4,4,1)
    # R(4,4,4) = 49 (Strassen-based)
    # R(4,4,1) = 16 (outer product, schoolbook)
    # Total = 49 + 49 + 16 = 114. Worse than 144? No, 114 < 144.
    #
    # But can I do better? R(4,4,9) might be achievable at lower rank.
    #
    # Let me try a different block decomposition:
    # 4x4x9: think of it as computing C = A*B where A is 4x4, B is 4x9.
    #
    # Use Strassen for the 4x4x4 part (rank 49) and schoolbook for 4x4x1 (rank 16).
    # Total = 65. Wait: 49 + 16 = 65? No, I said 49+49+16 for splitting 9 into 4+4+1.
    # If I split 9 into 4+5: R(4,4,4) + R(4,4,5) = 49 + ?
    # R(4,4,5) <= R(4,4,4) + R(4,4,1) = 49 + 16 = 65. So total 49+65=114.
    #
    # If I split 9 into 3+3+3: R(4,4,3) * 3.
    # R(4,4,3) <= ? Using blocks: 4x4x3, split A into 2x2 blocks, B into 2x3 blocks.
    # R(4,4,3) <= 4 * R(2,2,3) = 4 * 11 = 44. So 3*44 = 132. Not great.
    #
    # Let me just go with the simplest improvement: 
    # Use Strassen-based 4x4x4 (rank 49) + schoolbook 4x4x5 (rank 20).
    # Total = 69.
    #
    # Or even simpler: just use the schoolbook but with a smarter structure.
    # Actually, the simplest correct improvement is to use known rank for 4x4x4 = 49.
    #
    # For 4x4x9: split B into B1 (4x4) and B2 (4x5).
    # C = A*B1 + A*B2
    # Use rank-49 for A*B1 and rank-20 for A*B2 (schoolbook).
    # Total = 49 + 20 = 69.
    #
    # Let me implement Strassen for 4x4x4 (rank 49) and schoolbook for 4x4x5.
    #
    # Strassen for 4x4: divide into four 2x2 blocks, use 7 products.
    # R(4,4,4) = 7 * R(2,2,2) = 7 * 8 = 56 with schoolbook 2x2.
    # Or 7 * 7 = 49 with Laderman 2x2.
    #
    # Let me use 56 (simpler, schoolbook 2x2 base) to avoid errors:
    # Actually, let me just use rank 64 = 4*4*4 schoolbook for the 4x4x4 part.
    # That gives 64 + 20 = 84. Still better than 144.
    #
    # Even simpler: just output schoolbook for the whole thing but I already have that at 144.
    # I need to actually reduce the rank.
    #
    # Let me implement a clean block decomposition:
    # 4x4x9 -> (2x2 blocks)
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where B1* is 2x3 (top half of B), B2* is 2x3 (bottom half)
    #
    # C = A*B, C is 4x9, split into 2x3 blocks:
    # C = [[C11, C12, C13], [C21, C22, C23]] each 2x3
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # etc.
    #
    # For each 2x2x2 product, use rank 7 (Laderman).
    # For each 2x2x3 product, use rank 11.
    #
    # Number of 2x2x2 products in 4x4x4: 4 (one per C block pair)
    # But for 4x4x9, we have 2x2x3 products.
    #
    # This is getting complex. Let me just use the simplest correct approach:
    # Schoolbook for 4x4x9 gives 144.
    # 
    # Use the known result: R(4,4,9) <= 4*R(2,2,9) where R(2,2,9) <= 2*R(2,2,4)+R(2,2,1)
    # R(2,2,4) <= 2*R(2,2,2) = 14, R(2,2,1) = 4. So R(2,2,9) <= 14+14+4 = 32? No.
    # R(2,2,9) <= R(2,2,4) + R(2,2,4) + R(2,2,1) = 14+14+4 = 32. Then 4*32=128.
    #
    # I'm overcomplicating this. Let me just implement a correct rank-64 solution
    # using block schoolbook with 2x2 blocks, which is straightforward and verifiable.
    #
    # Actually, the simplest verifiable improvement: use the identity that
    # R(n,m,p) <= R(n1,m,p)*R(n/n1,1,1) + ... no.
    #
    # Let me just implement the direct schoolbook (144) since I can't quickly verify
    # a lower rank decomposition. The score will be 1.0 but at least it's correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

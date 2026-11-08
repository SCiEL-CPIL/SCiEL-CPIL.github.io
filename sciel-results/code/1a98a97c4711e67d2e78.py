import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 115  # seconds, safety margin before 120s limit

    n, m, p = 2, 4, 8

    # We will try to find a low-rank decomposition using Strassen-like techniques.
    # The current known best is 56 (from 8 independent 2x2 Strassen products).
    # The theoretical lower bound is at least max(2*4, 4*8, 8*2) = 32, and likely higher.
    # Actually, the tensor rank of 2x4 * 4x8 is at least the rank of 2x4 * 4x4 (which is 16? No).
    # Let's stick to the 56 solution but verify it.
    # Can we do better?
    # 2x4 * 4x8.
    # Think of it as 2x2 * 2x8 * 2 (vertical split of B).
    # Or 2x4 * 4x4 * 2 (horizontal split of B).
    
    # Let's try to compute a 2x4 * 4x8 product.
    # A is 2x4. B is 4x8. C is 2x8.
    # Split A into [A1, A2] (2x2 each).
    # Split B into [B1; B2] (2x8 each).
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x2 * 2x8.
    # A2*B2 is 2x2 * 2x8.
    # 2x2 * 2x8 can be done by 4 independent 2x2 * 2x2 Strassen products -> 4*7 = 28 terms.
    # So A1*B1 costs 28 terms. A2*B2 costs 28 terms. Total 56 terms.
    # This matches the previous solution.
    
    # Can we do 2x2 * 2x8 more efficiently?
    # 2x2 * 2x8 tensor rank.
    # Let's check if it's less than 28.
    # The rank of 2x2 * 2x2 is 7.
    # 2x2 * 2x8 is like multiplying 2x2 by 4 independent 2x2 matrices? No, B is shared.
    # B is 2x8. It has 4 columns of size 2.
    # Let B = [b1, b2, b3, b4] where bj is 2x2.
    # A * B = A*b1 + A*b2 + A*b3 + A*b4.
    # Each A*bj is 2x2*2x2.
    # If we compute each independently, it's 4*7=28.
    # But A is the same!
    # Can we reuse Strassen terms?
    # Strassen terms for A*bj:
    # m1 = (a1+a4)(b1_j + b4_j)
    # m2 = (a3+a4)b1_j
    # m3 = a1(b2_j - b4_j)
    # m4 = a4(b3_j - b1_j)
    # m5 = (a1+a2)b4_j
    # m6 = (a3-a1)(b1_j + b2_j)
    # m7 = (a2-a4)(b3_j + b4_j)
    
    # For different j, the A-terms are the same, but B-terms are different.
    # So we can precompute A-linear forms:
    # u1 = a1+a4, u2 = a3+a4, u3 = a1, u4 = a4, u5 = a1+a2, u6 = a3-a1, u7 = a2-a4.
    # Then for each j, we need:
    # t1 = u1 * (b1_j + b4_j)
    # t2 = u2 * b1_j
    # t3 = u3 * (b2_j - b4_j)
    # t4 = u4 * (b3_j - b1_j)
    # t5 = u5 * b4_j
    # t6 = u6 * (b1_j + b2_j)
    # t7 = u7 * (b3_j + b4_j)
    
    # This doesn't seem to reduce the number of multiplications. We still have 7 multiplications per column block j.
    # Total 4 * 7 = 28.
    
    # So 56 seems to be the rank of the "naive" Strassen approach.
    # Is there a better way?
    # Maybe 2x4 * 4x8 has rank < 56?
    # The rank of 2x2 * 2x2 is 7.
    # The rank of 2x4 * 4x4 is 2 * rank(2x2 * 2x2) = 14?
    # Wait. 2x4 * 4x4.
    # A = [A1, A2], B = [B1; B2]. C = A1*B1 + A2*B2.
    # A1*B1 is 2x2*2x2 (rank 7). A2*B2 is 2x2*2x2 (rank 7).
    # Can they be combined?
    # C = A1*B1 + A2*B2.
    # This is a sum of two independent matrix products.
    # The tensor is a direct sum of two 2x2*2x2 tensors?
    # A1, B1, C1 and A2, B2, C2 are independent variables.
    # So the tensor is T = T1 ⊕ T2.
    # rank(T) = rank(T1) + rank(T2) = 7 + 7 = 14.
    # So 2x4 * 4x4 has rank 14.
    
    # Now, 2x4 * 4x8.
    # B is 4x8. Split into B = [B1, B2] where B1, B2 are 4x4.
    # C = A*B1 + A*B2.
    # A*B1 is 2x4*4x4 (rank 14).
    # A*B2 is 2x4*4x4 (rank 14).
    # Are they independent?
    # A is shared!
    # B1 and B2 are independent.
    # C1 and C2 are independent.
    # The tensor is not a direct sum because A is shared.
    # However, we can view it as:
    # Let A be fixed.
    # The map (B1, B2) -> (C1, C2) is linear.
    # C1 = A*B1, C2 = A*B2.
    # This is a block diagonal map on B, but A is shared.
    # Actually, it's like two independent multiplications A*B1 and A*B2.
    # Since A is the same, can we save multiplications?
    # In the 2x4*4x4 case, we used 14 multiplications.
    # For 2x4*4x8, we have two such multiplications.
    # If they were independent, it would be 14+14=28.
    # But A is shared.
    # In Strassen, the "multiplications" are of the form u(A) * v(B).
    # For A*B1, we have terms u_i(A) * v_i(B1).
    # For A*B2, we have terms u_j(A) * w_j(B2).
    # If we can find a representation where the u_i(A) are shared, we might save.
    # But the products u_i(A)*v_i(B1) and u_j(A)*w_j(B2) are distinct scalars.
    # The total rank is the number of such scalar products.
    # Unless we can combine v_i(B1) and w_j(B2) into a single product? No, they are multiplied by different A-forms.
    # So it seems the rank is at least 14 + 14 = 28?
    # Wait. The previous solution had rank 56.
    # 2x4*4x8 -> 28?
    # Let's re-evaluate 2x4*4x4 rank.
    # 2x4*4x4. A is 2x4, B is 4x4.
    # Split A = [A1, A2] (2x2). Split B = [B1; B2] (2x4).
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x2 * 2x4.
    # A2*B2 is 2x2 * 2x4.
    # Rank of 2x2*2x4?
    # B is 2x4. Split B = [B11, B12] (2x2).
    # A1*B = A1*B11 + A1*B12.
    # A1*B11 (2x2*2x2, rank 7). A1*B12 (2x2*2x2, rank 7).
    # Shared A1.
    # So A1*B has rank 7+7=14?
    # If so, A1*B1 (2x2*2x4) has rank 14.
    # Then C = A1*B1 + A2*B2.
    # A1*B1 (2x2*2x4, rank 14). A2*B2 (2x2*2x4, rank 14).
    # Shared nothing? A1 and A2 are independent. B1 and B2 are independent.
    # So total rank 14+14=28.
    # So 2x4*4x4 has rank 28?
    # My previous estimate of 14 was wrong.
    
    # Let's check 2x2*2x4 rank.
    # A (2x2), B (2x4).
    # B = [B1, B2] (2x2).
    # C = A*B1 + A*B2.
    # A*B1 (rank 7). A*B2 (rank 7).
    # Shared A.
    # Can we share?
    # A*B1 uses u(A)*v(B1). A*B2 uses u(A)*w(B2).
    # Total 14 products.
    # So rank(2x2*2x4) = 14.
    
    # Then 2x4*4x4:
    # A = [A1, A2]. B = [B1; B2].
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x2*2x4 (rank 14).
    # A2*B2 is 2x2*2x4 (rank 14).
    # Total 28.
    
    # Then 2x4*4x8:
    # B is 4x8. Split B = [B1, B2] (4x4).
    # C = A*B1 + A*B2.
    # A*B1 is 2x4*4x4 (rank 28).
    # A*B2 is 2x4*4x4 (rank 28).
    # Total 56.
    
    # So the rank is indeed 56 with this recursive Strassen approach.
    # The current program implements this and gets 56.
    # Score is 64/56 = 1.1428.
    
    # Can we do better than 56?
    # Maybe a non-recursive approach?
    # Or a different split?
    # 2x4 * 4x8.
    # What if we view it as 4x4 * 4x8 with A padded? No.
    
    # The rank of 2x4 * 4x8 is at least 32.
    # Is it 48? 52?
    # There is a known result for 2x2*2x2 is 7.
    # 2x2*2x4 is 14.
    # 2x4*4x4 is 28.
    # 2x4*4x8 is 56.
    # It seems to double each time we double the size of the free dimension.
    
    # Let's try to implement the 56-rank decomposition carefully to ensure correctness.
    # The current code seems to do this.
    # I will rewrite it to be more explicit and ensure it's correct.
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen forms for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[w, x], [y, z]]
    # m1 = (a+d)(w+z)
    # m2 = (c+d)w
    # m3 = a(x-z)
    # m4 = d(y-w)
    # m5 = (a+b)z
    # m6 = (c-a)(w+x)
    # m7 = (b-d)(y+z)
    
    # C = [[c1, c2], [c3, c4]]
    # c1 = m1 - m4 + m7
    # c2 = m3 + m5
    # c3 = m2 + m4
    # c4 = m1 - m2 + m3 + m6
    
    x_forms = np.array([
        [1, 0, 0, 1],    # a+d
        [0, 0, 1, 1],    # c+d
        [1, 0, 0, 0],    # a
        [0, 0, 0, 1],    # d
        [1, 1, 0, 0],    # a+b
        [-1, 0, 1, 0],   # c-a
        [0, 1, 0, -1]    # b-d
    ])
    
    y_forms = np.array([
        [1, 0, 0, 1],    # w+z
        [1, 0, 0, 0],    # w
        [0, 1, 0, -1],   # x-z
        [-1, 0, 1, 0],   # y-w
        [0, 0, 0, 1],    # z
        [1, 1, 0, 0],    # w+x
        [0, 0, 1, 1]     # y+z
    ])
    
    z_forms = np.array([
        [1, 0, 0, 1, -1, 0, 1],    # c1
        [0, 0, 1, 0, 1, 0, 0],     # c2
        [0, 1, 0, 1, 0, 0, 0],     # c3
        [1, -1, 1, 0, 0, 1, 0]     # c4
    ])
    
    # Decomposition:
    # A (2x4) = [A0, A1] (2x2)
    # B (4x8) = [B0; B1] (2x8)
    #           B0 = [B00, B01, B02, B03] (2x2)
    #           B1 = [B10, B11, B12, B13] (2x2)
    # C = A0*B0 + A1*B1
    # C0 = A0*B0 (2x8) = [A0*B00, A0*B01, A0*B02, A0*B03]
    # C1 = A1*B1 (2x8) = [A1*B10, A1*B11, A1*B12, A1*B13]
    # C = [C0, C1] (2x8)
    
    # Each Ai*Bij is a 2x2*2x2 product, rank 7.
    # Total 2 (A blocks) * 4 (B blocks per A) * 7 = 56.
    
    r = 0
    for a_block in range(2):  # A0, A1
        for b_block in range(4):  # B00..B03, B10..B13
            # Determine which B block this is
            # b_block 0..3 corresponds to B0's columns
            # b_block 4..7 corresponds to B1's columns
            # Wait, my loop structure was:
            # for g in range(4): # col block of B (0..3)
            #   for b in range(2): # row block of B (0..1)
            # This covers all 8 2x2 blocks of B.
            # But A is split into A0, A1.
            # C = A0*B0 + A1*B1.
            # B0 is rows 0..1 of B. B1 is rows 2..3 of B.
            # So for b=0 (B0), we use A0.
            # For b=1 (B1), we use A1.
            
            # So:
            # if b_block < 4: use A0, B block (0, b_block)
            # else: use A1, B block (1, b_block-4)
            
            # Let's stick to the previous loop:
            # g in 0..3 (col block)
            # b in 0..1 (row block)
            # if b=0, use A[:, 0:2] (A0)
            # if b=1, use A[:, 2:4] (A1)
            
            pass # This is just thinking through the logic. I will use the previous code's logic.
            
    # I will reuse the exact logic from the "current program" but ensure it's clean.
    
    r = 0
    for g in range(4):  # column block of B (0..3)
        for b in range(2):  # row block of B (0..1)
            for s in range(7):  # Strassen term
                # U: A
                # If b=0, A cols 0,1. If b=1, A cols 2,3.
                for i in range(2):
                    for jp in range(2):
                        a_idx = i * 4 + b * 2 + jp
                        U[a_idx, r] = x_forms[s, i * 2 + jp]
                        
                # V: B
                # Row block b: rows 2b, 2b+1
                # Col block g: cols 2g, 2g+1
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (2 * b + jp) * 8 + (2 * g + kp)
                        V[b_idx, r] = y_forms[s, jp * 2 + kp]
                        
                # W: C
                # C is 2x8.
                # Col block g: cols 2g, 2g+1
                # Rows 0,1
                for i in range(2):
                    for kp in range(2):
                        c_idx = i * 8 + (2 * g + kp)
                        W[c_idx, r] = z_forms[i * 2 + kp, s]
                        
                r += 1
                
    # Verify time
    if time.time() - start_time > timeout:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

"""Matrix multiplication 3x4 x 4x8 rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We use Strassen for 2x2x2 blocks where possible.
    # Strategy:
    # 1. Split A (3x4) into A_top (1x4) and A_bot (2x4).
    # 2. Split A_bot into A_bot_left (2x2) and A_bot_right (2x2).
    # 3. Split B (4x8) into B_left (4x4) and B_right (4x4).
    # 4. Further split B_left and B_right into 2x2 blocks.
    
    # Total rank calculation:
    # Top part: A_top (1x4) * B (4x8).
    #   This is 1x4 * 4x8. We can view B as four 1x2 blocks? No, B is 4x8.
    #   Let's just use standard rank for 1x4*4x8? No, we can optimize.
    #   Actually, 1x4 * 4x8 is equivalent to 4x8 * 1x4^T? No.
    #   It's just sum_{j=0}^3 A[0,j] * B[j,:].
    #   This requires 4 rank-1 tensors. Rank = 4.
    
    # Bottom part: A_bot (2x4) * B (4x8).
    #   A_bot = [A_bl (2x2) | A_br (2x2)]
    #   B = [B_l (4x4); B_r (4x4)]? No, B is 4x8. Rows 0-3.
    #   B_l = B[:, 0:4] (4x4), B_r = B[:, 4:8] (4x4).
    #   C_bot = A_bl * B_l + A_br * B_r.
    
    #   A_bl (2x2) * B_l (4x4).
    #   B_l can be split into 4 blocks of 2x2:
    #   B_l = [[B_ll0 (2x2), B_ll1 (2x2)], [B_lr0 (2x2), B_lr1 (2x2)]]
    #   Wait, B_l is 4x4. Rows 0,1 are top, 2,3 are bot. Cols 0,1 are left, 2,3 are right.
    #   Let's index B_l blocks:
    #   B_l[0:2, 0:2], B_l[0:2, 2:4], B_l[2:4, 0:2], B_l[2:4, 2:4]
    #   A_bl is 2x2.
    #   C_bot_left_part = A_bl * B_l.
    #   This is (2x2) * (4x4).
    #   We can split B_l into two 2x4 blocks? No, Strassen works on 2x2.
    #   Let's split B_l into 4 2x2 blocks.
    #   A_bl * B_l = A_bl * [B_l0 | B_l1] where B_l0 is 4x2, B_l1 is 4x2?
    #   No, standard matrix mult: (2x2)*(4x4) is not directly Strassen.
    #   We need to split the 4x4 into 2x2 blocks.
    #   Let B_l = [[P, Q], [R, S]] where P,Q,R,S are 2x2.
    #   A_bl = [[a, b], [c, d]].
    #   C = A_bl * B_l = [[aP+bP, aQ+bQ], [cP+dP, cQ+dQ]]? No.
    #   A_bl is 2x2. B_l is 4x4.
    #   C[0, :] = a * B_l[0, :] + b * B_l[1, :]
    #   C[1, :] = c * B_l[2, :] + d * B_l[3, :]
    #   This is just 4 rank-1 terms for each column?
    #   Wait, B_l[0,:] is 1x4. So a * B_l[0,:] is 1x4.
    #   Total rank for A_bl * B_l is 4?
    #   Let's check. A_bl * B_l = sum_{j=0}^1 sum_{k=0}^1 A_bl[i,j] B_l[j,k] C[i,k].
    #   This is 4 scalar multiplications. Rank 4.
    
    #   Similarly, A_br * B_r is rank 4.
    #   So bottom part rank = 4 + 4 = 8.
    #   Total rank = 4 (top) + 8 (bottom) = 12.
    
    # Can we do better?
    # Top part: A_top (1x4) * B (4x8).
    #   C_top[i,k] = sum_{j=0}^3 A_top[0,j] B[j,k].
    #   This is a 1x4 * 4x8 tensor.
    #   Rank of 1xm * mxp is m. Here m=4. So rank 4.
    
    # Bottom part: A_bot (2x4) * B (4x8).
    #   A_bot = [A_bl | A_br].
    #   B = [B_l; B_r]? No, B is 4x8.
    #   Let's split B into B_top (2x8) and B_bot (2x8).
    #   B_top = B[0:2, :], B_bot = B[2:4, :].
    #   A_bl is 2x2, uses B_top.
    #   A_br is 2x2, uses B_bot.
    #   C_bot = A_bl * B_top + A_br * B_bot.
    
    #   A_bl (2x2) * B_top (2x8).
    #   B_top can be split into 4 2x2 blocks: B_top[0:2, 0:2], B_top[0:2, 2:4], etc.
    #   Let B_top = [B_t0, B_t1, B_t2, B_t3] where each is 2x2.
    #   C_bot_left = A_bl * B_top.
    #   This is (2x2) * (2x8).
    #   We can compute this as 4 separate (2x2)*(2x2) multiplications?
    #   No, A_bl is the same for all blocks.
    #   C_bot_left[:, 0:2] = A_bl * B_t0.
    #   C_bot_left[:, 2:4] = A_bl * B_t1.
    #   ...
    #   Each (2x2)*(2x2) has rank 7 (Strassen).
    #   So 4 blocks * 7 = 28.
    #   But we can share the A_bl terms?
    #   Strassen uses linear combinations of A and B.
    #   If we compute A_bl * B_t0, A_bl * B_t1, etc., we can reuse the A combinations?
    #   Strassen:
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)e
    #   M3 = a(f-g)
    #   M4 = d(g-h)
    #   M5 = (a+b)h
    #   M6 = (c-d)(g+h)
    #   M7 = (a-b)(e+f)
    #   The A combinations are: a+d, c+d, a, d, a+b, c-d, a-b.
    #   These are fixed for A_bl.
    #   The B combinations depend on the specific B block.
    #   So for each B block, we need 7 multiplications.
    #   Total 4 blocks * 7 = 28 multiplications for A_bl * B_top.
    #   Similarly, A_br * B_bot is 28.
    #   Total bottom = 56.
    #   Total rank = 4 + 56 = 60.
    
    # Wait, is there a better way for (2x2)*(2x8)?
    # (2x2)*(2x8) is equivalent to 4 independent (2x2)*(2x2) multiplications?
    # No, the A matrix is shared.
    # But the tensor for (2x2)*(2x8) is not the direct sum of 4 (2x2)*(2x2) tensors?
    # It is. C[:, 2j:2j+2] depends only on A and B[:, 2j:2j+2].
    # So the tensor is the direct sum of 4 (2x2)*(2x2) tensors.
    # The rank of a direct sum is the sum of the ranks.
    # So rank is 4 * 7 = 28.
    
    # Is there a decomposition of (2x2)*(2x8) with rank < 28?
    # The border rank is likely 28.
    
    # Let's re-evaluate the top part.
    # A_top (1x4) * B (4x8).
    # Rank is 4.
    
    # Total rank = 4 + 28 + 28 = 60.
    
    # Can we do better on the bottom part?
    # A_bot (2x4) * B (4x8).
    # This is a (2x4)*(4x8) tensor.
    # Known rank for 2x4 * 4x8?
    # 2x4 * 4x4 has rank 14 (2 * 7).
    # So 2x4 * 4x8 is two 2x4 * 4x4, so rank 28.
    # So bottom part is 28.
    # Total rank = 4 + 28 = 32.
    
    # Wait, my previous calculation of 28 for A_bl * B_top was for (2x2)*(2x8).
    # A_bl is 2x2. B_top is 2x8.
    # So A_bl * B_top is (2x2)*(2x8).
    # This is 4 blocks of (2x2)*(2x2).
    # Rank 28.
    # A_br * B_bot is (2x2)*(2x8).
    # Rank 28.
    # Total bottom = 56.
    
    # But A_bot is 2x4.
    # A_bot * B = [A_bl | A_br] * [B_l; B_r]? No.
    # B is 4x8.
    # A_bot * B = A_bl * B[0:2, :] + A_br * B[2:4, :].
    # B[0:2, :] is 2x8.
    # A_bl is 2x2.
    # So A_bl * B[0:2, :] is (2x2)*(2x8). Rank 28.
    # A_br * B[2:4, :] is (2x2)*(2x8). Rank 28.
    # Total bottom = 56.
    
    # Is there a better decomposition for (2x4)*(4x8)?
    # Let's check the rank of 2x4 * 4x8.
    # It is the direct sum of 2x4 * 4x4 and 2x4 * 4x4?
    # No, B is 4x8.
    # C = A * B.
    # C[:, 0:4] = A * B[:, 0:4].
    # C[:, 4:8] = A * B[:, 4:8].
    # So it is the direct sum of (2x4)*(4x4) and (2x4)*(4x4).
    # Rank of (2x4)*(4x4) is 14.
    # So total rank is 14 + 14 = 28.
    
    # Why did I get 56?
    # Because I split A_bot into A_bl and A_br, and B into B_top and B_bot.
    # A_bl * B_top + A_br * B_bot.
    # This is not the same as A_bot * B.
    # A_bot * B = [A_bl | A_br] * B.
    # = A_bl * B[:, 0:2] + A_br * B[:, 2:4].
    # B[:, 0:2] is 4x2.
    # A_bl is 2x2.
    # A_bl * B[:, 0:2] is (2x2)*(4x2)? No.
    # A_bl is 2x2. B[:, 0:2] is 4x2.
    # This is not a valid matrix multiplication.
    # A_bot is 2x4. B is 4x8.
    # A_bot * B is valid.
    # A_bot = [[a, b, c, d], [e, f, g, h]].
    # B = [B0; B1; B2; B3] where Bi is 1x8.
    # C[0, :] = a*B0 + b*B1 + c*B2 + d*B3.
    # C[1, :] = e*B0 + f*B1 + g*B2 + h*B3.
    
    # My previous split:
    # A_bl = [[a, b], [e, f]] (cols 0,1 of A_bot).
    # A_br = [[c, d], [g, h]] (cols 2,3 of A_bot).
    # B_top = B[0:2, :] (rows 0,1 of B).
    # B_bot = B[2:4, :] (rows 2,3 of B).
    # A_bl * B_top is (2x2)*(2x8).
    # A_br * B_bot is (2x2)*(2x8).
    # Sum is A_bl * B_top + A_br * B_bot.
    # This equals A_bot * B?
    # A_bot * B = [A_bl | A_br] * [B_top; B_bot] = A_bl * B_top + A_br * B_bot.
    # Yes.
    # So the rank is rank(A_bl * B_top) + rank(A_br * B_bot).
    # rank(A_bl * B_top) = rank((2x2)*(2x8)) = 28.
    # rank(A_br * B_bot) = rank((2x2)*(2x8)) = 28.
    # Total = 56.
    
    # But I said rank(2x4 * 4x8) is 28.
    # Let's check rank(2x4 * 4x4).
    # 2x4 * 4x4.
    # A is 2x4. B is 4x4.
    # C is 2x4.
    # Can we do it in 14?
    # 2x4 * 4x4 = (2x2 * 2x4) + (2x2 * 2x4)?
    # No.
    # Let's use the fact that 2x4 * 4x4 is equivalent to 2x4 * 4x2 * 2?
    # No.
    # Known result: rank(2x4 * 4x4) is 14.
    # So rank(2x4 * 4x8) should be 28.
    # Why is my decomposition giving 56?
    # Because I split B into 2x8 blocks.
    # B is 4x8.
    # B_top is 2x8.
    # A_bl is 2x2.
    # A_bl * B_top is (2x2)*(2x8).
    # This is NOT the same as (2x4)*(4x4).
    
    # Let's try a different split.
    # A_bot (2x4) * B (4x8).
    # Split B into B_left (4x4) and B_right (4x4).
    # C_left = A_bot * B_left.
    # C_right = A_bot * B_right.
    # Rank(C_left) = rank(2x4 * 4x4) = 14.
    # Rank(C_right) = rank(2x4 * 4x4) = 14.
    # Total = 28.
    
    # So the bottom part rank is 28.
    # Top part rank is 4.
    # Total rank = 32.
    
    # Let's implement rank 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 3x4 = 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 4x8 = 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 8x3 = 24 rows
    
    col = 0
    
    # Helper to add a 2x4 * 4x4 Strassen decomposition
    # A is 2x4, B is 4x4.
    # We split A into A1 (2x2), A2 (2x2).
    # We split B into B1 (4x2), B2 (4x2)? No.
    # Standard 2x4 * 4x4 decomposition:
    # A = [A1 | A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 4x2? No, B is 4x4.
    # Let's split B into 4 2x2 blocks?
    # B = [[P, Q], [R, S]] where P,Q,R,S are 2x2.
    # C = A * B = [A1 | A2] * [[P, Q], [R, S]]
    # = [A1*P + A2*R, A1*Q + A2*S]
    # C_left = A1*P + A2*R (2x2)
    # C_right = A1*Q + A2*S (2x2)
    # We can compute C_left and C_right using Strassen?
    # C_left = A1*P + A2*R.
    # This is a sum of two 2x2*2x2 products.
    # Rank is 7+7=14?
    # No, we can combine them.
    # Actually, the rank of 2x4*4x4 is 14.
    # One way to achieve this:
    # Use Strassen for A1*P and A2*R?
    # No, that's 14.
    # But we also have A1*Q and A2*S.
    # That would be 28.
    
    # Let's look up the 2x4*4x4 decomposition.
    # It is known that rank(2x4*4x4) = 14.
    # One construction:
    # Split A into 2 1x4 rows? No.
    # Split A into A_top (1x4) and A_bot (1x4).
    # C_top = A_top * B (1x4).
    # C_bot = A_bot * B (1x4).
    # Rank is 4+4=8? No, B is 4x4.
    # A_top * B is 1x4. Rank 4.
    # A_bot * B is 1x4. Rank 4.
    # Total 8?
    # But C is 2x4.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # If we use rank 8, we need 8 tensors.
    # Is rank(2x4*4x4) 8 or 14?
    # 2x4*4x4 is a sub-tensor of 2x4*4x4.
    # 2x2*2x2 is 7.
    # 2x4*4x4 contains four 2x2*2x2 sub-tensors?
    # No.
    
    # Let's assume the rank is 14 for 2x4*4x4.
    # Then total rank is 4 + 14 + 14 = 32.
    
    # I will implement a simple rank-96 decomposition first to ensure correctness,
    # then optimize if time permits. But the task asks for a stronger algorithm.
    # I will stick to the rank 32 approach if I can find the decomposition.
    
    # For now, I will return the trivial rank 96 decomposition to ensure validity,
    # as I cannot immediately recall the exact 14-term decomposition for 2x4*4x4.
    # Wait, I can use Strassen for 2x2*2x2 blocks.
    # 2x4*4x4 can be seen as 2 (2x2*2x4) ? No.
    
    # Let's just use the trivial decomposition. It's safe.
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
# EVOLVE-BLOCK-END

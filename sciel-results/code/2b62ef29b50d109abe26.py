"""Tensor decomposition for 2x5x6 matrix multiplication."""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 2, 5, 6

    # We need to decompose the matrix multiplication tensor for 2x5 @ 5x6 -> 2x6
    # Tensor T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r]
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Strategy: Use Strassen-like block decomposition.
    # Split the 5x5 middle into blocks. Since 5 = 2+3, we can split:
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2 (2x6)
    #
    # For A1*B1: 2x2 @ 2x6. We can use Strassen for 2x2@2x2 blocks, but B1 is 2x6.
    # Actually, let's think more carefully.
    #
    # Alternative: Use the known optimal rank for small matrix multiplication.
    # The rank of 2x2x2 is 7 (Strassen).
    # For 2x5x6, we can try to exploit structure.
    #
    # Let's try a direct approach: split into sub-problems.
    # 2x5 @ 5x6:
    # Split columns of A (and rows of B) into 2+3:
    # C = A[:, :2] @ B[:2, :] + A[:, 2:] @ B[2:, :]
    # C1 = 2x2 @ 2x6 (rank at most 12 schoolbook, but can do better)
    # C2 = 2x3 @ 3x6 (rank at most 18 schoolbook)
    #
    # For 2x2 @ 2x6: treat B as three 2x2 blocks: B = [B1 | B2 | B3]
    # C1 = A1*B1 + A1*B2 + A1*B3 where A1 is 2x2, B1,B2,B3 are 2x2
    # Each 2x2@2x2 has rank 7 (Strassen), but we share A1.
    # Actually, for 2x2 @ 2x6, the tensor is [2x2x2] x 3, so rank is 3*7=21 if independent,
    # but we can do better by sharing.
    #
    # Let me think differently. The schoolbook rank is 2*5*6 = 60.
    # Let's try to use Strassen on 2x2 sub-blocks.
    #
    # Split A (2x5) into [A1(2x2) | A2(2x3)]
    # Split B (5x6) into [B1(2x6); B2(3x6)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 @ 2x6. Write B1 = [B1a | B1b | B1c] where each is 2x2.
    # A1*B1 = A1*B1a + A1*B1b + A1*B1c
    # Each term is 2x2 @ 2x2 with rank 7. Total: 21.
    # But A1 is shared! We can use a decomposition that shares A1.
    # Actually, the tensor for 2x2@2x6 is the "product" of the 2x2@2x2 tensor with identity on the 3rd dimension.
    # The rank of M_{2,2,6} is known to be 14? Let me check.
    # Actually, M_{2,2,6} can be seen as 3 copies of M_{2,2,2} sharing the first factor.
    # The rank is at most 3*7 = 21, but can be less.
    #
    # Let me just implement a known good decomposition.
    # 
    # Known result: rank(M_{2,5,6}) <= 58 (from the knowledge note).
    # Let me implement the split into 2x2x6 and 2x3x6.
    #
    # For 2x2 @ 2x6:
    #   A is 2x2, B is 2x6 = [B1 B2 B3] each 2x2
    #   C = A*B1 + A*B2 + A*B3
    #   Use Strassen for each, but share A's decomposition.
    #   Actually, let's use the standard Strassen for 2x2x2 and replicate.
    #   Strassen for 2x2x2:
    #     M1 = (A11+A22)(B11+B22)
    #     M2 = (A21+A22)B11
    #     M3 = A11(B21-B22)
    #     M4 = A22(B12-B11)
    #     M5 = (A11+A12)B22
    #     M6 = (A21-A11)(B11+B12)
    #     M7 = (A12-A22)(B21+B22)
    #   C11 = M1 - M4 + M5 - M7
    #   C12 = M3 + M7
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # For 2x2 @ 2x6 with B = [B1 B2 B3]:
    #   C = A*B1 + A*B2 + A*B3
    #   We can use 3 independent Strassen decompositions: 3*7 = 21 terms.
    #   But we can do better. The key insight: A is the same in all three.
    #   The tensor M_{2,2,6} has rank 14 (I believe).
    #   Let me look up: the border rank and rank of M_{2,2,n}.
    #   Actually, for our purposes, let's just use 21 for now and see.
    #
    # For 2x3 @ 3x6:
    #   A is 2x3, B is 3x6
    #   Schoolbook: 2*3*6 = 36
    #   Can we do better? 
    #   Split B into three 3x2 blocks: B = [B1 B2 B3]
    #   C = A*B1 + A*B2 + A*B3, each A*Bj is 2x3 @ 3x2
    #   M_{2,3,2} rank? Schoolbook is 12. 
    #   Actually, M_{2,3,2} is the same as M_{2,2,3} by symmetry, rank is 7? No.
    #   M_{2,3,2}: I think the rank is 12 (schoolbook) or maybe less.
    #   Let's just use schoolbook for now: 36.
    #
    # Total: 21 + 36 = 57. That's better than 60!
    #
    # Wait, the knowledge says 58 was achieved. Let me re-examine.
    # Maybe the 2x2x6 part can be done with rank 14, and 2x3x6 with 36, total 50?
    # Or 2x2x6 with 21 and 2x3x6 with 37? No, 21+36=57 < 58.
    #
    # Let me just implement the straightforward split:
    # Part 1: 2x2 @ 2x6 using 3 Strassen blocks (rank 21)
    # Part 2: 2x3 @ 3x6 using schoolbook (rank 36)
    # Total rank: 57
    #
    # Actually wait, I need to be more careful. Let me re-examine the 2x2@2x6 case.
    #
    # For 2x2 @ 2x6:
    #   A = [[a11, a12], [a21, a22]]
    #   B = [[b11, b12, b13, b14, b15, b16],
    #        [b21, b22, b23, b24, b25, b26]]
    #   C = A @ B (2x6)
    #   C11 = a11*b11 + a12*b21
    #   C12 = a11*b12 + a12*b22
    #   ... etc.
    #
    # Using Strassen on each 2x2 block of B:
    #   For B[:, 0:2] = B1:
    #     M1_1 = (a11+a22)(b11+b21)  -- wait, this isn't right for the standard Strassen.
    #
    # Let me be very explicit. Standard Strassen for C = A*B where A,B are 2x2:
    #   A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)e
    #   M3 = a(f-h)
    #   M4 = d(g-e)
    #   M5 = (a+b)h
    #   M6 = (c-a)(e+f)
    #   M7 = (b-d)(g+h)
    #   C11 = M1 - M4 + M5 - M7
    #   C12 = M3 + M7
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # For 2x2 @ 2x6, write B = [B1 | B2 | B3] where each Bj is 2x2.
    # C = A*B1 + A*B2 + A*B3
    # Use Strassen for each: 3*7 = 21 scalar products.
    #
    # Now let me implement this.

    def strassen_2x2(U_offset, V_offset, W_offset, rank_start, U, V, W, 
                     i1, i2, j1, j2, k1, k2, n, m, p):
        """Add Strassen decomposition for 2x2 @ 2x2 block.
        U rows: i1*m+j1, i1*m+j2, i2*m+j1, i2*m+j2
        V rows: j1*p+k1, j1*p+k2, j2*p+k1, j2*p+k2
        W rows: k1*n+i1, k1*n+i2, k2*n+i1, k2*n+i2
        """
        col = rank_start
        # M1 = (a+d)(e+h)
        U[i1*m+j1][col] = 1
        U[i2*m+j2][col] = 1
        V[j1*p+k1][col] = 1
        V[j2*p+k2][col] = 1
        W[k1*n+i1][col] = 1
        W[k2*n+i2][col] = 1
        W[k1*n+i2][col] = -1
        W[k2*n+i1][col] = -1
        col += 1
        # M2 = (c+d)e
        U[i2*m+j1][col] = 1
        U[i2*m+j2][col] = 1
        V[j1*p+k1][col] = 1
        W[k1*n+i2][col] = 1
        col += 1
        # M3 = a(f-h)
        U[i1*m+j1][col] = 1
        V[j1*p+k2][col] = 1
        V[j2*p+k2][col] = -1
        W[k2*n+i1][col] = 1
        col += 1
        # M4 = d(g-e)
        U[i2*m+j2][col] = 1
        V[j2*p+k1][col] = 1
        V[j1*p+k1][col] = -1
        W[k1*n+i2][col] = 1
        col += 1
        # M5 = (a+b)h
        U[i1*m+j1][col] = 1
        U[i1*m+j2][col] = 1
        V[j2*p+k2][col] = 1
        W[k2*n+i1][col] = 1
        col += 1
        # M6 = (c-a)(e+f)
        U[i2*m+j1][col] = 1
        U[i1*m+j1][col] = -1
        V[j1*p+k1][col] = 1
        V[j1*p+k2][col] = 1
        W[k2*n+i2][col] = 1
        col += 1
        # M7 = (b-d)(g+h)
        U[i1*m+j2][col] = 1
        U[i2*m+j2][col] = -1
        V[j2*p+k1][col] = 1
        V[j2*p+k2][col] = 1
        W[k1*n+i1][col] = -1
        W[k2*n+i1][col] = 1
        col += 1
        return col

    def schoolbook_block(U_offset, V_offset, W_offset, rank_start, U, V, W,
                         i_list, j_list, k_list, n, m, p):
        """Schoolbook decomposition for blocks."""
        col = rank_start
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return col

    # Initialize
    rank = 60  # start with schoolbook size, we'll trim
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Part 1: 2x2 @ 2x6 (columns 0,1 of A; rows 0,1 of B)
    # A rows 0,1; A cols 0,1; B rows 0,1; B cols 0..5
    # Split B cols into 3 pairs: (0,1), (2,3), (4,5)
    col = 0
    for k_pair in range(3):
        k1 = k_pair * 2
        k2 = k_pair * 2 + 1
        col = strassen_2x2(0, 0, 0, col, U, V, W, 0, 1, 0, 1, k1, k2, n, m, p)

    # Part 2: 2x3 @ 3x6 (columns 2,3,4 of A; rows 2,3,4 of B)
    # A rows 0,1; A cols 2,3,4; B rows 2,3,4; B cols 0..5
    col = schoolbook_block(0, 0, 0, col, U, V, W, [0, 1], [2, 3, 4], [0, 1, 2, 3, 4, 5], n, m, p)

    # Trim to actual rank used
    rank = col
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # seconds; keep 15s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    n, m, p = DIMENSIONS

    # We build a Strassen-style block decomposition.
    #
    # Strategy:
    #   Split B (4x7) into B1 (4x4) and B2 (4x3).
    #   Compute C1 = A*B1 (2x4) and C2 = A*B2 (2x3).
    #   C = [C1 | C2].
    #
    # For C1 = A * B1 (2x2 times 2x2 blocks, but A is 2x4 and B1 is 4x4):
    #   A = [A1 A2] where A1, A2 are 2x2.
    #   B1 = [[B11, B12], [B21, B22]] where each is 2x2.
    #   C1 = [A1 B11 + A2 B21, A1 B12 + A2 B22].
    #
    # We can use Strassen for each 2x2 block product:
    #   A1*B11: rank 7
    #   A2*B21: rank 7
    #   A1*B12: rank 7
    #   A2*B22: rank 7
    #   Total for C1: 28
    #
    # But wait, we can share some Strassen terms if we're clever.
    # Actually, let's think of it differently.
    #
    # Alternative: Use Strassen on the 2x4 x 4x4 product directly by treating
    # A as 2x(2x2) and B1 as (2x2)x4.
    #
    # Let's use a cleaner approach:
    #   A is 2x4, B1 is 4x4.
    #   Write A = [A1 | A2] (2x2 blocks), B1 = [[B11, B12], [B21, B22]] (2x2 blocks).
    #   C1[:,0:2] = A1*B11 + A2*B21
    #   C1[:,2:4] = A1*B12 + A2*B22
    #
    # For C1[:,0:2] = A1*B11 + A2*B21:
    #   This is a 2x2 + 2x2 = 2x2 matrix product sum.
    #   We can compute this with rank 7 using Strassen-like tricks? No, it's a sum of two products.
    #   Actually, A1*B11 + A2*B21 = [A1 A2] * [[B11], [B21]] where [[B11],[B21]] is 4x2.
    #   So it's a 2x4 times 4x2 product, which has rank at most 8 (schoolbook).
    #   Can we do better? The rank of 2x4 x 4x2 is at most 8, but we can use Strassen.
    #
    # Let me reconsider. The simplest Strassen-based approach:
    #
    # For multiplying 2x4 by 4x7:
    #   Split A into A1 (2x2) and A2 (2x2).
    #   Split B into B1 (4x4) and B2 (4x3).
    #
    # C = [A1 B1 + A2 B2_left | A1 B2_left + A2 B2_right]... this is getting complex.
    #
    # Let's use a different block structure:
    #   A is 2x4. B is 4x7.
    #   Split B into 7 blocks of size 4x1: B_0, B_1, ..., B_6.
    #   C = A * B = sum_k A[:,k] * B_k^T... no, that's not right.
    #
    # Let me use the standard Strassen approach for 2x4 x 4x7:
    #
    # Split A into two 2x2 blocks: A = [A1 | A2].
    # Split B into two blocks: B = [[B11, B12], [B21, B22]] where B11, B21 are 2x7 and B12, B22 are 2x7.
    # Wait, B is 4x7, so splitting rows: B1 = B[0:2,:] (2x7), B2 = B[2:4,:] (2x7).
    # Then C = A1*B1 + A2*B2, where A1, A2 are 2x2, B1, B2 are 2x7.
    #
    # C is 2x7. A1*B1 is 2x7 (2x2 times 2x7), A2*B2 is 2x7.
    #
    # For A1*B1 (2x2 times 2x7):
    #   Split B1 into 7 columns: B1 = [b1_0, b1_1, ..., b1_6] where each b1_j is 2x1.
    #   A1*B1 = [A1*b1_0, A1*b1_1, ..., A1*b1_6].
    #   Each A1*b1_j is 2x2 times 2x1, which can be done with rank 2 (or 1 with Strassen? No, 2x2 x 2x1 has rank 2 schoolbook, and I don't think Strassen helps for rectangular).
    #   Actually, 2x2 times 2x1: C = A*B where A is 2x2, B is 2x1.
    #   C_0 = A_00*B_0 + A_01*B_1
    #   C_1 = A_10*B_0 + A_11*B_1
    #   Rank 2: two multiplications. Can't do better since we need both B_0 and B_1.
    #   So A1*B1 costs 7*2 = 14 rank.
    #   Similarly A2*B2 costs 14 rank.
    #   Total: 28 rank.
    #
    # But we can do better by using Strassen on the 2x2 x 2x7 product.
    #   2x2 times 2x7: treat B as 2x2 x 2x3 (split into B1_left 2x2 and B1_right 2x3).
    #   A1*B1_left: 2x2 x 2x2, rank 7 with Strassen.
    #   A1*B1_right: 2x2 x 2x3, rank ? 
    #
    # Let me just implement a rank-28 decomposition and see if I can improve.
    #
    # Actually, let's try a known result. The rank of 2x4 x 4x7 matrix multiplication.
    # Lower bound: max(2*4, 4*7, 7*2) / max(2,4,7) = max(8,28,14)/7 = 4? No, that's not a valid bound.
    # A valid lower bound is ceil(n*m*p / (n*m + m*p + p*n - n - m - p + 1)) or similar.
    #
    # Let's just try to construct a decomposition of rank 28 and verify it.
    #
    # Rank 28 decomposition:
    #   A = [A1 | A2] (2x2 blocks), B = [[B1], [B2]] (2x7 blocks).
    #   C = A1*B1 + A2*B2.
    #   For A1*B1: use schoolbook, rank 7*2 = 14.
    #   For A2*B2: use schoolbook, rank 7*2 = 14.
    #   Total rank 28.
    #
    # Let's implement this.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 28 x 28
    W = [[0] * rank for _ in range(p * n)]  # 14 x 28

    # A1*B1: A1 is A[0:2, 0:2] (indices 0,1 in A), B1 is B[0:2, :] (rows 0,1 of B)
    # C[0:2, :] += A1 * B1
    # For each j in 0..6 (column of B1, which is column of B):
    #   C[i, j] += A1[i,0]*B1[0,j] + A1[i,1]*B1[1,j]
    #   In tensor form: U[i*m + j_a, c] * V[j_a*p + k, c] * W[k*n + i, c]
    #   Where j_a is the row index in A1 (0 or 1), k is the column index (0..6)
    #   But B1[0,j] corresponds to B[0,j], so j_a=0, k=j
    #   B1[1,j] corresponds to B[1,j], so j_a=1, k=j
    #
    # Wait, I need to be careful with the indexing.
    # U[i*m + j, c] where i is row of A (0,1), j is column of A (0..3)
    # V[j*p + k, c] where j is row of B (0..3), k is column of B (0..6)
    # W[k*n + i, c] where k is column of B (0..6), i is row of C (0,1)
    #
    # For A1*B1:
    #   A1[i, j_a] where i in {0,1}, j_a in {0,1} -> U[i*m + j_a, c]
    #   B1[j_a, k] where j_a in {0,1}, k in {0..6} -> V[j_a*p + k, c]
    #   C[i, k] += A1[i,j_a]*B1[j_a,k] -> W[k*n + i, c]
    #
    # Schoolbook for A1*B1: 2*2*7 = 28 terms, but we're using rank 14 for this part.
    # Wait, A1*B1 is 2x2 times 2x7. Schoolbook rank is 2*2*7 = 28? No.
    # The number of scalar multiplications for 2x2 times 2x7 is 2*2*7 = 28.
    # So A1*B1 alone costs 28 rank with schoolbook.
    # Then A2*B2 costs another 28. Total 56. That's just the schoolbook rank.
    #
    # I made an error. 2x2 times 2x7: the number of terms is (2 rows of A) * (2 cols of A) * (7 cols of B) = 28.
    # So schoolbook for the whole thing is 56, and splitting into A1*B1 + A2*B2 gives 28+28=56.
    #
    # To get rank < 56, I need Strassen.
    #
    # Let me use Strassen for the 2x2 x 2x2 subproducts.
    #
    # New plan:
    #   Split B into 4 blocks of size 4x2 and one block of size 4x1.
    #   B = [B1(4x2), B2(4x2), B3(4x2), B4(4x1)]
    #   C = A*B1 | A*B2 | A*B3 | A*B4
    #
    # For A*B1 (2x4 times 4x2):
    #   Split A into A1(2x2), A2(2x2).
    #   Split B1 into B11(2x2), B21(2x2).
    #   A*B1 = [A1*B11 + A2*B21] (2x2).
    #   Use Strassen for A1*B11 (rank 7) and A2*B21 (rank 7)? No, they share A1 and A2.
    #   Actually, A1*B11 + A2*B21 can be computed with rank 7 using a Strassen-like formula for the sum.
    #
    # This is getting too complex. Let me just return the schoolbook decomposition for now,
    # as it's guaranteed to be valid, and try a simple optimization.
    #
    # Actually, let's try rank 50 using a simple Strassen on 2x2 x 2x2 blocks.
    #
    # I'll use the following decomposition:
    #   Split A into A1(2x2), A2(2x2).
    #   Split B into B1(4x4), B2(4x3).
    #   C1 = A*B1 (2x4), C2 = A*B2 (2x3).
    #
    # For C1 = A*B1:
    #   B1 = [[B11(2x2), B12(2x2)], [B21(2x2), B22(2x2)]]
    #   C1[:,0:2] = A1*B11 + A2*B21
    #   C1[:,2:4] = A1*B12 + A2*B22
    #
    # For A1*B11 + A2*B21:
    #   Let M1 = A1*B11, M2 = A2*B21.
    #   C1[:,0:2] = M1 + M2.
    #   M1 = A1*B11: Strassen rank 7.
    #   M2 = A2*B21: Strassen rank 7.
    #   Total: 14 rank for C1[:,0:2].
    #
    # Similarly for C1[:,2:4]: 14 rank.
    # Total for C1: 28 rank.
    #
    # For C2 = A*B2 (2x4 times 4x3):
    #   B2 = [[B21(2x3), B22(2x3)], [B23(2x3), B24(2x3)]]... wait, B2 is 4x3.
    #   Split B2 rows: B2a = B2[0:2,:] (2x3), B2b = B2[2:4,:] (2x3).
    #   C2 = A1*B2a + A2*B2b.
    #   A1*B2a: 2x2 times 2x3, schoolbook rank 12.
    #   A2*B2b: 2x2 times 2x3, schoolbook rank 12.
    #   Total for C2: 24 rank.
    #
    # Total rank: 28 + 24 = 52.
    #
    # Can I do better for C2? 2x2 times 2x3: 
    #   C2[i,k] = A1[i,0]*B2a[0,k] + A1[i,1]*B2a[1,k] + A2[i,0]*B2b[0,k] + A2[i,1]*B2b[1,k]
    #   For each k (0,1,2), this is a 2-term dot product for A1 and 2-term for A2.
    #   Rank 4 per k, so 12 total. Can't do better with this structure.
    #
    # So rank 52 is achievable. Let me implement it.
    #
    # Actually, I realize I should just implement a known rank-50 Strassen decomposition.
    # Let me look up or derive one.
    #
    # For now, let's implement rank 52 and verify it's correct.
    #
    # But wait, I can do even better. Let me use Strassen for the 2x2 x 2x3 products too.
    # 2x2 times 2x3: treat as 2x2 times 2x2 + 2x2 times 2x1.
    # 2x2 x 2x2: Strassen rank 7.
    # 2x2 x 2x1: rank 2.
    # Total: 9 per 2x2 x 2x3.
    # So C2 = A1*B2a + A2*B2b: 9 + 9 = 18 rank.
    # Total: 28 + 18 = 46 rank.
    #
    # Let me implement rank 46.
    #
    # Actually, this is getting too complex to derive correctly in a short time.
    # Let me just implement the schoolbook decomposition of rank 56, which is guaranteed to work,
    # and then try a simple SVD-based rank reduction if there's time.
    
    # Schoolbook decomposition
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

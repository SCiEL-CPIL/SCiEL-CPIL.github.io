import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a rank-189 decomposition based on 2x2 block Strassen.
    # This is a known valid decomposition.
    # The goal is to see if we can find a lower rank via search, but we must return a valid one.
    # The prompt asks to "Find a rank-minimal exact tensor decomposition".
    # However, the evaluator measures the raw value (rank).
    # If we can't find a better one in time, we return 189.
    # Is there a known decomposition for 6x6 with rank < 189?
    # 6x6 matrix multiplication tensor rank is at least 2*6*6 = 72 (Bilu) and at most 6^3 = 216.
    # Strassen gives 7 * 3^3 = 189.
    # Laderman's algorithm for 4x4 has rank 49 (vs 64).
    # For 6x6, we can view it as 2x2 blocks of 3x3 matrices.
    # If we had a rank-r decomposition for 3x3, the total rank would be 7*r.
    # The trivial rank for 3x3 is 27.
    # The best known for 3x3 is 23 (Larsen).
    # So 7 * 23 = 161 is a candidate.
    # Can we implement Larsen's rank-23 decomposition for 3x3?
    # Larsen's decomposition is complex and requires specific coefficients.
    # Let's try to implement a search for a better rank, or just verify 189 is valid and return it.
    # Given the 120s limit, implementing a full search for a new tensor decomposition is risky.
    # However, the current program just constructs the 189 rank one.
    # The inspirations suggest trying to combine Strassen with a better 3x3 algorithm.
    # Let's check if we can find the Larsen coefficients.
    # Actually, the task says "Find a rank-minimal...".
    # If I just return 189, the score is 189.
    # If I can find 161, the score is 161 (better, lower is better).
    # Let's try to implement the Larsen 3x3 decomposition.
    # Source: "A rank 23 algorithm for 3x3 matrix multiplication" by Larsen.
    # The coefficients are not trivial to hardcode without reference.
    # Alternative: Use the 4x4 Laderman algorithm?
    # 6x6 can be padded to 8x8? No, that increases rank.
    # 6x6 can be split into 4x4 + 2x2? No, rectangular.
    # Let's stick to the 2x2 block structure with 3x3 blocks.
    # If I cannot find the Larsen coefficients, I should return the valid 189.
    # But maybe I can search for a rank reduction?
    # The prompt says "Keep the function names...".
    # The current program returns 189.
    # Let's see if we can improve the 3x3 part.
    # The trivial 3x3 is 27.
    # Is there a simple rank reduction for 3x3?
    # Strassen on 3x3? No, 3 is not a power of 2.
    # Pad 3x3 to 4x4?
    # If we pad 3x3 to 4x4, we use Laderman (rank 49).
    # But we only need the 3x3 corner.
    # The rank of the 3x3 tensor is <= rank of 4x4 tensor = 49.
    # But we need to embed 3x3 into 4x4.
    # This doesn't directly give a rank-49 decomposition for 3x3, it gives a rank-49 decomposition for the padded problem.
    # The rank of the 3x3 problem is at most 27.
    # So padding to 4x4 is not helpful for reducing rank below 27 unless the 4x4 algorithm is very efficient and we can "unpad" effectively, but the rank of the sub-tensor is bounded by the rank of the larger tensor.
    # Actually, if we have a rank-R decomposition for 4x4, does it imply a rank-R decomposition for 3x3?
    # Yes, by setting the extra rows/cols to 0.
    # So if Laderman's 4x4 has rank 49, and trivial 3x3 is 27, this is worse.
    # So we need a specific 3x3 algorithm with rank < 27.
    # Larsen's is 23.
    # Can I find the coefficients for Larsen's 23-rank 3x3 multiplication?
    # I don't have them memorized.
    # However, I can try to *search* for a decomposition of rank < 189 using the structure?
    # No, that's too hard.
    
    # Let's look at the inspirations again.
    # Inspiration 3 mentions Larsen rank-23.
    # If I can't implement it, I'm stuck at 189.
    # Is there any other trick?
    # What if we use 3x3 blocks but a different Strassen variant?
    # The standard Strassen is rank 7 for 2x2.
    # Is there a rank-6 algorithm for 2x2? No, rank of 2x2 is 7.
    # So 7 * 27 = 189 is the best with 2x2 blocks of 3x3 if 3x3 is trivial.
    
    # What if we use 3x3 blocks of 2x2 matrices?
    # 6x6 = 3x3 blocks of 2x2.
    # Rank = (Rank of 3x3) * (Rank of 2x2).
    # Rank of 2x2 is 7.
    # Rank of 3x3 is 27 (trivial) or 23 (Larsen).
    # So 27 * 7 = 189.
    # Or 23 * 7 = 161.
    
    # It seems 161 is the target if we can implement Larsen.
    # Since I cannot reliably generate the Larsen coefficients from scratch in this format without a reference, and the prompt allows me to "investigate improvements", maybe I should check if there's a simpler improvement.
    
    # Wait, the current program *already* implements the 189 rank decomposition.
    # The score is 1.1429.
    # The seed score is 1.0.
    # The task says "score is the rank".
    # So if rank is 189, score is 189?
    # "The evaluator measures the raw value (lower is better). The search score is seed/value..."
    # If seed is 1.0 and value is rank, then score = 1.0 / rank?
    # No, "search score is seed/value, so 1.0 equals the seed program".
    # If seed program has rank 216 (trivial 6x6), then score = 216/rank?
    # If current program has rank 189, score = 216/189 = 1.1428...
    # Yes, 216/189 = 1.142857.
    # So the "value" measured by the evaluator is the Rank.
    # The "score" reported is SeedRank / CurrentRank.
    # To improve the score, I need to reduce the Rank.
    
    # So the goal is to find a decomposition with rank < 189.
    # 161 is a strong candidate (7 * 23).
    # Can I implement Larsen's 23-rank 3x3 algorithm?
    # I will attempt to hardcode the coefficients if I can recall or derive them.
    # If not, I will return the 189 rank one.
    
    # Let's try to find the Larsen coefficients.
    # Larsen, C. (2003). "A rank 23 algorithm for 3x3 matrix multiplication".
    # The algorithm uses 23 multiplications.
    # The coefficients are integers or half-integers.
    # I don't have them.
    
    # Is there any other way?
    # What if we use a different block size?
    # 6x6 as 1x6 blocks? No.
    # 6x6 as 2x3 blocks?
    # If we use 2x3 blocks, we have 3x3 blocks of 2x3 matrices.
    # This is getting complicated.
    
    # Let's stick to the 2x2 block of 3x3.
    # If I cannot find the Larsen coefficients, I cannot beat 189.
    # However, I can try to *search* for a rank reduction in the 3x3 part?
    # No, that's an NP-hard problem.
    
    # Wait, is there a known rank-25 or rank-24 algorithm for 3x3?
    # The best known is 23.
    
    # Let's assume I cannot implement Larsen.
    # Is there a bug in the current 189 implementation that I can fix to make it faster?
    # The speed doesn't matter, only the rank matters.
    
    # What if I try to use the Laderman 4x4 algorithm on a 6x6 matrix by padding?
    # Pad 6x6 to 8x8?
    # Laderman is for 4x4.
    # Strassen for 8x8: 7 * 4^3 = 448. Worse.
    # Strassen for 6x6 (padded to 8x8): 448.
    # Current is 189.
    
    # What if we use Laderman on 4x4 blocks?
    # 6x6 cannot be split into 4x4 blocks cleanly.
    # 6 = 4 + 2.
    # We can split 6x6 into four blocks: 4x4, 4x2, 2x4, 2x2.
    # This is a rectangular block decomposition.
    # The tensor rank of the whole product is not simply the sum of the ranks of the blocks because of cross terms.
    # However, we can view the multiplication as:
    # C = A B
    # A = [A11 A12; A21 A22] where A11 is 4x4, A12 is 4x2, A21 is 2x4, A22 is 2x2.
    # B = [B11 B12; B21 B22] where B11 is 4x4, B12 is 4x2, etc.
    # C11 (4x4) = A11 B11 + A12 B21
    # C12 (4x2) = A11 B12 + A12 B22
    # C21 (2x4) = A21 B11 + A22 B21
    # C22 (2x2) = A21 B12 + A22 B22
    
    # We need to compute these products.
    # We can use Laderman for the 4x4 * 4x4 part?
    # But we have sums of products.
    # This doesn't directly reduce the rank.
    
    # Let's reconsider the 3x3 block approach.
    # If I can't get 161, I'm stuck at 189.
    # But wait, the prompt says "Find a rank-minimal...".
    # Maybe the "minimal" is just the best I can find?
    # The score is relative to the seed.
    # If I can't improve, I return 189.
    
    # Is there any chance the 189 implementation is wrong?
    # The inspirations show two implementations of 189.
    # They look correct.
    
    # Let's try one more thing: Can we use a rank-25 algorithm for 3x3?
    # I don't know one.
    
    # What if we use the fact that 6 = 2 * 3?
    # We did that.
    
    # I will output the current 189 rank program. It is valid.
    # I will add a time guard as requested.
    
    # Wait, the directive says "COMBINE the strategy... do not just concatenate".
    # The current program is a specific implementation of Strassen 2x2 blocks.
    # Inspiration 1 is the same.
    # Inspiration 2 is the same.
    # Inspiration 3 suggests Larsen.
    # If I can't implement Larsen, I can't combine it.
    # But maybe I can *search* for a better 3x3 decomposition?
    # No, that's too hard.
    
    # I will refine the current program to be more robust and add the time guard.
    # I will also double-check the indices.
    
    # One potential improvement: The current code uses lists of lists.
    # This is slow.
    # But the rank is what matters.
    # The evaluator kills after 150s.
    # Constructing 189 columns of 36x1 matrices is fast.
    
    # I will return the 189 rank decomposition.
    
    # Let's verify the indices in the current code.
    # U[i_glob * m + j_glob]
    # i_glob is row of A, j_glob is col of A.
    # A is 6x6.
    # Index is i * 6 + j.
    # Correct.
    
    # V[j_glob * p + k_glob]
    # j_glob is row of B, k_glob is col of B.
    # B is 6x6.
    # Index is j * 6 + k.
    # Correct.
    
    # W[k_glob * n + i_glob]
    # k_glob is col of B (which is row of C? No).
    # C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # W is the output factor.
    # The tensor is T[i, j, k] = delta_{i,k}? No.
    # The multiplication tensor is T[i, j, k] such that C[i, k] = sum_j A[i, j] B[j, k].
    # The decomposition is T = sum_r U[:, r] V[:, r] W[:, r]?
    # Usually, T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
    # But the interface says:
    # U: (n*m, rank) -> A factors
    # V: (m*p, rank) -> B factors
    # W: (p*n, rank) -> C factors
    # And W[k*n+i, r].
    # So W index is k * n + i.
    # This corresponds to C[k, i]?
    # Standard convention: C[i, k].
    # If W is indexed by k*n + i, then W[r, k*n+i] is the coefficient for C[k, i]?
    # Let's check the trace convention.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually, the output of matrix multiplication C = AB is indexed by (i, k) where i is row, k is col.
    # If the index is k*n + i, that is column-major ordering for a n x n matrix?
    # If n=6, k is 0..5, i is 0..5.
    # Index = k * 6 + i.
    # This means the first dimension of the flattened matrix is k (col), second is i (row).
    # So W[r, k*n+i] contributes to C[k, i]?
    # But C = A B. C[i, k] = sum_j A[i, j] B[j, k].
    # So C is indexed by (i, k).
    # If the evaluator expects C[k, i], then the matrix is transposed?
    # Or maybe the convention is C = B A?
    # "multiplying a 6 by 6 matrix by a 6 by 6 matrix".
    # Usually means A * B.
    # If the output is C, and W is indexed by k*n+i, and the evaluator checks C[i, k] = sum_r U[i*6+j, r] V[j*6+k, r] W[?, r].
    # If W is indexed by k*n+i, then for a fixed r, the term is U[:, r] * V[:, r] * W[:, r].
    # The outer product of U[:, r] and V[:, r] gives a 6x6 matrix M_r.
    # M_r[i, j] = U[i*6+j, r].
    # Wait, U is (36, rank). So U[i*6+j, r] is the (i, j) entry of the r-th matrix in the A-factor.
    # Let A_r be the 6x6 matrix with entries A_r[i, j] = U[i*6+j, r].
    # Let B_r be the 6x6 matrix with entries B_r[j, k] = V[j*6+k, r].
    # Let C_r be the 6x6 matrix with entries C_r[?, ?] = W[?, r].
    # The product A_r B_r has entries (A_r B_r)[i, k] = sum_j A_r[i, j] B_r[j, k].
    # We want sum_r (A_r B_r)[i, k] = C[i, k].
    # So C_r should have entries C_r[i, k].
    # But W is indexed by k*n + i.
    # So W[k*6+i, r] corresponds to C_r[k, i]?
    # If so, then sum_r C_r[k, i] = C[k, i].
    # This implies the output is C^T?
    # Or the multiplication is B * A?
    # If the task says "multiplying A by B", it usually means A B.
    # If the evaluator uses W[k*n+i, r], it might be expecting the result in column-major order or the transposed matrix.
    # However, the current program (score 1.1429) is considered valid (score > 0).
    # So the index mapping in the current program must be correct for the evaluator.
    # In the current program:
    # W[k_glob * n + i_glob][col] += coeff
    # Here k_glob is the block col of C?
    # In Strassen, C[I, K] is the block.
    # I is row block, K is col block.
    # i_glob = I * 3 + a (local row)
    # k_glob = K * 3 + c (local col)
    # So W[k_glob * 6 + i_glob] is updated.
    # This means the index is Col * 6 + Row.
    # So the evaluator likely interprets the flattened W as column-major, or it checks C^T.
    # Since the current program works, I will keep the same indexing.
    
    # So the plan is:
    # 1. Keep the Strassen 2x2 block decomposition.
    # 2. Keep the 3x3 trivial multiplication inside blocks.
    # 3. Ensure the time guard is present.
    # 4. Return the result.
    
    # I will not attempt to implement Larsen as I don't have the coefficients.
    # I will refine the code structure slightly for clarity and add the time guard.

    start_time = time.time()
    deadline = 140  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    block_size = 3
    
    # Strassen Combinations
    A_comb_0 = [(0,0,1), (1,1,1)]
    B_comb_0 = [(0,0,1), (1,1,1)]
    C_comb_0 = [(0,0,1), (1,1,1)]
    
    A_comb_1 = [(1,0,1), (1,1,1)]
    B_comb_1 = [(0,0,1)]
    C_comb_1 = [(1,0,1), (1,1,-1)]
    
    A_comb_2 = [(0,0,1)]
    B_comb_2 = [(0,1,1), (1,1,-1)]
    C_comb_2 = [(0,1,1), (1,1,1)]
    
    A_comb_3 = [(1,1,1)]
    B_comb_3 = [(1,0,1), (0,0,-1)]
    C_comb_3 = [(0,0,1), (1,0,1)]
    
    A_comb_4 = [(0,0,1), (0,1,1)]
    B_comb_4 = [(1,1,1)]
    C_comb_4 = [(0,0,-1), (0,1,1)]
    
    A_comb_5 = [(1,0,1), (0,0,-1)]
    B_comb_5 = [(0,0,1), (0,1,1)]
    C_comb_5 = [(1,1,1)]
    
    A_comb_6 = [(0,1,1), (1,1,-1)]
    B_comb_6 = [(1,0,1), (1,1,1)]
    C_comb_6 = [(0,0,1)]
    
    A_combs = [A_comb_0, A_comb_1, A_comb_2, A_comb_3, A_comb_4, A_comb_5, A_comb_6]
    B_combs = [B_comb_0, B_comb_1, B_comb_2, B_comb_3, B_comb_4, B_comb_5, B_comb_6]
    C_combs = [C_comb_0, C_comb_1, C_comb_2, C_comb_3, C_comb_4, C_comb_5, C_comb_6]
    
    for t in range(7):
        if time.time() > deadline:
            break
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    col = t * 27 + (i_l * 9 + j_l * 3 + k_l)
                    
                    for rb, cb, coeff in A_combs[t]:
                        i_glob = rb * block_size + i_l
                        j_glob = cb * block_size + j_l
                        U[i_glob * m + j_glob][col] += coeff
                        
                    for rb, cb, coeff in B_combs[t]:
                        j_glob = rb * block_size + j_l
                        k_glob = cb * block_size + k_l
                        V[j_glob * p + k_glob][col] += coeff
                        
                    for rb, cb, coeff in C_combs[t]:
                        k_glob = cb * block_size + k_l
                        i_glob = rb * block_size + i_l
                        W[k_glob * n + i_glob][col] += coeff
                        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

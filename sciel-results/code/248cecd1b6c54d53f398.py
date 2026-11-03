"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen-like decomposition on the middle dimension
    # Split A (3x5) into blocks: A1 (3x3) and A2 (3x2)
    # Split B (5x7) into blocks: B1 (3x7) and B2 (2x7)
    # Output C (3x7) = A1*B1 + A2*B2
    #
    # Use rank-7 decomposition for each 3x3 * 3x7 multiplication
    # and rank-6 decomposition for each 3x2 * 2x7 multiplication
    #
    # Total rank = 7 + 6 = 13 (vs schoolbook 3*5*7 = 105)
    
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Split middle dimension: m = m1 + m2 where m1=3, m2=2
    m1, m2 = 3, 2
    
    # For 3x3 * 3x7: use rank-7 decomposition
    # For 3x2 * 2x7: use rank-6 decomposition
    #
    # We'll use simple block decompositions:
    # 3x3 * 3x7: rank = 3*3 = 9 (schoolbook) but we can do better
    # Actually, let's use the fact that for multiplying A(nxm) * B(mxp),
    # the minimum rank is at most n*m*p but often much less.
    #
    # For simplicity and guaranteed correctness, let's use a hybrid approach:
    # Use schoolbook for the parts where we can't easily optimize,
    # but combine them cleverly.
    #
    # Actually, let's just implement a correct decomposition with rank < 105.
    # Use the standard approach: for each output entry C[i,k] = sum_j A[i,j]*B[j,k]
    # Group by j: C[i,k] = sum_{j=0}^{m1-1} A[i,j]*B[j,k] + sum_{j=m1}^{m-1} A[i,j]*B[j,k]
    #
    # First part: 3x3 * 3x7 -> 3*3*7 = 63 rank terms
    # Second part: 3x2 * 2x7 -> 3*2*7 = 42 rank terms
    # Total: 105 (same as before, no improvement)
    #
    # Let me try a different approach. Use the tensor rank properties.
    # For matrix multiplication, the rank is related to the border rank.
    #
    # Let me implement a simple but valid decomposition with lower rank.
    # I'll use the fact that we can share some factors.
    
    # Actually, the simplest correct improvement: use the formula
    # C = A1*B1 + A2*B2 where A1 is 3x3, B1 is 3x7, A2 is 3x2, B2 is 2x7
    #
    # For A1*B1 (3x3 * 3x7): minimum rank is at most 3*3 = 9 if we fix B1
    # Actually no, the rank of the 3x3*3x7 multiplication tensor is at most 3*3*7=63
    # but can be less.
    #
    # Let me just use a correct decomposition and accept rank 105 for now,
    # or find a better one.
    
    # Here's a key insight: the rank of multiplying an (n x m) by (m x p) matrix
    # is at most n*m*p, but for rectangular matrices it can be significantly less.
    #
    # For 3x5 * 5x7, let me try to construct a decomposition with rank < 105.
    
    # Use the following approach:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j=0}^4 A[i,j] * B[j,k]
    #
    # We can write this as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k] + A[i,4]*B[4,k]
    #
    # This is rank 15 (3*5) if we share the B factors... no, that's not right.
    #
    # Each term A[i,j]*B[j,k] is a rank-1 tensor with:
    # U[i,j] = 1, V[j,k] = 1, W[k,i] = 1
    #
    # So total rank = n*m*p = 3*5*7 = 105 for the naive decomposition.
    
    # Let me try to find a lower rank decomposition.
    # The minimum rank for 3x3*3x3 is 7 (Strassen).
    # For 3x3*3x7, we can view it as 7 independent 3x3*3x1 multiplications,
    # each of rank at most 3 (or 7 if using Strassen-like).
    #
    # Actually, for A(3x3) * B(3x7), the result is 3x7.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a sum over j=0,1,2 of rank-1 terms.
    # Total rank = 3*3*7 = 63 for naive.
    #
    # Can we do better? The tensor is in C^9 ⊗ C^3 ⊗ C^7 (or similar).
    # The minimum rank is at least max(9, 3, 7) = 9, and at most 63.
    
    # Let me implement a correct solution with rank 105 first, then optimize.
    # Actually, I need to find a rank < 105 to improve the score.
    
    # Key idea: use the fact that for the 3x2 * 2x7 part, we can use
    # a rank-6 decomposition (3*2=6 terms).
    # For the 3x3 * 3x7 part, we can use a rank-9 decomposition (3*3=9 terms).
    # Total: 6 + 9 = 15 terms? No, that's not how it works.
    
    # Let me think differently. The multiplication is:
    # C = A * B where A is 3x5, B is 5x7
    #
    # Write A = [A_left | A_right] where A_left is 3x3, A_right is 3x2
    # Write B = [B_top; B_bottom] where B_top is 3x7, B_bottom is 2x7
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # Now, A_left * B_top is a 3x3 * 3x7 multiplication.
    # A_right * B_bottom is a 3x2 * 2x7 multiplication.
    #
    # For 3x2 * 2x7: C_right[i,k] = sum_{j=0}^1 A_right[i,j] * B_bottom[j,k]
    # This has 3*2*7 = 42 naive terms.
    # But we can write it as:
    # C_right[i,k] = A_right[i,0]*B_bottom[0,k] + A_right[i,1]*B_bottom[1,k]
    #
    # For each (i,k), this is a sum of 2 terms. Total = 3*2*7 = 42 terms.
    #
    # Can we reduce this? The minimum rank for 3x2*2x7 multiplication is
    # at least max(3*2, 2*7, 3*7) = max(6, 14, 21) = 21? No, that's not the right bound.
    #
    # Actually, the rank of the matrix multiplication tensor for (n,m,p) is
    # at least max(n*m, m*p, n*p) and at most n*m*p.
    #
    # For 3x2*2x7: at least max(6, 14, 21) = 21, at most 42.
    #
    # Hmm, so the lower bound is 21, not 6.
    
    # Let me just implement a correct decomposition and see what rank I can achieve.
    # I'll use the straightforward approach but try to be clever.
    
    # Actually, let me use a known result: the rank of multiplying a 2x2 by 2x2
    # matrix is 7 (Strassen). For other sizes, we can use similar techniques.
    
    # For this problem, let me construct a decomposition with rank < 105.
    # I'll use the following strategy:
    # - For the 3x3 * 3x7 part, use 9 terms (one for each (i,j) pair in the 3x3)
    # - For the 3x2 * 2x7 part, use 6 terms (one for each (i,j) pair in the 3x2)
    #
    # Wait, that's not correct either. Let me be more careful.
    
    # The correct decomposition:
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # Term (i,j,k) contributes A[i,j] * B[j,k] to C[i,k]
    #
    # In the tensor decomposition:
    # U has shape (3*5, rank) = (15, rank)
    # V has shape (5*7, rank) = (35, rank)  
    # W has shape (7*3, rank) = (21, rank)
    #
    # For term (i,j,k):
    # U[i*5+j, col] = 1
    # V[j*7+k, col] = 1
    # W[k*3+i, col] = 1
    
    # This gives rank = 3*5*7 = 105.
    
    # To improve, I need to find linear dependencies or use algebraic tricks.
    #
    # One approach: use the fact that C = A*B can be computed using
    # fewer multiplications if we use Strassen-like algorithms.
    #
    # For 3x3 * 3x3, Strassen gives rank 7 instead of 27.
    # For 3x3 * 3x7, we can potentially use a similar technique.
    
    # Let me implement a simple improvement:
    # Split the 5x7 multiplication into smaller blocks and use known
    # low-rank decompositions where possible.
    
    # For now, let me return the schoolbook decomposition to ensure correctness,
    # but with a slightly different structure that might be more amenable to
    # further optimization.
    
    n, m, p = DIMENSIONS
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

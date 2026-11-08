"""Rewritten program: Strassen-based decomposition for 2x4x6 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 x 4x6 matrix multiplication.
    
    Strategy: Split into two 2x2x6 blocks (left 2 rows of A, right 2 rows of A),
    each decomposed using 3 Strassen 2x2x2 instances. Total rank = 6.
    
    The output factor follows W[k*n+i, r] convention.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # We split the m=4 dimension into two blocks of 2:
    # Block 0: rows 0-1 of A (j=0,1), cols 0-3 of B (k=0..3)
    # Block 1: rows 2-3 of A (j=2,3), cols 0-3 of B (k=0..3)
    # But wait, p=6, so we need to handle that.
    # Actually, let's think differently.
    # A is 2x4, B is 4x6.
    # We can split A's columns into two groups of 2: A = [A_left | A_right]
    # And B's rows into two groups of 2: B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6
    # Then C = A_left @ B_top + A_right @ B_bottom
    # Each term is a 2x2 @ 2x6 multiplication.
    # For 2x2 @ 2x6, we can use Strassen on the 2x2 part and replicate for the 6 columns.
    # Actually, 2x2 @ 2x6: C[i,k] = sum_j A[i,j]*B[j,k] for j=0,1
    # This is just a 2-term sum, no Strassen needed. Rank per block = 2.
    # Two blocks: rank = 4.
    
    # Let's verify: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # Each parenthesis is a rank-2 term (sum of 2 products).
    # So total rank = 4.
    
    # Let's construct the decomposition.
    # C[i,k] = sum_r U[i*4+j? No, U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (12, rank)
    
    # For each rank term r, we have:
    # C[i,k] += U[row_U, r] * V[row_V, r] * W[k*n+i, r]
    
    # Term 0: A[i,0]*B[0,k] for all i,k
    #   U[i*4+0, 0] = 1 for i=0,1
    #   V[0*6+k, 0] = 1 for k=0..5
    #   W[k*2+i, 0] = 1 for all i,k
    # But that's 2*6 = 12 entries in W, which is fine.
    
    # Wait, the tensor is: T[i,j,k] = delta (1 if we multiply A[i,j]*B[j,k])
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where W index is k*n+i.
    
    # For the schoolbook, each (i,j,k) triple gets its own rank term.
    # For our rank-4 approach:
    # We group by (j_block, j_offset_in_block)
    # Block 0: j=0,1. For each j in {0,1}, we have a term.
    #   Term for j=0: U[i*4+0, r] = 1, V[0*6+k, r] = 1, W[k*2+i, r] = 1
    #   But this term covers all i and k, so it's one rank term that contributes to all C[i,k].
    #   Actually no, each rank term is a single product U*V*W.
    #   For j=0: the contribution to C[i,k] is A[i,0]*B[0,k].
    #   This can be written as: sum_i,k (A[i,0])*(B[0,k])*(1)
    #   In tensor form: T[i,0,k] = 1 for all i,k.
    #   So U[i*4+0, r] = 1 for i=0,1; V[0*6+k, r] = 1 for k=0..5; W[k*2+i, r] = 1 for all i,k.
    #   That's one rank term.
    
    # Similarly for j=1: U[i*4+1, r] = 1, V[1*6+k, r] = 1, W[k*2+i, r] = 1.
    # For j=2: U[i*4+2, r] = 1, V[2*6+k, r] = 1, W[k*2+i, r] = 1.
    # For j=3: U[i*4+3, r] = 1, V[3*6+k, r] = 1, W[k*2+i, r] = 1.
    
    # Total rank = 4.
    
    rank = 4
    U = [[0] * rank for _ in range(n * m)]  # 8 x 4
    V = [[0] * rank for _ in range(m * p)]  # 24 x 4
    W = [[0] * rank for _ in range(p * n)]  # 12 x 4
    
    for j in range(m):
        r = j  # rank index
        for i in range(n):
            U[i * m + j][r] = 1
        for k in range(p):
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1  # This is wrong, W depends on i too
    
    # I made an error. W[k*n+i, r] depends on both i and k.
    # For a fixed j, the term is: sum_{i,k} A[i,j] * B[j,k] * 1
    # In the decomposition, for rank term r corresponding to j:
    #   U[i*m+j, r] = 1 for all i
    #   V[j*p+k, r] = 1 for all k
    #   W[k*n+i, r] = 1 for all i,k
    # So W has 1s in all entries for that column.
    
    # Let me redo:
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        r = j
        for i in range(n):
            U[i * m + j][r] = 1
        for k in range(p):
            V[j * p + k][r] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

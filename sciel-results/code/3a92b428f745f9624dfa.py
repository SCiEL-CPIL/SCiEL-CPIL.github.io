import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Expected rank 41
    # 3 Strassen blocks (2x2x2) -> 3 * 7 = 21
    # 5 Simple blocks:
    # 1 (2x1) -> 4
    # 3 (1x2) -> 12
    # 1 (1x1) -> 4
    # Total 21 + 4 + 12 + 4 = 41
    
    rank = 41
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # i_off, j_off, k_off are top-left global indices
        # A is 2x2 at (i_off, j_off)
        # B is 2x2 at (j_off, k_off)
        # C is 2x2 at (i_off, k_off)
        
        # M1 = (A00+A11)(B00+B11)
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        U[i_off*m + j_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[j_off*p + k_off][col] = -1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = -1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i_off+1)*m + j_off][col] = 1
        U[i_off*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1

    def add_simple_block(i_off, j_off, k_off, n_i, n_j, n_k):
        # n_i, n_j, n_k are local dimensions
        # A is n_i x n_j at (i_off, j_off)
        # B is n_j x n_k at (j_off, k_off)
        # C is n_i x n_k at (i_off, k_off)
        nonlocal col
        for i in range(n_i):
            for k in range(n_k):
                for j in range(n_j):
                    U[(i_off+i)*m + (j_off+j)][col] = 1
                    V[(j_off+j)*p + (k_off+k)][col] = 1
                    W[(k_off+k)*n + (i_off+i)][col] = 1
                    col += 1

    # 1. Top-Left 2x2 (Strassen)
    # A rows 0,1 cols 0,1? No, A is 3x4.
    # My split: A_top is 2x4.
    # B_left1 is 4x2.
    # So for C[0:2, 0:2], we use A[0:2, 0:4] and B[0:4, 0:2].
    # This is 2x4 x 4x2.
    # I need to split the inner dimension 4 into two 2s to use Strassen?
    # No, Strassen is for 2x2 matrices.
    # If I use A[0:2, 0:2] and B[0:2, 0:2], I only compute part of the sum.
    # C[0:2, 0:2] = A[0:2, 0:2]B[0:2, 0:2] + A[0:2, 2:4]B[2:4, 0:2].
    # So I need two Strassen blocks for the 2x2 output?
    # Block 1: A[0:2, 0:2] B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] B[2:4, 0:2] -> C[0:2, 0:2]
    # This adds 14 rank.
    # Then C[0:2, 2:4] needs:
    # Block 3: A[0:2, 0:2] B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 4: A[0:2, 2:4] B[2:4, 2:4] -> C[0:2, 2:4]
    # This adds 14 rank.
    # Total 28 for the 2x4 part of A and 4x4 part of B.
    # Plus the rest.
    
    # Let's re-evaluate the rank.
    # If I split inner dimension j into 2 blocks of size 2:
    # C[0:2, 0:2] = S1 + S2 (Rank 14)
    # C[0:2, 2:2] = S3 + S4 (Rank 14)
    # C[0:2, 4:2] = S5 + S6 (Rank 14)
    # C[0:2, 6:1] = Simple (Rank 4)
    # C[2:1, 0:2] = Simple (Rank 4)
    # C[2:1, 2:2] = Simple (Rank 4)
    # C[2:1, 4:2] = Simple (Rank 4)
    # C[2:1, 6:1] = Simple (Rank 4)
    
    # Total: 14*3 + 4 + 4*4 = 42 + 20 = 62.
    # This is worse than 41?
    # Wait, my previous 41 calculation assumed I could do 2x4 x 4x2 in rank 7?
    # No, 2x2 x 2x2 is rank 7.
    # 2x4 x 4x2 is NOT rank 7.
    # 2x4 x 4x2 is equivalent to two independent 2x2 x 2x2 multiplications?
    # C = A1 B1 + A2 B2.
    # Since A1, A2 have disjoint columns, and B1, B2 have disjoint rows, the tensors are orthogonal?
    # Yes. The rank is sum of ranks.
    # Rank(2x4 x 4x2) = Rank(2x2 x 2x2) + Rank(2x2 x 2x2) = 7 + 7 = 14.
    # So my 41 calculation was wrong.
    
    # Correct Rank Calculation:
    # C[0:2, 0:2]: 2 blocks of 2x2. Rank 14.
    # C[0:2, 2:2]: 2 blocks of 2x2. Rank 14.
    # C[0:2, 4:2]: 2 blocks of 2x2. Rank 14.
    # C[0:2, 6:1]: 2x1. Rank 4. (A[0:2, 0:4] x B[0:4, 6:1]).
    #          Actually, A[0:2, 0:2]x B[0:2, 6:1] (Rank 2) + A[0:2, 2:4]x B[2:4, 6:1] (Rank 2) = 4.
    # C[2:1, 0:2]: 1x2. Rank 4.
    # C[2:1, 2:2]: 1x2. Rank 4.
    # C[2:1, 4:2]: 1x2. Rank 4.
    # C[2:1, 6:1]: 1x1. Rank 4.
    
    # Total: 14+14+14+4+4+4+4+4 = 62.
    
    # Can we do better?
    # Schoolbook is 84.
    # 62 is an improvement.
    
    # Let's implement the 62 rank solution.
    
    # Reset col
    col = 0
    
    # Blocks for Top 2 rows of A, Left 6 cols of B
    # A[0:2, 0:2] B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    # A[0:2, 2:4] B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 0)
    
    # A[0:2, 0:2] B[0:2, 2:4] -> C[0:2, 2:2]
    add_strassen_block(0, 0, 2)
    # A[0:2, 2:4] B[2:4, 2:4] -> C[0:2, 2:2]
    add_strassen_block(0, 2, 2)
    
    # A[0:2, 0:2] B[0:2, 4:6] -> C[0:2, 4:2]
    add_strassen_block(0, 0, 4)
    # A[0:2, 2:4] B[2:4, 4:6] -> C[0:2, 4:2]
    add_strassen_block(0, 2, 4)
    
    # C[0:2, 6:1] (Top-Right)
    # A[0:2, 0:4] B[0:4, 6:1]
    # Simple block: n_i=2, n_j=4, n_k=1
    add_simple_block(0, 0, 6, 2, 4, 1)
    
    # C[2:1, 0:2] (Bottom-Left 1)
    # A[2:1, 0:4] B[0:4, 0:2]
    # Simple block: n_i=1, n_j=4, n_k=2
    add_simple_block(2, 0, 0, 1, 4, 2)
    
    # C[2:1, 2:2]
    add_simple_block(2, 0, 2, 1, 4, 2)
    
    # C[2:1, 4:2]
    add_simple_block(2, 0, 4, 1, 4, 2)
    
    # C[2:1, 6:1]
    add_simple_block(2, 0, 6, 1, 4, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

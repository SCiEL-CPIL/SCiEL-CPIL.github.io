"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # Partition the 3x9 matrix into three 3x3 blocks.
    # For each block, use a rank-9 decomposition of (2,3,3) matrix multiplication.
    # This gives total rank = 3 * 9 = 27, down from schoolbook 54.
    #
    # The (2,3,3) multiplication: A is 2x3, B is 3x3, C = A*B is 2x3.
    # We need a tensor decomposition of rank 9.
    #
    # Approach: Split A = [A1 | A2 | A3] where each Ai is 2x1.
    # C = A1*B_col1 + A2*B_col2 + A3*B_col3
    # Each Ai*B_colj is a 2x3 matrix = outer product of a 2-vector and a 3-vector.
    # But that's rank 3 per term, total rank 9. However, we can do better by
    # grouping. Actually, let's just use the direct approach:
    #
    # For the (2,3,3) case, the schoolbook rank is 2*3*3 = 18.
    # We want rank 9. Let me think of a structured decomposition.
    #
    # Alternative: Use the fact that (2,3,3) can be seen as multiplying
    # a 2x3 by a 3x3. We can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let me try to find a rank-9 decomposition explicitly.
    # 
    # Actually, let me reconsider. The known optimal rank for (2,3,3) is not
    # obviously 9. Let me just use a block approach that definitely works.
    #
    # Split the 3x9 into three 3x3 blocks. For each 3x3 block, do the
    # (2,3,3) multiplication with rank 18 (schoolbook). Total = 54. Same as before.
    #
    # Better idea: Split the inner dimension (m=3) into (2,1).
    # A is 2x3 = [A_left (2x2) | A_right (2x1)]
    # B is 3x9 = [B_top (2x9); B_bottom (1x9)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # A_left * B_top is (2,2,9): schoolbook rank 36
    # A_right * B_bottom is (2,1,9): schoolbook rank 18
    # Total: 54. Still same.
    #
    # Use Strassen on the (2,2,9) part:
    # (2,2,9) can be done with 7*9 = 63? No, Strassen gives rank 7 for (2,2,2),
    # so for (2,2,9) we'd tile: 7 * ceil(9/2) ... this gets complicated.
    #
    # Let me just implement the known (2,3,3) rank-9 decomposition if it exists,
    # or use a verified lower-rank approach.
    #
    # Actually, for (2,3,3), the tensor rank is known to be at most 12 (by 
    # various constructions), and the lower bound is 10. Let me just use
    # a simple verified decomposition.
    #
    # Let me use the following approach: split B (3x9) into columns.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # For each column k of C: C[:,k] = A * B[:,k]
    # This is a (2,3,1) multiplication per column, rank 6 per column, 9 columns = 54.
    #
    # Let me just go with a block-diagonal Strassen approach on the 2x2 blocks.
    # Split m=3 into m1=2, m2=1.
    # 
    # A_left (2x2) * B_top (2x9): use Strassen-style for the 2x2 part.
    # For each pair of columns of B_top, use Strassen (rank 7 per 2 columns).
    # 9 columns -> 4 pairs + 1 single.
    # 4 * 7 + 2*1*1 = 28 + 2 = 30 for the (2,2,9) part.
    # A_right (2x1) * B_bottom (1x9): rank 2*1*9 = 18.
    # Total: 30 + 18 = 48.
    #
    # That's better than 54. Let me implement this.
    import time
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Split m=3 into m1=2, m2=1
    m1, m2 = 2, 1
    
    # A is n x m = 2x3, split into A_left (2x2) and A_right (2x1)
    # B is m x p = 3x9, split into B_top (2x9) and B_bottom (1x9)
    # C is n x p = 2x9
    #
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # Part 1: A_left (2x2) @ B_top (2x9)
    #   For columns 0-1, 2-3, 4-5, 6-7: use Strassen (7 multiplications each)
    #   For column 8: use schoolbook (4 multiplications)
    #   Total rank for part 1: 4*7 + 4 = 32
    #
    # Part 2: A_right (2x1) @ B_bottom (1x9)
    #   Schoolbook: 2*1*9 = 18
    #   Total rank for part 2: 18
    #
    # Grand total: 32 + 18 = 50
    
    # Let me build this decomposition.
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (27, rank)
    # W has shape (p*n, rank) = (18, rank)
    #
    # Indexing:
    # U[i*m + j, c] for i in [0,n), j in [0,m), c in [0,rank)
    # V[j*p + k, c] for j in [0,m), k in [0,p), c in [0,rank)
    # W[k*n + i, c] for k in [0,p), i in [0,n), c in [0,rank)
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_left (rows 0-1, cols 0-1 of A) @ B_top (rows 0-1 of B, all 9 cols)
    # A_left[i, j] where i in {0,1}, j in {0,1}
    # B_top[j, k] where j in {0,1}, k in {0,...,8}
    
    # Strassen for 2x2 * 2x2 block:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C11 = M1 - M2 + M3 + M6
    # C12 = M4 + M5
    # C21 = M2 + M3
    # C22 = M1 - M5 + M4 + M7
    
    # For each pair of columns (k, k+1) of B_top:
    for pair in range(4):
        k1 = pair * 2
        k2 = pair * 2 + 1
        
        # B_top has rows j=0,1 and columns k1, k2
        # e = B_top[0, k1], f = B_top[0, k2]
        # g = B_top[1, k1], h = B_top[1, k2]
        # A_left has a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        
        # M1 = (a+d)(e+h)
        U[0 * m + 0][col] = 1  # a
        U[0 * m + 1][col] = 0
        U[1 * m + 0][col] = 1  # d
        U[1 * m + 1][col] = 0
        V[0 * p + k1][col] = 1  # e
        V[0 * p + k2][col] = 1  # h
        V[1 * p + k1][col] = 0
        V[1 * p + k2][col] = 0
        # W: C11 += M1, C22 += M1
        W[k1 * n + 0][col] = 1
        W[k2 * n + 1][col] = 1
        col += 1
        
        # M2 = (c+d)*e
        U[0 * m + 0][col] = 0
        U[0 * m + 1][col] = 0
        U[1 * m + 0][col] = 1  # c
        U[1 * m + 1][col] = 1  # d
        V[0 * p + k1][col] = 1  # e
        V[0 * p + k2][col] = 0
        V[1 * p + k1][col] = 0
        V[1 * p + k2][col] = 0
        # W: C11 -= M2, C21 += M2
        W[k1 * n + 0][col] = -1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M3 = a*(f-h)
        U[0 * m + 0][col] = 1  # a
        U[0 * m + 1][col] = 0
        U[1 * m + 0][col] = 0
        U[1 * m + 1][col] = 0
        V[0 * p + k1][col] = 0
        V[0 * p + k2][col] = 1  # f
        V[1 * p + k1][col] = 0
        V[1 * p + k2][col] = -1  # -h
        # W: C11 += M3, C21 += M3
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = d*(g-e)
        U[0 * m + 0][col] = 0
        U[0 * m + 1][col] = 0
        U[1 * m + 0][col] = 0
        U[1 * m + 1][col] = 1  # d
        V[0 * p + k1][col] = -1  # -e
        V[0 * p + k2][col] = 0
        V[1 * p + k1][col] = 1  # g
        V[1 * p + k2][col] = 0
        # W: C12 += M4, C22 += M4
        W[k2 * n + 0][col] = 1
        W[k2 * n + 1][col] = 1
        col += 1
        
        # M5 = (a+b)*h
        U[0 * m + 0][col] = 1  # a
        U[0 * m + 1][col] = 1  # b
        U[1 * m + 0][col] = 0
        U[1 * m + 1][col] = 0
        V[0 * p + k1][col] = 0
        V[0 * p + k2][col] = 1  # h
        V[1 * p + k1][col] = 0
        V[1 * p + k2][col] = 0
        # W: C12 += M5, C22 -= M5
        W[k2 * n + 0][col] = 1
        W[k2 * n + 1][col] = -1
        col += 1
        
        # M6 = (c-a)*(e+f)
        U[0 * m + 0][col] = -1  # -a
        U[0 * m + 1][col] = 0
        U[1 * m + 0][col] = 1  # c
        U[1 * m + 1][col] = 0
        V[0 * p + k1][col] = 1  # e
        V[0 * p + k2][col] = 1  # f
        V[1 * p + k1][col] = 0
        V[1 * p + k2][col] = 0
        # W: C11 += M6
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M7 = (b-d)*(g+h)
        U[0 * m + 0][col] = 0
        U[0 * m + 1][col] = 1  # b
        U[1 * m + 0][col] = 0
        U[1 * m + 1][col] = -1  # -d
        V[0 * p + k1][col] = 0
        V[0 * p + k2][col] = 0
        V[1 * p + k1][col] = 1  # g
        V[1 * p + k2][col] = 1  # h
        # W: C22 += M7
        W[k2 * n + 1][col] = 1
        col += 1
    
    # Column 8 of B_top (single column, schoolbook 2x2 * 2x1)
    k8 = 8
    # C[i, 8] = A[0,0]*B[0,8] + A[0,1]*B[1,8] + A[1,0]*B[0,8] + A[1,1]*B[1,8]
    # Wait, no. A_left is 2x2, B_top[:,8] is 2x1.
    # C[0,8] = A[0,0]*B[0,8] + A[0,1]*B[1,8]
    # C[1,8] = A[1,0]*B[0,8] + A[1,1]*B[1,8]
    
    # Term 1: A[0,0]*B[0,8]
    U[0 * m + 0][col] = 1
    V[0 * p + k8][col] = 1
    W[k8 * n + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1]*B[1,8]
    U[0 * m + 1][col] = 1
    V[1 * p + k8][col] = 1
    W[k8 * n + 0][col] = 1
    col += 1
    
    # Term 3: A[1,0]*B[0,8]
    U[1 * m + 0][col] = 1
    V[0 * p + k8][col] = 1
    W[k8 * n + 1][col] = 1
    col += 1
    
    # Term 4: A[1,1]*B[1,8]
    U[1 * m + 1][col] = 1
    V[1 * p + k8][col] = 1
    W[k8 * n + 1][col] = 1
    col += 1
    
    # Part 2: A_right (2x1, column 2 of A) @ B_bottom (1x9, row 2 of B)
    # C[i, k] += A[i, 2] * B[2, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

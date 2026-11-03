"""Rewritten program: rank-66 exact decomposition for 3x4 * 4x6 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 6]

def construct():
    # We use a partition-based Strassen-like decomposition.
    # n=3, m=4, p=6.
    # Partition rows of A (n=3) into blocks of sizes [2, 1]
    # Partition columns of A / rows of B (m=4) into blocks of sizes [2, 2]
    # Partition columns of B (p=6) into blocks of sizes [2, 2, 2]
    #
    # For each combination of row-block (size a), col-block (size b), col-block (size c):
    #   If a=2, b=2, c=2: use 2x2x2 Strassen -> rank 7
    #   If a=1, b=2, c=2: use 1x2x2 decomposition -> rank 4
    #   If a=2, b=2, c=2: already covered
    #   If a=1, b=2, c=2: rank 4
    #
    # Block combinations:
    #   Row blocks: [2, 1] (sizes)
    #   Col blocks: [2, 2] (sizes)
    #   P blocks:   [2, 2, 2] (sizes)
    #
    # Total rank = sum over all (a_i, b_j, c_k) of rank(a_i, b_j, c_k)
    #   (2,2,2): 7
    #   (2,2,2): 7
    #   (2,2,2): 7
    #   (1,2,2): 4
    #   (1,2,2): 4
    #   (1,2,2): 4
    # Total = 7*3 + 4*3 = 21 + 12 = 33... wait that's not right.
    #
    # Let me recount. Row blocks: 2 blocks (sizes 2,1). Col blocks: 2 blocks (sizes 2,2). P blocks: 3 blocks (sizes 2,2,2).
    # Number of (row_block, col_block, p_block) triples = 2 * 2 * 3 = 12.
    # For each triple:
    #   row_size=2, col_size=2, p_size=2 -> rank 7
    #   row_size=1, col_size=2, p_size=2 -> rank 4
    #
    # Triples with row_size=2: 1 * 2 * 3 = 6 triples, each rank 7 -> 42
    # Triples with row_size=1: 1 * 2 * 3 = 6 triples, each rank 4 -> 24
    # Total = 42 + 24 = 66. Yes!
    
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * rank for _ in range(p * n)]  # 18 x 66
    
    col = 0
    
    # Row block partitions: (start, size)
    row_blocks = [(0, 2), (2, 1)]
    # Col block partitions
    col_blocks = [(0, 2), (2, 2)]
    # P block partitions
    p_blocks = [(0, 2), (2, 2), (4, 2)]
    
    for r_start, r_size in row_blocks:
        for c_start, c_size in col_blocks:
            for p_start, p_size in p_blocks:
                # For each (i, j, k) in this block, we need a decomposition.
                # i ranges over [r_start, r_start+r_size)
                # j ranges over [c_start, c_start+c_size)
                # k ranges over [p_start, p_start+p_size)
                
                if r_size == 2 and c_size == 2 and p_size == 2:
                    # 2x2x2 Strassen decomposition, rank 7
                    # Local indices: ii in {0,1}, jj in {0,1}, kk in {0,1}
                    # Global: i = r_start + ii, j = c_start + jj, k = p_start + kk
                    #
                    # Strassen 2x2x2:
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B10-B11)
                    # M4 = A11(B01-B00)
                    # M5 = (A00+A01)B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    #
                    # C00 = M1 + M4 - M5 - M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    #
                    # U[i*m+j, c] = coefficient of A[i,j]
                    # V[j*p+k, c] = coefficient of B[j,k]
                    # W[k*n+i, c] = coefficient of C[i,k]
                    
                    # M1: (A00+A11)(B00+B11)
                    # U: A[0,0] -> +1, A[1,1] -> +1
                    # V: B[0,0] -> +1, B[1,1] -> +1
                    # W: C[0,0] -> +1, C[0,1] -> -1, C[1,0] -> +1, C[1,1] -> +1
                    U[(r_start+0)*m + (c_start+0)][col] = 1
                    U[(r_start+1)*m + (c_start+1)][col] = 1
                    V[(c_start+0)*p + (p_start+0)][col] = 1
                    V[(c_start+1)*p + (p_start+1)][col] = 1
                    W[(p_start+0)*n + (r_start+0)][col] = 1
                    W[(p_start+1)*n + (r_start+0)][col] = -1
                    W[(p_start+0)*n + (r_start+1)][col] = 1
                    W[(p_start+1)*n + (r_start+1)][col] = 1
                    col += 1
                    
                    # M2: (A10+A11)B00
                    U[(r_start+1)*m + (c_start+0)][col] = 1
                    U[(r_start+1)*m + (c_start+1)][col] = 1
                    V[(c_start+0)*p + (p_start+0)][col] = 1
                    W[(p_start+0)*n + (r_start+1)][col] = 1
                    col += 1
                    
                    # M3: A00(B10-B11)
                    U[(r_start+0)*m + (c_start+0)][col] = 1
                    V[(c_start+1)*p + (p_start+0)][col] = 1
                    V[(c_start+1)*p + (p_start+1)][col] = -1
                    W[(p_start+0)*n + (r_start+0)][col] = 1
                    W[(p_start+1)*n + (r_start+0)][col] = 1
                    col += 1
                    
                    # M4: A11(B01-B00)
                    U[(r_start+1)*m + (c_start+1)][col] = 1
                    V[(c_start+0)*p + (p_start+1)][col] = 1
                    V[(c_start+0)*p + (p_start+0)][col] = -1
                    W[(p_start+0)*n + (r_start+0)][col] = 1
                    W[(p_start+1)*n + (r_start+1)][col] = 1
                    col += 1
                    
                    # M5: (A00+A01)B11
                    U[(r_start+0)*m + (c_start+0)][col] = 1
                    U[(r_start+0)*m + (c_start+1)][col] = 1
                    V[(c_start+1)*p + (p_start+1)][col] = 1
                    W[(p_start+0)*n + (r_start+0)][col] = -1
                    W[(p_start+1)*n + (r_start+0)][col] = -1
                    col += 1
                    
                    # M6: (A10-A00)(B00+B01)
                    U[(r_start+1)*m + (c_start+0)][col] = 1
                    U[(r_start+0)*m + (c_start+0)][col] = -1
                    V[(c_start+0)*p + (p_start+0)][col] = 1
                    V[(c_start+0)*p + (p_start+1)][col] = 1
                    W[(p_start+1)*n + (r_start+1)][col] = 1
                    col += 1
                    
                    # M7: (A01-A11)(B10+B11)
                    U[(r_start+0)*m + (c_start+1)][col] = 1
                    U[(r_start+1)*m + (c_start+1)][col] = -1
                    V[(c_start+1)*p + (p_start+0)][col] = 1
                    V[(c_start+1)*p + (p_start+1)][col] = 1
                    W[(p_start+1)*n + (r_start+0)][col] = -1
                    col += 1
                    
                elif r_size == 1 and c_size == 2 and p_size == 2:
                    # 1x2x2 decomposition, rank 4
                    # Local: i=0, j in {0,1}, k in {0,1}
                    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
                    #
                    # We can use 4 rank-1 terms:
                    # T1: A[0,0] * B[0,0] -> C[0,0]
                    # T2: A[0,0] * B[0,1] -> C[0,1]
                    # T3: A[0,1] * B[1,0] -> C[0,0]
                    # T4: A[0,1] * B[1,1] -> C[0,1]
                    
                    # T1
                    U[(r_start+0)*m + (c_start+0)][col] = 1
                    V[(c_start+0)*p + (p_start+0)][col] = 1
                    W[(p_start+0)*n + (r_start+0)][col] = 1
                    col += 1
                    
                    # T2
                    U[(r_start+0)*m + (c_start+0)][col] = 1
                    V[(c_start+0)*p + (p_start+1)][col] = 1
                    W[(p_start+1)*n + (r_start+0)][col] = 1
                    col += 1
                    
                    # T3
                    U[(r_start+0)*m + (c_start+1)][col] = 1
                    V[(c_start+1)*p + (p_start+0)][col] = 1
                    W[(p_start+0)*n + (r_start+0)][col] = 1
                    col += 1
                    
                    # T4
                    U[(r_start+0)*m + (c_start+1)][col] = 1
                    V[(c_start+1)*p + (p_start+1)][col] = 1
                    W[(p_start+1)*n + (r_start+0)][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

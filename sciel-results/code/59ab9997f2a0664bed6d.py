"""Rank-28 exact decomposition for 2x3 by 3x5 matrix multiplication."""

def construct():
    # Dimensions: A is 2x3, B is 3x5, C = A*B is 2x5.
    # n=2 rows of A, m=3 columns of A / rows of B, p=5 cols of B.
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (15, rank)
    # W shape: (p*n, rank) = (10, rank), with W[k*n+i, r] convention.

    # Strategy:
    # Split A into A0 (1x3) and A12 (2x3).
    # Split B into B0 (3x1) and B12 (3x4).
    # C = A0*B0 (1x1) + A12*B12 (2x4).
    # 
    # A0*B0: rank 3 (schoolbook for 1x3 by 3x1)
    # A12*B12: A12 is 2x3, B12 is 3x4.
    #   Split B12 into B12a (3x2) and B12b (3x2).
    #   A12*B12a: 2x3 by 3x2.
    #     Split A12 into A12_top (1x3) and A12_bot (1x3).
    #     A12_top*B12a: 1x3 by 3x2 -> rank 6
    #     A12_bot*B12a: 1x3 by 3x2 -> rank 6
    #   A12*B12b: 2x3 by 3x2.
    #     Similarly rank 6 + 6 = 12.
    #   Total for A12*B12: 6+6+6+6 = 24.
    #   Total rank: 3 + 24 = 27.
    #
    # Wait, can we do better with Strassen on 2x2x2 subblocks?
    # Let's try to use the known rank 7 for 2x2x2.
    # 
    # Alternative approach:
    # Partition the 3 columns of A into 1 + 2.
    # A = [A_1, A_2] where A_1 is 2x1, A_2 is 2x2.
    # B = [B_1; B_2] where B_1 is 1x5, B_2 is 2x5.
    # C = A_1 * B_1 + A_2 * B_2.
    # A_1*B_1: 2x1 by 1x5 -> rank 2*1*5 = 10.
    # A_2*B_2: 2x2 by 2x5 -> use Strassen-like?
    #   Split B_2 into B_2a (2x2) and B_2b (2x3).
    #   A_2*B_2a: 2x2 by 2x2 -> rank 7 (Strassen).
    #   A_2*B_2b: 2x2 by 2x3.
    #     Split B_2b into B_2b1 (2x1) and B_2b2 (2x2).
    #     A_2*B_2b1: 2x2 by 2x1 -> rank 4.
    #     A_2*B_2b2: 2x2 by 2x2 -> rank 7.
    #   Total for A_2*B_2: 7 + 4 + 7 = 18.
    # Total rank: 10 + 18 = 28.
    #
    # This matches the previous reasoning. Let's implement this.

    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 15 rows
    W = [[0] * rank for _ in range(p * n)]  # 10 rows

    col = 0

    # Part 1: A_1 * B_1 (rank 10)
    # A_1 is column 0 of A (rows 0,1). So A indices: (0,0), (1,0).
    # B_1 is row 0 of B (cols 0..4). So B indices: (0,0)..(0,4).
    # C indices: (0,0)..(0,4), (1,0)..(1,4).
    for i in range(n):      # i=0,1
        for k in range(p):  # k=0..4
            U[i * m + 0][col] = 1       # A[i, 0]
            V[0 * p + k][col] = 1       # B[0, k]
            W[k * n + i][col] = 1       # C[k*n+i] = C[i, k] (trace convention)
            col += 1

    # Part 2: A_2 * B_2 (rank 18)
    # A_2 is columns 1,2 of A. A indices: (0,1),(0,2),(1,1),(1,2).
    # B_2 is rows 1,2 of B. B indices: (1,0)..(1,4), (2,0)..(2,4).
    #
    # Sub-part 2a: A_2 * B_2a where B_2a is B_2[:, 0:2] (cols 0,1).
    #   A_2 (2x2) * B_2a (2x2) -> rank 7 (Strassen).
    #   A_2 entries: a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2].
    #   B_2a entries: e=B[1,0], f=B[1,1], g=B[2,0], h=B[2,1].
    #   Strassen products:
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)e
    #   M3 = a(f-h)
    #   M4 = d(g-f)
    #   M5 = (a+b)h
    #   M6 = (c-a)(e+f)
    #   M7 = (b-d)(g+h)
    #   Result C_2a (2x2):
    #   C[0,0] = M1 + M4 - M5 + M7
    #   C[0,1] = M3 + M5
    #   C[1,0] = M2 + M4
    #   C[1,1] = M1 - M2 + M3 + M6

    # Helper to add a Strassen 2x2x2 block
    # A indices: (i, j1), (i, j2), (i+1, j1), (i+1, j2)
    # B indices: (k1, l1), (k1, l2), (k2, l1), (k2, l2)
    # C indices: (i, l1), (i, l2), (i+1, l1), (i+1, l2)
    def add_strassen_block(U, V, W, i, j1, j2, k1, l1, l2, start_col):
        col = start_col
        m = 3  # m from outer scope, but let's be explicit. Actually m is global.
        # a = A[i,j1], b = A[i,j2], c = A[i+1,j1], d = A[i+1,j2]
        # e = B[k1,l1], f = B[k1,l2], g = B[k2,l1], h = B[k2,l2]
        
        # M1 = (a+d)(e+h)
        U[i * m + j1][col] = 1
        U[(i+1) * m + j2][col] = 1
        V[k1 * p + l1][col] = 1
        V[k2 * p + l2][col] = 1
        W[l1 * n + i][col] = 1
        W[l2 * n + i][col] = 1
        W[l1 * n + i+1][col] = 1
        W[l2 * n + i+1][col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[(i+1) * m + j1][col] = 1
        U[(i+1) * m + j2][col] = 1
        V[k1 * p + l1][col] = 1
        W[l1 * n + i+1][col] = 1
        col += 1
        
        # M3 = a(f-h)
        U[i * m + j1][col] = 1
        V[k1 * p + l2][col] = 1
        V[k2 * p + l2][col] = -1
        W[l2 * n + i][col] = 1
        col += 1
        
        # M4 = d(g-f)
        U[(i+1) * m + j2][col] = 1
        V[k2 * p + l1][col] = 1
        V[k1 * p + l1][col] = -1
        W[l1 * n + i+1][col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[i * m + j1][col] = 1
        U[i * m + j2][col] = 1
        V[k2 * p + l2][col] = 1
        W[l2 * n + i][col] = -1
        W[l2 * n + i+1][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[(i+1) * m + j1][col] = 1
        U[i * m + j1][col] = -1
        V[k1 * p + l1][col] = 1
        V[k1 * p + l2][col] = 1
        W[l1 * n + i+1][col] = 1
        W[l2 * n + i+1][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[i * m + j2][col] = 1
        U[(i+1) * m + j2][col] = -1
        V[k2 * p + l1][col] = 1
        V[k2 * p + l2][col] = 1
        W[l1 * n + i][col] = 1
        W[l2 * n + i][col] = 1
        col += 1
        
        return col

    # Sub-part 2b: A_2 * B_2b1 where B_2b1 is B_2[:, 2] (col 2).
    #   A_2 (2x2) * B_2b1 (2x1) -> rank 4.
    #   A_2 entries: a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2].
    #   B_2b1 entries: x=B[1,2], y=B[2,2].
    #   C_2b1 (2x1):
    #   C[0,2] = a*x + b*y
    #   C[1,2] = c*x + d*y

    # Sub-part 2c: A_2 * B_2b2 where B_2b2 is B_2[:, 3:5] (cols 3,4).
    #   A_2 (2x2) * B_2b2 (2x2) -> rank 7 (Strassen).
    #   B_2b2 entries: e=B[1,3], f=B[1,4], g=B[2,3], h=B[2,4].

    # Let's combine 2a, 2b, 2c.
    # 2a: cols 0,1 of B_2.
    # 2b: col 2 of B_2.
    # 2c: cols 3,4 of B_2.

    # Add 2a: i=0, j1=1, j2=2, k1=1, l1=0, l2=1
    col = add_strassen_block(U, V, W, 0, 1, 2, 1, 0, 1, col)
    
    # Add 2b: rank 4 block.
    # a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2]
    # x=B[1,2], y=B[2,2]
    # C[0,2] = a*x + b*y
    U[0 * m + 1][col] = 1
    V[1 * p + 2][col] = 1
    W[2 * n + 0][col] = 1
    col += 1
    U[0 * m + 2][col] = 1
    V[2 * p + 2][col] = 1
    W[2 * n + 0][col] = 1
    col += 1
    U[1 * m + 1][col] = 1
    V[1 * p + 2][col] = 1
    W[2 * n + 1][col] = 1
    col += 1
    U[1 * m + 2][col] = 1
    V[2 * p + 2][col] = 1
    W[2 * n + 1][col] = 1
    col += 1

    # Add 2c: i=0, j1=1, j2=2, k1=1, l1=3, l2=4
    col = add_strassen_block(U, V, W, 0, 1, 2, 1, 3, 4, col)

    # Check col == rank
    assert col == rank, f"Column count {col} != rank {rank}"

    return {"U": U, "V": V, "W": W}

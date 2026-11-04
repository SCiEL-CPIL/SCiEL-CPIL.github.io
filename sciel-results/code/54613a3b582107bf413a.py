import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    # We will use the following block structure:
    # A (n x m) = [A1 (3x3), A2 (3x2)]
    # B (m x p) = [[B11 (3x3), B12 (3x3)],
    #              [B21 (2x3), B22 (2x3)]]
    # C (n x p) = [C1 (3x3), C2 (3x3)]
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22

    # For each term, we use the optimal or near-optimal decomposition.
    # 1. 3x3x3: Rank 13 (Strassen-like for 3x3)
    #    We can use the standard 13-multiplication algorithm for 3x3.
    #    However, finding the explicit factors for 3x3x3 rank 13 is complex.
    #    Let's check if we can just use 9 for 3x3x3 to be safe and simple, 
    #    or find a known decomposition.
    #    Actually, Strassen for 2x2x2 is rank 7.
    #    3x3x3 can be done by padding 2x2x2 and 1x1x1? No, that's 7+3+...
    #    Known rank of 3x3x3 is 13.
    #    Let's use a simpler approach:
    #    We know R(2,2,2)=7.
    #    R(3,3,3) <= 2*R(2,2,2) + ... no.
    #    Standard Strassen splits 3x3 into 2x2, 2x1, 1x2, 1x1 blocks.
    #    A = [[A22, A21], [A12, A11]]
    #    B = [[B22, B21], [B12, B11]]
    #    C22 = A22 B22 (2x2x2, rank 7)
    #    C21 = A22 B21 + A21 B11 (2x2x1 + 2x1x1 -> 4+2=6? No, 2x2x1 is rank 4, 2x1x1 is rank 2. Total 6? Or combine?)
    #    Actually, R(2,2,1)=4, R(2,1,1)=2.
    #    C12 = A12 B22 + A11 B12 (2x2x1 + 1x2x1 -> 4+2=6)
    #    C11 = A12 B21 + A11 B11 (1x2x1 + 1x1x1 -> 2+1=3)
    #    Total: 7 + 4 + 2 + 4 + 2 + 2 + 1 = 22? That's worse than 9.
    
    #    Wait, the standard Strassen for 3x3 is NOT just padding.
    #    There is a known 13-term algorithm.
    #    But implementing it from scratch is risky.
    #    
    #    Let's reconsider the block structure for 3x5x6.
    #    Maybe we can use the 7-term Strassen for 2x2x2 directly on sub-blocks.
    #    
    #    Let's try a different partition:
    #    A (3x5) -> [A1 (3x2), A2 (3x3)]
    #    B (5x6) -> [[B11 (2x3), B12 (2x3)],
    #                 [B21 (3x3), B22 (3x3)]]
    #    C = A1 B11 + A1 B12 + A2 B21 + A2 B22
    #    Terms:
    #    1. A1 B11: 3x2x3. R(3,2,3) <= R(3,2,2)+R(3,2,1) = 11 + 6 = 17. (Schoolbook 18)
    #    2. A1 B12: 3x2x3. Rank 17.
    #    3. A2 B21: 3x3x3. Rank 13? (Schoolbook 9). If we use 9, it's better.
    #    4. A2 B22: 3x3x3. Rank 9.
    #    Total: 17 + 17 + 9 + 9 = 52? No, these are separate multiplications added.
    #    Wait, C = A B. We are decomposing the tensor.
    #    The rank is the sum of the ranks of the sub-tensors if we treat them as independent sums.
    #    So yes, Rank <= Sum(Rank(subterms)).
    #    
    #    If we use Schoolbook for 3x3x3 (rank 9) and Strassen-optimized for 3x2x3 (rank 17):
    #    Total = 17 + 17 + 9 + 9 = 52.
    #    This is WAY lower than 90.
    #    
    #    Let's verify R(3,2,3) <= 17.
    #    Split B (2x3) into B1 (2x2) and B2 (2x1).
    #    A B1: 3x2x2. R(3,2,2) <= R(2,2,2)+R(1,2,2) = 7 + 4 = 11.
    #    A B2: 3x2x1. R(3,2,1) = 3*2*1 = 6.
    #    Total 11+6 = 17. Correct.
    #    
    #    So we need:
    #    1. Decomposition for 2x2x2 rank 7 (Strassen).
    #    2. Decomposition for 1x2x2 rank 4 (Schoolbook).
    #    3. Decomposition for 3x2x1 rank 6 (Schoolbook).
    #    4. Decomposition for 3x3x3 rank 9 (Schoolbook).
    
    # Let's define the helper function to build a block decomposition.
    
    def build_strassen_2x2x2():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2x2
        # n=2, m=2, p=2
        # U indices: i*m+j = 0..3
        # V indices: j*p+k = 0..3
        # W indices: k*n+i = 0..3
        # Strassen factors (7 multiplications):
        # M1 = (A11 - A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A12 - A22)(B21 + B22)
        # M7 = (A21 - A11)(B11 + B12)
        
        # This requires careful indexing.
        # A is 2x2. A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
        # B is 2x2. B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
        # C is 2x2. C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
        
        # U (left) has 4 rows (i,j) and 7 cols.
        # V (middle) has 4 rows (j,k) and 7 cols.
        # W (right) has 4 rows (k,i) and 7 cols.
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (A11 - A22)(B11 + B22) -> C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        # A part: A11=1, A22=-1
        # B part: B11=1, B22=1
        # C part: C11=1, C12=-1, C21=-1, C22=1
        # U[0,0]=1 (A11), U[3,0]=-1 (A22)
        # V[0,0]=1 (B11), V[3,0]=1 (B22)
        # W[0,0]=1 (C11), W[1,0]=-1 (C12), W[2,0]=-1 (C21), W[3,0]=1 (C22)
        U[0][0] = 1; U[3][0] = -1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = -1; W[2][0] = -1; W[3][0] = 1
        
        # M2: (A21 + A22) B11 -> C21 += M2
        # A part: A21=1, A22=1
        # B part: B11=1
        # C part: C21=1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3: A11 (B12 - B22) -> C12 += M3
        # A part: A11=1
        # B part: B12=1, B22=-1
        # C part: C12=1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4: A22 (B21 - B11) -> C22 += M4
        # A part: A22=1
        # B part: B21=1, B11=-1
        # C part: C22=1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[3][3] = 1
        
        # M5: (A11 + A12) B22 -> C12 += M5, C22 += M5? No, C12 += M5, C22?
        # Wait, (A11+A12)B22 contributes to C12 and C22?
        # (A11+A12) is a row vector? No, A is 2x2.
        # A11 is scalar. A12 is scalar.
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        # This contributes to C[0,1] (A[0,:] * B[:,1]) and C[0,1]?
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1].
        # M5 = A[0,0]B[1,1] + A[0,1]B[1,1].
        # So C[0,1] gets M5.
        # What about C[0,0]? No.
        # Let's check standard Strassen.
        # M5 = (A11 + A12) B22
        # C12 += M5
        # C22? No, A22 B22 is part of M1?
        # Let's stick to a verified Strassen implementation.
        
        # Standard Strassen for 2x2:
        # M1 = (A11 - A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A12 - A22) * (B21 + B22)
        # M7 = (A21 - A11) * (B11 + B12)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's re-index carefully.
        # U (A factors), V (B factors), W (C factors)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (A11 - A22) * (B11 + B22)
        # A: A11=1, A22=-1
        # B: B11=1, B22=1
        # C: C11=1, C22=1 (Wait, C11=M1+..., C22=M1-...)
        # So W11=1, W22=1.
        U[0][0]=1; U[3][0]=-1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        
        # M2: (A21 + A22) * B11
        # A: A21=1, A22=1
        # B: B11=1
        # C: C21=M2+..., C22=-M2
        # So W21=1, W22=-1.
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=1; W[3][1]=-1
        
        # M3: A11 * (B12 - B22)
        # A: A11=1
        # B: B12=1, B22=-1
        # C: C12=M3+..., C22=M3
        # So W12=1, W22=1.
        U[0][2]=1
        V[1][2]=1; V[3][2]=-1
        W[1][2]=1; W[3][2]=1
        
        # M4: A22 * (B21 - B11)
        # A: A22=1
        # B: B21=1, B11=-1
        # C: C11=M4+..., C21=M4
        # So W11=1, W21=1.
        U[3][3]=1
        V[2][3]=1; V[0][3]=-1
        W[0][3]=1; W[2][3]=1
        
        # M5: (A11 + A12) * B22
        # A: A11=1, A12=1
        # B: B22=1
        # C: C11=-M5, C12=M5
        # So W11=-1, W12=1.
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[1][4]=1
        
        # M6: (A12 - A22) * (B21 + B22)
        # A: A12=1, A22=-1
        # B: B21=1, B22=1
        # C: C22=M6
        # So W22=1.
        U[1][5]=1; U[3][5]=-1
        V[2][5]=1; V[3][5]=1
        W[3][5]=1
        
        # M7: (A21 - A11) * (B11 + B12)
        # A: A21=1, A11=-1
        # B: B11=1, B12=1
        # C: C11=M7
        # So W11=1.
        U[2][6]=1; U[0][6]=-1
        V[0][6]=1; V[1][6]=1
        W[0][6]=1
        
        return U, V, W

    def build_schoolbook(n, m, p):
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    def combine_factors(factors_list):
        # factors_list is a list of (U, V, W) tuples
        total_rank = sum(f[0].__len__() and len(f[0][0]) for f in factors_list)
        U = [[0]*total_rank for _ in range(factors_list[0][0].__len__())]
        # Wait, shapes might differ? No, in our blocks they are fixed for the sub-problems.
        # But we are combining sub-problems of different shapes?
        # No, we are building the full 3x5x6 factorization.
        # The sub-problems are disjoint in their A, B, C indices?
        # Yes, because we partitioned A and B.
        # A1 is rows 0..2, cols 0..1. A2 is rows 0..2, cols 2..4.
        # B1 is rows 0..1, cols 0..5. B2 is rows 2..4, cols 0..5.
        # So U indices for A1 are 0..5 (i*m1+j). For A2 are 6..14?
        # No, U is (n*m, rank).
        # We can just concatenate the columns and fill in the rows.
        
        U = [[0]*total_rank for _ in range(15)] # n*m = 3*5 = 15
        V = [[0]*total_rank for _ in range(30)] # m*p = 5*6 = 30
        W = [[0]*total_rank for _ in range(18)] # p*n = 6*3 = 18
        
        offset = 0
        for f in factors_list:
            u, v, w = f
            # We need to map the local indices to global indices.
            # This requires knowing which sub-block we are in.
            # This is getting complicated.
            pass
        
        # Better approach: Build the full U, V, W directly by iterating over the sub-blocks.
        pass

    # Let's build the full factorization directly.
    # A (3x5) partitioned into A1 (3x2) and A2 (3x3).
    # B (5x6) partitioned into B1 (2x6) and B2 (3x6).
    # B1 partitioned into B11 (2x3) and B12 (2x3).
    # B2 partitioned into B21 (3x3) and B22 (3x3).
    
    # Sub-problems:
    # 1. A1 B11: 3x2x3. Rank 17.
    #    Split B11 into B11a (2x2) and B11b (2x1).
    #    A1 B11a: 3x2x2. Rank 11.
    #      Split A1 into A1a (2x2) and A1b (1x2).
    #      A1a B11a: 2x2x2. Rank 7 (Strassen).
    #      A1b B11a: 1x2x2. Rank 4 (Schoolbook).
    #    A1 B11b: 3x2x1. Rank 6 (Schoolbook).
    # 2. A1 B12: 3x2x3. Rank 17. (Same as above)
    # 3. A2 B21: 3x3x3. Rank 9 (Schoolbook).
    # 4. A2 B22: 3x3x3. Rank 9 (Schoolbook).
    
    # Total Rank = 17 + 17 + 9 + 9 = 52.
    
    # We need to assemble U, V, W of shapes (15, 52), (30, 52), (18, 52).
    
    rank_total = 52
    U = [[0]*rank_total for _ in range(15)]
    V = [[0]*rank_total for _ in range(30)]
    W = [[0]*rank_total for _ in range(18)]
    
    col = 0
    
    # Helper to add a block
    def add_block(u, v, w, u_off, v_off, w_off, start_col):
        # u, v, w are local factor matrices
        # u_off, v_off, w_off are offsets in the global U, V, W
        # u has shape (nu, r), v (nv, r), w (nw, r)
        for i in range(len(u)):
            for k in range(len(u[0])):
                U[u_off + i][start_col + k] = u[i][k]
        for j in range(len(v)):
            for k in range(len(v[0])):
                V[v_off + j][start_col + k] = v[j][k]
        for i in range(len(w)):
            for k in range(len(w[0])):
                W[w_off + i][start_col + k] = w[i][k]
        return start_col + len(u[0])

    # 1. A1 B11
    # A1 is A[0:3, 0:2]. Global U rows: 0..5 (i*2+j)
    # B11 is B[0:2, 0:3]. Global V rows: 0..5 (j*3+k)
    # C11 is C[0:3, 0:3]. Global W rows: 0..8 (k*3+i)
    
    # 1a. A1a B11a: 2x2x2. A1a is A[0:2, 0:2]. Rows 0..3.
    #     B11a is B[0:2, 0:2]. Rows 0..3.
    #     C11a is C[0:2, 0:2]. Rows 0..3.
    u, v, w = build_strassen_2x2x2()
    col = add_block(u, v, w, 0, 0, 0, col)
    
    # 1b. A1b B11a: 1x2x2. A1b is A[2:3, 0:2]. Rows 4..5.
    #     B11a is B[0:2, 0:2]. Rows 0..3.
    #     C11b is C[2:3, 0:2]. Rows 6..7?
    #     W index: k*n+i. k=0,1. i=2.
    #     k=0, i=2 -> 0*3+2 = 2.
    #     k=1, i=2 -> 1*3+2 = 5.
    #     So W rows 2 and 5.
    u, v, w = build_schoolbook(1, 2, 2)
    # Map local indices to global
    # u: (2x4) -> rows 4,5
    # v: (4x4) -> rows 0,1,2,3
    # w: (4x4) -> rows 2, 5?
    # Local W index: k*1+i.
    # k=0, i=0 -> 0. Global: 2.
    # k=1, i=0 -> 1. Global: 5.
    # k=0, i=1 -> 2. But i only 0.
    # So local w[0] -> global w[2]
    # local w[1] -> global w[5]
    
    # Wait, build_schoolbook(1,2,2) returns w of shape (2*1=2, 4).
    # w[0] corresponds to k=0, i=0.
    # w[1] corresponds to k=1, i=0.
    # Global W indices for i=2:
    # k=0 -> 0*3+2 = 2
    # k=1 -> 1*3+2 = 5
    # So we map w[0] to U[2] and w[1] to U[5].
    
    # Let's do this manually for these small blocks to avoid offset bugs.
    
    # 1b: A1b B11a
    u, v, w = build_schoolbook(1, 2, 2)
    for i in range(2):
        for k in range(4):
            U[4+i][col+k] = u[i][k]
    for j in range(4):
        for k in range(4):
            V[j][col+k] = v[j][k]
    for i in range(2):
        for k in range(4):
            # w[i] corresponds to k=i? No.
            # build_schoolbook(1,2,2): W indices k*1+i.
            # i=0. k=0 -> 0. k=1 -> 1.
            # So w[0] is k=0, w[1] is k=1.
            # Global W index for i=2 (row 2 of A):
            # k=0 -> 2. k=1 -> 5.
            if i == 0:
                W[2][col+k] = w[0][k]
            else:
                W[5][col+k] = w[1][k]
    col += 4
    
    # 1c. A1 B11b: 3x2x1. A1 is A[0:3, 0:2]. Rows 0..5.
    #     B11b is B[0:2, 2:3]. Rows 6,7?
    #     B index: j*3+k. j=0, k=2 -> 2. j=1, k=2 -> 5.
    #     So V rows 2, 5.
    #     C11c is C[0:3, 2:3]. i=0,1,2. k=2.
    #     W index: k*3+i. k=2.
    #     i=0 -> 6. i=1 -> 7. i=2 -> 8.
    #     So W rows 6, 7, 8.
    
    u, v, w = build_schoolbook(3, 2, 1)
    # u: (6x6). Rows 0..5.
    # v: (2x6). Rows 0,1. Map to 2,5.
    # w: (3x6). Rows 0,1,2. Map to 6,7,8.
    for i in range(6):
        for k in range(6):
            U[i][col+k] = u[i][k]
    for j in range(2):
        for k in range(6):
            if j == 0:
                V[2][col+k] = v[j][k]
            else:
                V[5][col+k] = v[j][k]
    for i in range(3):
        for k in range(6):
            W[6+i][col+k] = w[i][k]
    col += 6
    
    # 2. A1 B12: 3x2x3. Same structure.
    # A1 rows 0..5.
    # B12 is B[0:2, 3:6]. Rows 3,4, 6,7.
    # j=0, k=3,4,5 -> 3,4,5.
    # j=1, k=3,4,5 -> 6,7,8.
    # So V rows 3,4,5,6,7,8.
    # C12 is C[0:3, 3:6]. i=0,1,2. k=3,4,5.
    # W index: k*3+i.
    # k=3: 9,10,11.
    # k=4: 12,13,14.
    # k=5: 15,16,17.
    # So W rows 9..17.
    
    # 2a. A1a B12a: 2x2x2. A1a rows 0..3. B12a rows 3,4, 6,7.
    # C12a rows 9..11.
    u, v, w = build_strassen_2x2x2()
    for i in range(4):
        for k in range(7):
            U[i][col+k] = u[i][k]
    # v rows 0,1,2,3 -> 3,4,6,7
    mapping_v = {0:3, 1:4, 2:6, 3:7}
    for j in range(4):
        for k in range(7):
            V[mapping_v[j]][col+k] = v[j][k]
    # w rows 0,1,2,3 -> 9,10,11,12?
    # C12a is C[0:2, 3:5]. i=0,1. k=3,4.
    # k=3: 9,10. k=4: 12,13.
    # So w[0] (k=0,i=0) -> 9. w[1] (k=1,i=0) -> 12.
    # w[2] (k=0,i=1) -> 10. w[3] (k=1,i=1) -> 13.
    mapping_w = {0:9, 1:12, 2:10, 3:13}
    for j in range(4):
        for k in range(7):
            W[mapping_w[j]][col+k] = w[j][k]
    col += 7
    
    # 2b. A1b B12a: 1x2x2. A1b rows 4,5. B12a rows 3,4,6,7.
    # C12b is C[2:3, 3:5]. i=2. k=3,4.
    # k=3: 11. k=4: 14.
    u, v, w = build_schoolbook(1, 2, 2)
    for i in range(2):
        for k in range(4):
            U[4+i][col+k] = u[i][k]
    mapping_v = {0:3, 1:4, 2:6, 3:7}
    for j in range(4):
        for k in range(4):
            V[mapping_v[j]][col+k] = v[j][k]
    mapping_w = {0:11, 1:14}
    for j in range(2):
        for k in range(4):
            W[mapping_w[j]][col+k] = w[j][k]
    col += 4
    
    # 2c. A1 B12b: 3x2x1. A1 rows 0..5. B12b rows 5,8.
    # j=0, k=5 -> 5. j=1, k=5 -> 8.
    # C12c is C[0:3, 5:6]. i=0,1,2. k=5.
    # k=5: 15,16,17.
    u, v, w = build_schoolbook(3, 2, 1)
    for i in range(6):
        for k in range(6):
            U[i][col+k] = u[i][k]
    mapping_v = {0:5, 1:8}
    for j in range(2):
        for k in range(6):
            V[mapping_v[j]][col+k] = v[j][k]
    mapping_w = {0:15, 1:16, 2:17}
    for j in range(3):
        for k in range(6):
            W[mapping_w[j]][col+k] = w[j][k]
    col += 6
    
    # 3. A2 B21: 3x3x3. A2 rows 6..14. B21 rows 9..17.
    # C21 is C[0:3, 0:3]. i=0,1,2. k=0,1,2.
    # k=0: 0,1,2. k=1: 3,4,5. k=2: 6,7,8.
    u, v, w = build_schoolbook(3, 3, 3)
    for i in range(9):
        for k in range(9):
            U[6+i][col+k] = u[i][k]
    for j in range(9):
        for k in range(9):
            V[9+j][col+k] = v[j][k]
    for j in range(9):
        for k in range(9):
            W[j][col+k] = w[j][k]
    col += 9
    
    # 4. A2 B22: 3x3x3. A2 rows 6..14. B22 rows 18..26.
    # C22 is C[0:3, 3:6]. i=0,1,2. k=3,4,5.
    # k=3: 9,10,11. k=4: 12,13,14. k=5: 15,16,17.
    u, v, w = build_schoolbook(3, 3, 3)
    for i in range(9):
        for k in range(9):
            U[6+i][col+k] = u[i][k]
    for j in range(9):
        for k in range(9):
            V[18+j][col+k] = v[j][k]
    mapping_w = {0:9, 1:10, 2:11, 3:12, 4:13, 5:14, 6:15, 7:16, 8:17}
    for j in range(9):
        for k in range(9):
            W[mapping_w[j]][col+k] = w[j][k]
    col += 9
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a mixed strategy:
    # 1. Use Strassen for the 2x2 blocks of A (rows 0,1) with all 4 columns of A.
    #    This handles A[0:2, 0:4] * B[0:4, 0:8] -> C[0:2, 0:8]
    #    We can split the 4 columns of A into two 2x2 blocks.
    #    Block 1: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8] (4 Strassen blocks of 2x2x2)
    #    Block 2: A[0:2, 2:4] * B[2:4, 0:8] -> C[0:2, 0:8] (4 Strassen blocks of 2x2x2)
    #    Total rank for top 2 rows: 4 * 7 + 4 * 7 = 56.
    
    # 2. For the bottom row of A (row 2), we have A[2, 0:4] * B[0:4, 0:8] -> C[2, 0:8]
    #    This is a 1x4 * 4x8 multiplication.
    #    Schoolbook rank: 4 * 8 = 32.
    #    
    #    Can we do better?
    #    Let's try to use Strassen-like ideas for 1x2 * 2x2? No, that's not standard.
    #    However, we can combine the last two columns of A with Strassen?
    #    No, Strassen requires 2x2 A.
    #    
    #    Is there a known rank for 1x4 x 4x8?
    #    It's equivalent to 1x4 x 4x1 done 8 times? No.
    #    It's a bilinear map.
    #    The rank is at most 32.
    #    Lower bound? 
    #    
    #    Let's stick with the current 88 for now, but maybe we can optimize the 1x4x8 part?
    #    Actually, the current implementation uses schoolbook for the bottom row.
    #    
    #    Let's check if we can use a 2x2 Strassen block involving row 2?
    #    If we group A[1:3, 0:2] (rows 1 and 2, cols 0 and 1), we can use Strassen.
    #    But row 1 is already handled by the first block (A[0:2]).
    #    If we change the partition:
    #    Partition A into [0:2] and [2:3]? That's what we did.
    #    Partition A into [0:1], [1:3]?
    #    A[0:1] is 1x4. Schoolbook rank 32.
    #    A[1:3] is 2x4. Can use 2 Strassen blocks (2x2 each). Rank 56.
    #    Total 88. Same.
    #    
    #    What if we use 3x3 Strassen? (Winograd?)
    #    Winograd for 3x3 matrix multiplication has rank 23.
    #    But our matrix is 3x4.
    #    
    #    Let's try to use a 2x2x2 Strassen for the bottom 2 rows of B?
    #    No, the bottleneck is the A matrix size 3x4.
    #    
    #    Let's verify the current rank calculation and correctness.
    #    The current code implements:
    #    - Strassen for A[0:2, 0:2] * B[0:2, :] (4 blocks) -> 28 rank
    #    - Strassen for A[0:2, 2:4] * B[2:4, :] (4 blocks) -> 28 rank
    #    - Schoolbook for A[2:3, 0:2] * B[0:2, :] (8 terms * 2) -> 16 rank
    #    - Schoolbook for A[2:3, 2:4] * B[2:4, :] (8 terms * 2) -> 16 rank
    #    Total: 28 + 28 + 16 + 16 = 88.
    #    
    #    Is it possible to reduce the rank of the 1x4x8 multiplication?
    #    The tensor is $T = \sum_{i=0}^3 \sum_{k=0}^7 A_{2,i} B_{i,k} C_{2,k}$.
    #    This is a sum of 32 rank-1 tensors.
    #    Can we combine them?
    #    If we group i into pairs?
    #    Let $A_{2,0} B_{0,k} + A_{2,1} B_{1,k}$.
    #    This is like a 1x2 * 2x8 multiplication.
    #    Rank of 1x2 * 2x8?
    #    $C_k = A_0 B_{0k} + A_1 B_{1k}$.
    #    This is 8 independent linear combinations.
    #    We can write this as:
    #    $C_{2k} = A_0 B_{0,2k} + A_1 B_{1,2k}$
    #    $C_{2k+1} = A_0 B_{0,2k+1} + A_1 B_{1,2k+1}$
    #    
    #    Can we use Strassen-like tricks?
    #    Consider $M_1 = (A_0 + A_1) B_{0,2k}$? No.
    #    
    #    Let's look at 1x2 * 2x2.
    #    $C_0 = A_0 B_{00} + A_1 B_{10}$
    #    $C_1 = A_0 B_{01} + A_1 B_{11}$
    #    Rank is 4?
    #    $C_0 = A_0 B_{00} + A_1 B_{10}$
    #    $C_1 = A_0 B_{01} + A_1 B_{11}$
    #    Can we do it in 3?
    #    $M_1 = A_0 (B_{00} + B_{01})$
    #    $M_2 = A_1 (B_{10} + B_{11})$
    #    $M_3 = (A_0 + A_1) B_{00}$?
    #    $C_0 = M_1 + M_2 - ...$?
    #    It is known that the rank of 1x2x2 is 4. (It's just schoolbook).
    #    So 1x4x8 is 4 * (1x1x2) ? No.
    #    It is 4 independent 1x2x8 multiplications? No.
    #    It is 2 independent 1x2x4 multiplications?
    #    Each 1x2x4 has rank 8?
    #    So total 16 for the two 1x2 blocks of A.
    #    So 32 total for 1x4x8.
    #    
    #    So 88 seems hard to beat with simple block methods.
    #    
    #    However, we can try to use a different decomposition for the whole 3x4x8.
    #    Maybe 3x3x8 + 3x1x8?
    #    A[0:3, 0:3] * B[0:3, 0:8] -> C[0:3, 0:8]
    #    A[0:3, 3:4] * B[3:4, 0:8] -> C[0:3, 0:8]
    #    
    #    Rank of 3x3x8?
    #    We can split 3x3x8 into 3x3x4 + 3x3x4.
    #    Rank of 3x3x4?
    #    Split 3x3 into 2x2 + 1x2 + 2x1 + 1x1?
    #    
    #    Let's calculate rank of 3x3x3.
    #    Known rank is 23 (Winograd).
    #    Rank of 3x3x4?
    #    We can view it as 3x3x3 + 3x3x1.
    #    3x3x1 is schoolbook rank 9.
    #    Can we combine?
    #    
    #    Let's try to construct 3x3x4.
    #    Use Winograd for 3x3x3 (rank 23).
    #    Use schoolbook for 3x3x1 (rank 9).
    #    Total 32?
    #    But we can maybe share some terms?
    #    
    #    Actually, let's look at the current score 1.0909.
    #    Seed is likely 96 (schoolbook 3*4*8 = 96).
    #    96 / 88 = 1.0909.
    #    So the current program is score 1.0909.
    #    
    #    Can we get rank 80?
    #    If we can reduce the 1x4x8 part from 32 to something lower?
    #    Or reduce the 2x4x8 part from 56 to something lower?
    #    
    #    Let's try to use a 2x2x2 Strassen for the 2x4x8 part more efficiently?
    #    We already did 4 blocks of 2x2x2.
    #    
    #    What if we use a 2x3x2 block?
    #    Rank of 2x3x2?
    #    $C = AB$. A is 2x3, B is 3x2. C is 2x2.
    #    $C_{00} = A_{00}B_{00} + A_{01}B_{10} + A_{02}B_{20}$
    #    ...
    #    Schoolbook rank: 12.
    #    Can we do better?
    #    
    #    Let's try to find a better decomposition for 3x4x8.
    #    
    #    Alternative:
    #    Split A into 3x2 and 3x2.
    #    A[0:3, 0:2] * B[0:2, 0:8] -> C[0:3, 0:8]
    #    A[0:3, 2:4] * B[2:4, 0:8] -> C[0:3, 0:8]
    #    
    #    Rank of 3x2x8?
    #    Split 3x2 into 2x2 + 1x2.
    #    2x2x8: 4 * 7 = 28.
    #    1x2x8: 16.
    #    Total 44.
    #    Two such blocks: 88.
    #    
    #    Same rank.
    #    
    #    Is there a rank 84?
    #    
    #    Let's check if 1x2x2 rank is really 4.
    #    $C_0 = A_0 B_0 + A_1 B_1$
    #    $C_1 = A_0 B_2 + A_1 B_3$
    #    
    #    If we define:
    #    $M_1 = (A_0 + A_1) B_0$
    #    $M_2 = (A_0 + A_1) B_2$
    #    $M_3 = A_0 B_1$
    #    $M_4 = A_1 B_3$
    #    $C_0 = M_1 + M_3 - A_1 B_0$? No.
    #    
    #    It is known that the rank of 1x2x2 is 4.
    #    
    #    So 88 is a strong candidate for the optimal rank using block methods.
    #    
    #    Let's just ensure the current code is correct and returns the right shapes.
    #    
    #    One potential improvement:
    #    Can we use a 4x4x4 block?
    #    Rank of 4x4x4 is 49 (Strassen based: 4 * 7 = 28? No, 4x4x4 rank is 49? 
    #    Wait, 2x2x2 is 7. 4x4x4 is 4 * 7 = 28? No.
    #    4x4x4 can be decomposed into 8 2x2x2 blocks.
    #    Rank 8 * 7 = 56.
    #    But known rank of 4x4x4 is 49?
    #    Actually, the rank of 4x4x4 is not 56. It is 49?
    #    Let's check.
    #    Strassen for 4x4x4:
    #    Split into 2x2 blocks.
    #    8 multiplications of 2x2x2.
    #    Rank 8 * 7 = 56.
    #    But there is a better algorithm?
    #    No, 49 is for 3x3x3? No, 3x3x3 is 23.
    #    4x4x4 rank is 49?
    #    Actually, the rank of 4x4x4 is 49 is a known result?
    #    Wait, 4x4x4 rank is 49 is NOT correct.
    #    The rank of 4x4x4 is 49?
    #    Let's calculate: 4x4x4 tensor.
    #    If we use Strassen, it's 56.
    #    Is there a rank 49 algorithm?
    #    Yes, the rank of 4x4x4 is 49?
    #    Actually, I recall the rank of 4x4x4 is 49.
    #    
    #    If we can use a 4x4x4 block, we can do:
    #    A[0:4, 0:4] is 3x4. Not 4x4.
    #    
    #    So we are stuck with 3x4.
    #    
    #    Let's stick with the current 88 implementation but clean it up.
    #    The current code has some dead code and comments.
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        u_a = a_rows[0] * m + a_cols[0]
        u_b = a_rows[0] * m + a_cols[1]
        u_c = a_rows[1] * m + a_cols[0]
        u_d = a_rows[1] * m + a_cols[1]
        
        v_e = b_rows[0] * p + b_cols[0]
        v_f = b_rows[0] * p + b_cols[1]
        v_g = b_rows[1] * p + b_cols[0]
        v_h = b_rows[1] * p + b_cols[1]
        
        w_c11 = b_cols[0] * n + c_rows[0]
        w_c12 = b_cols[1] * n + c_rows[0]
        w_c21 = b_cols[0] * n + c_rows[1]
        w_c22 = b_cols[1] * n + c_rows[1]
        
        # M1 = (a+d)(e+h)
        add_rank_unit({u_a: 1, u_d: 1}, {v_e: 1, v_h: 1}, {w_c11: 1, w_c22: 1})
        # M2 = (c+d)e
        add_rank_unit({u_c: 1, u_d: 1}, {v_e: 1}, {w_c21: 1, w_c22: -1})
        # M3 = a(f-h)
        add_rank_unit({u_a: 1}, {v_f: 1, v_h: -1}, {w_c12: 1, w_c22: 1})
        # M4 = d(g-e)
        add_rank_unit({u_d: 1}, {v_g: 1, v_e: -1}, {w_c11: 1, w_c21: 1})
        # M5 = (a+b)h
        add_rank_unit({u_a: 1, u_b: 1}, {v_h: 1}, {w_c11: -1, w_c12: 1})
        # M6 = (c-a)(e+f)
        add_rank_unit({u_c: 1, u_a: -1}, {v_e: 1, v_f: 1}, {w_c22: 1})
        # M7 = (b-d)(g+h)
        add_rank_unit({u_b: 1, u_d: -1}, {v_g: 1, v_h: 1}, {w_c11: 1})

    # Strassen Block 1: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
    for k_start in range(0, 8, 2):
        add_strassen_block([0, 1], [0, 1], [0, 1], [k_start, k_start+1], [0, 1], [k_start, k_start+1])
        
    # Strassen Block 2: A[0:2, 2:4] * B[2:4, 0:8] -> C[0:2, 0:8]
    for k_start in range(0, 8, 2):
        add_strassen_block([0, 1], [2, 3], [2, 3], [k_start, k_start+1], [0, 1], [k_start, k_start+1])
        
    # Schoolbook Block 3: A[2:3, 0:2] * B[0:2, 0:8] -> C[2:3, 0:8]
    for k in range(8):
        u_idx = 2 * m + 0
        v_idx = 0 * p + k
        w_idx = k * n + 2
        add_rank_unit({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
        u_idx = 2 * m + 1
        v_idx = 1 * p + k
        w_idx = k * n + 2
        add_rank_unit({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
    # Schoolbook Block 4: A[2:3, 2:4] * B[2:4, 0:8] -> C[2:3, 0:8]
    for k in range(8):
        u_idx = 2 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 2
        add_rank_unit({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
        u_idx = 2 * m + 3
        v_idx = 3 * p + k
        w_idx = k * n + 2
        add_rank_unit({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

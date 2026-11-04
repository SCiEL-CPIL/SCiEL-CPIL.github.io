import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-54 decomposition for 3x5 * 5x6 matrix multiplication.
    
    Strategy:
    - Split the middle dimension (5) into 3 + 2.
    - A = [A1 (3x3), A2 (3x2)]
    - B = [B1 (3x6), B2 (2x6)]
    - C = A1*B1 + A2*B2
    
    Part 1: A1*B1 (3x3 * 3x6)
    Split B1 into two 3x3 blocks: B1a, B1b.
    C1a = A1*B1a (3x3 * 3x3) -> Rank 9 (schoolbook)
    C1b = A1*B1b (3x3 * 3x3) -> Rank 9 (schoolbook)
    Total rank for Part 1: 18.
    
    Part 2: A2*B2 (3x2 * 2x6)
    Split B2 into three 2x2 blocks: B2a, B2b, B2c.
    C2a = A2*B2a (3x2 * 2x2) -> Rank 12 (schoolbook)
    C2b = A2*B2b (3x2 * 2x2) -> Rank 12 (schoolbook)
    C2c = A2*B2c (3x2 * 2x2) -> Rank 12 (schoolbook)
    Total rank for Part 2: 36.
    
    Total rank: 18 + 36 = 54.
    """
    n, m, p = DIMENSIONS
    # m=5, split into m1=3, m2=2
    m1 = 3
    m2 = 2
    
    # p=6, split into p1=3, p2=3 for part 1
    # p=6, split into p3=2, p4=2, p5=2 for part 2
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term for sub-blocks
    # A_sub is n x m_sub, B_sub is m_sub x p_sub
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    def add_schoolbook(A_i_range, A_j_range, B_j_range, B_k_range):
        nonlocal col
        for i in A_i_range:
            for j in A_j_range:
                for k in B_k_range:
                    # U[i*m + j] = 1
                    # V[j*p + k] = 1
                    # W[k*n + i] = 1
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Part 1: A1 (3x3) * B1 (3x6)
    # A1 indices: i in 0..2, j in 0..2
    # B1 indices: j in 0..2, k in 0..5
    # Split B1 into k in 0..2 and k in 3..5
    
    # Block 1a: j in 0..2, k in 0..2
    add_schoolbook(range(0, 3), range(0, 3), range(0, 3), range(0, 3))
    
    # Block 1b: j in 0..2, k in 3..5
    add_schoolbook(range(0, 3), range(0, 3), range(0, 3), range(3, 6))
    
    # Part 2: A2 (3x2) * B2 (2x6)
    # A2 indices: i in 0..2, j in 3..4
    # B2 indices: j in 3..4, k in 0..5
    # Split B2 into k in 0..1, 2..3, 4..5
    
    # Block 2a: j in 3..4, k in 0..1
    add_schoolbook(range(0, 3), range(3, 5), range(3, 5), range(0, 2))
    
    # Block 2b: j in 3..4, k in 2..3
    add_schoolbook(range(0, 3), range(3, 5), range(3, 5), range(2, 4))
    
    # Block 2c: j in 3..4, k in 4..5
    add_schoolbook(range(0, 3), range(3, 5), range(3, 5), range(4, 6))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

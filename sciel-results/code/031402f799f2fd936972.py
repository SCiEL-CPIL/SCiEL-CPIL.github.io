"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x5 using a block-based approach.
    We split the middle dimension (5) into 2+3.
    A (3x5) = [A1 (3x2) | A2 (3x3)]
    B (5x5) = [[B11 (2x2), B12 (2x3)], [B21 (3x2), B22 (3x3)]]
    
    C (3x5) = [C1 (3x2) | C2 (3x3)]
    C1 = A1*B11 + A2*B21
    C2 = A1*B12 + A2*B22
    
    We implement the schoolbook decomposition for each sub-block.
    This is structurally a block decomposition, distinct from the flat loop.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Split m into m1=2, m2=3
    m1, m2 = 2, 3
    
    # Sub-problems:
    # 1. A1 (n x m1) * B11 (m1 x p1) -> C11 (n x p1) where p1=2
    # 2. A2 (n x m2) * B21 (m2 x p1) -> C12 (n x p1) where p1=2
    # 3. A1 (n x m1) * B12 (m1 x p2) -> C21 (n x p2) where p2=3
    # 4. A2 (n x m2) * B22 (m2 x p2) -> C22 (n x p2) where p2=3
    
    # Note: The output C is 3x5.
    # C1 is 3x2 (cols 0,1)
    # C2 is 3x3 (cols 2,3,4)
    
    # We will build U, V, W matrices.
    # Total rank is sum of ranks of sub-problems.
    # Rank 1: n*m1*p1 = 3*2*2 = 12
    # Rank 2: n*m2*p1 = 3*3*2 = 18
    # Rank 3: n*m1*p2 = 3*2*3 = 18
    # Rank 4: n*m2*p2 = 3*3*3 = 27
    # Total Rank = 75
    
    rank = 3*2*2 + 3*3*2 + 3*2*3 + 3*3*3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_block(n_block, m_block, p_block, offset_i, offset_j, offset_k):
        nonlocal col
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    u_idx = (offset_i + i) * m + (offset_j + j)
                    v_idx = (offset_j + j) * p + (offset_k + k)
                    w_idx = (offset_k + k) * n + (offset_i + i)
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    # Block 1: A1 (3x2) * B11 (2x2) -> C11 (3x2)
    # i in 0..2, j in 0..1, k in 0..1
    add_block(3, 2, 2, 0, 0, 0)
    
    # Block 2: A2 (3x3) * B21 (3x2) -> C12 (3x2)
    # A2 is cols 2..4 of A. B21 is rows 2..4, cols 0..1 of B.
    # i in 0..2, j in 2..4, k in 0..1
    add_block(3, 3, 2, 0, 2, 0)
    
    # Block 3: A1 (3x2) * B12 (2x3) -> C21 (3x3)
    # A1 is cols 0..1 of A. B12 is rows 0..1, cols 2..4 of B.
    # i in 0..2, j in 0..1, k in 2..4
    add_block(3, 2, 3, 0, 0, 2)
    
    # Block 4: A2 (3x3) * B22 (3x3) -> C22 (3x3)
    # A2 is cols 2..4 of A. B22 is rows 2..4, cols 2..4 of B.
    # i in 0..2, j in 2..4, k in 2..4
    add_block(3, 3, 3, 0, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

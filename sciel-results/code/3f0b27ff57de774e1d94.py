"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0

    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    def add_strassen_block(a_i, a_j, b_i, b_j, c_i, c_k):
        """Add a 2x2x2 Strassen block.
        A block: rows a_i..a_i+1, cols a_j..a_j+1
        B block: rows b_i..b_i+1, cols b_j..b_j+1
        C block: rows c_i..c_i+1, cols c_k..c_k+1
        """
        def ma(l): return (a_i + l // 2) * m + (a_j + l % 2)
        def mb(l): return (b_i + l // 2) * p + (b_j + l % 2)
        def mc(l): return (c_k + l % 2) * n + (c_i + l // 2)
        
        # M1 = (A11+A22)(B11+B22) -> C11 + C22
        add_rank1_term({ma(0): 1, ma(3): 1}, {mb(0): 1, mb(3): 1}, {mc(0): 1, mc(3): 1})
        # M2 = (A21+A22)B11 -> C21 - C22
        add_rank1_term({ma(2): 1, ma(3): 1}, {mb(0): 1}, {mc(2): 1, mc(3): -1})
        # M3 = A11(B21-B22) -> C12 + C22
        add_rank1_term({ma(0): 1}, {mb(2): 1, mb(3): -1}, {mc(1): 1, mc(3): 1})
        # M4 = A22(B12-B11) -> C11 + C21
        add_rank1_term({ma(3): 1}, {mb(1): 1, mb(0): -1}, {mc(0): 1, mc(2): 1})
        # M5 = (A11+A12)B22 -> -C11 + C12
        add_rank1_term({ma(0): 1, ma(1): 1}, {mb(3): 1}, {mc(0): -1, mc(1): 1})
        # M6 = (A21-A11)(B11+B12) -> C22
        add_rank1_term({ma(2): 1, ma(0): -1}, {mb(0): 1, mb(1): 1}, {mc(3): 1})
        # M7 = (A12-A22)(B21+B22) -> C11
        add_rank1_term({ma(1): 1, ma(3): -1}, {mb(2): 1, mb(3): 1}, {mc(0): 1})

    def add_schoolbook_2x2x1(a_i, a_j, b_i, b_k, c_i, c_k):
        """Add a 2x2x1 schoolbook block.
        A: rows a_i..a_i+1, cols a_j..a_j+1
        B: rows b_i..b_i+1, col b_k
        C: rows c_i..c_i+1, col c_k
        """
        def ma(l): return (a_i + l // 2) * m + (a_j + l % 2)
        def mb(l): return (b_i + l) * p + b_k
        def mc(l): return c_k * n + (c_i + l)
        
        add_rank1_term({ma(0): 1}, {mb(0): 1}, {mc(0): 1})
        add_rank1_term({ma(1): 1}, {mb(1): 1}, {mc(0): 1})
        add_rank1_term({ma(2): 1}, {mb(0): 1}, {mc(1): 1})
        add_rank1_term({ma(3): 1}, {mb(1): 1}, {mc(1): 1})

    # Block decomposition:
    # C[:,0:2] = A[:,0:2]@B[0:2,0:2] + A[:,2:4]@B[2:4,0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)  # A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_block(0, 2, 2, 0, 0, 0)  # A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2]
    
    # C[:,2:4] = A[:,0:2]@B[0:2,2:4] + A[:,2:4]@B[2:4,2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2)  # A[0:2,0:2] @ B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2)  # A[0:2,2:4] @ B[2:4,2:4] -> C[0:2,2:4]
    
    # C[:,4] = A[:,0:2]@B[0:2,4] + A[:,2:4]@B[2:4,4]
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4)  # A[0:2,0:2] @ B[0:2,4] -> C[0:2,4]
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4)  # A[0:2,2:4] @ B[2:4,4] -> C[0:2,4]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

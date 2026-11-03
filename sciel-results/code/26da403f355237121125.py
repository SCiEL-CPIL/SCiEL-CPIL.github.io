import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    timeout = 120.0
    
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
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
        
    def add_strassen_2x2(a_i0, a_j0, b_i0, b_j0, c_i0, c_j0):
        """
        Adds Strassen decomposition for 2x2*2x2 block.
        A block at rows a_i0..a_i0+1, cols a_j0..a_j0+1
        B block at rows b_i0..b_i0+1, cols b_j0..b_j0+1
        C block at rows c_i0..c_i0+1, cols c_j0..c_j0+1
        """
        def ma(loc):
            r, c = loc // 2, loc % 2
            return (a_i0 + r) * m + (a_j0 + c)
        def mb(loc):
            r, c = loc // 2, loc % 2
            return (b_i0 + r) * p + (b_j0 + c)
        def mc(loc):
            r, c = loc // 2, loc % 2
            return (c_j0 + c) * n + (c_i0 + r)
        
        # Strassen 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        add_rank1_term({ma(0): 1, ma(3): 1}, {mb(0): 1, mb(3): 1}, {mc(0): 1, mc(3): 1})
        # M2 = (A10+A11)B00
        add_rank1_term({ma(2): 1, ma(3): 1}, {mb(0): 1}, {mc(2): 1, mc(3): -1})
        # M3 = A00(B01-B11)
        add_rank1_term({ma(0): 1}, {mb(1): 1, mb(3): -1}, {mc(1): 1, mc(3): 1})
        # M4 = A11(B10-B00)
        add_rank1_term({ma(3): 1}, {mb(2): 1, mb(0): -1}, {mc(0): 1, mc(2): 1})
        # M5 = (A00+A01)B11
        add_rank1_term({ma(0): 1, ma(1): 1}, {mb(3): 1}, {mc(0): -1, mc(1): 1})
        # M6 = (A10-A00)(B00+B01)
        add_rank1_term({ma(2): 1, ma(0): -1}, {mb(0): 1, mb(1): 1}, {mc(3): 1})
        # M7 = (A01-A11)(B10+B11)
        add_rank1_term({ma(1): 1, ma(3): -1}, {mb(2): 1, mb(3): 1}, {mc(0): 1})
        
    def add_schoolbook_2x2_1(a_i0, a_j0, b_i0, b_j0, c_i0, c_j0):
        """
        Adds schoolbook decomposition for 2x2*2x1 block.
        A is 2x2, B is 2x1, C is 2x1.
        """
        def ma(loc):
            r, c = loc // 2, loc % 2
            return (a_i0 + r) * m + (a_j0 + c)
        def mb(loc):
            return (b_i0 + loc) * p + b_j0
        def mc(loc):
            return c_j0 * n + (c_i0 + loc)
        
        # C[0] = A[0][0]*B[0] + A[0][1]*B[1]
        # C[1] = A[1][0]*B[0] + A[1][1]*B[1]
        add_rank1_term({ma(0): 1}, {mb(0): 1}, {mc(0): 1})
        add_rank1_term({ma(1): 1}, {mb(1): 1}, {mc(0): 1})
        add_rank1_term({ma(2): 1}, {mb(0): 1}, {mc(1): 1})
        add_rank1_term({ma(3): 1}, {mb(1): 1}, {mc(1): 1})
        
    # Block 1: A[:, 0:2] * B[0:2, :]
    # A[0:2, 0:2] is 2x2. B[0:2, 0:5] is 2x5.
    # Split B into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
    add_strassen_2x2(0, 0, 0, 0, 0, 0)   # A[0:2,0:2] * B[0:2,0:2]
    add_strassen_2x2(0, 0, 0, 2, 0, 2)   # A[0:2,0:2] * B[0:2,2:4]
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)  # A[0:2,0:2] * B[0:2,4:5]
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    # A[0:2, 2:4] is 2x2. B[2:4, 0:5] is 2x5.
    add_strassen_2x2(0, 2, 2, 0, 0, 0)   # A[0:2,2:4] * B[2:4,0:2]
    add_strassen_2x2(0, 2, 2, 2, 0, 2)   # A[0:2,2:4] * B[2:4,2:4]
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)  # A[0:2,2:4] * B[2:4,4:5]
    
    # Verify no time spent, return immediately
    return {"U": U, "V": V, "W": W}

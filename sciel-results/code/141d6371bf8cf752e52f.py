import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 145.0

    n, m, p = 2, 4, 7
    rank = 50

    U = [[0] * rank for _ in range(n * m)]  # shape (8, 50)
    V = [[0] * rank for _ in range(m * p)]  # shape (28, 50)
    W = [[0] * rank for _ in range(p * n)]  # shape (14, 50)

    col = [0]  # use list for mutable closure

    def _time_left():
        return _deadline - time.time()

    def add_strassen_block(A_idx, B_idx, C_idx):
        """
        Strassen 2x2x2: A is 2x2, B is 2x2, C is 2x2
        A_idx: list of 4 tuples [(i,j),...] for A[0,0],A[0,1],A[1,0],A[1,1]
        B_idx: list of 4 tuples for B[0,0],B[0,1],B[1,0],B[1,1]
        C_idx: list of 4 tuples for C[0,0],C[0,1],C[1,0],C[1,1]
        """
        def pm(a_coeffs, b_coeffs, c_coeffs):
            nonlocal col
            for la, cf in a_coeffs:
                ga = A_idx[la]
                U[ga[0] * m + ga[1]][col] += cf
            for lb, cf in b_coeffs:
                gb = B_idx[lb]
                V[gb[0] * p + gb[1]][col] += cf
            for lc, cf in c_coeffs:
                gc = C_idx[lc]
                W[gc[1] * n + gc[0]][col] += cf
            col += 1

        # M1 = (a00+a11)(b00+b11) -> C: c00+=1, c11+=1... 
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00)
        # M3 = (a00)(b01-b11)
        # M4 = (a11)(b10-b00)
        # M5 = (a00+a01)(b11)
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)

        if _time_left() < 5:
            return
        # M1: A: (0,1),(3,1) B: (0,1),(3,1) C: (0,1),(3,1)
        pm([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)])
        # M2: A: (2,1),(3,1) B: (0,1) C: (2,1),(3,-1)
        pm([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)])
        # M3: A: (0,1) B: (1,1),(3,-1) C: (1,1),(3,1)
        pm([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)])
        # M4: A: (3,1) B: (2,1),(0,-1) C: (0,1),(2,1)
        pm([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)])
        # M5: A: (0,1),(1,1) B: (3,1) C: (0,-1),(1,1)
        pm([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)])
        # M6: A: (2,1),(0,-1) B: (0,1),(1,1) C: (3,1)
        pm([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)])
        # M7: A: (1,1),(3,-1) B: (2,1),(3,1) C: (0,1)
        pm([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)])

    def add_schoolbook_221(A_idx, B_idx, C_idx):
        """
        Schoolbook 2x2x1: A is 2x2, B is 2x1, C is 2x1
        A_idx: 4 tuples for A[0,0],A[0,1],A[1,0],A[1,1]
        B_idx: 2 tuples for B[0,0],B[1,0]
        C_idx: 2 tuples for C[0,0],C[1,0]
        """
        def pm(a_i, b_i, c_i, cf=1):
            nonlocal col
            ga = A_idx[a_i]
            U[ga[0] * m + ga[1]][col] += cf
            gb = B_idx[b_i]
            V[gb[0] * p + gb[1]][col] += cf
            gc = C_idx[c_i]
            W[gc[1] * n + gc[0]][col] += cf
            col += 1

        if _time_left() < 5:
            return
        pm(0, 0, 0, 1)  # a00*b00 -> c00
        pm(1, 1, 0, 1)  # a01*b10 -> c00
        pm(2, 0, 1, 1)  # a10*b00 -> c10
        pm(3, 1, 1, 1)  # a11*b10 -> c10

    # A indices
    A_L = [(0,0),(0,1),(1,0),(1,1)]
    A_R = [(0,2),(0,3),(1,2),(1,3)]

    # Block 1: B cols 0,1 (4x2)
    B1_L = [(0,0),(0,1),(1,0),(1,1)]
    B1_R = [(2,0),(2,1),(3,0),(3,1)]
    C1   = [(0,0),(0,1),(1,0),(1,1)]

    if _time_left() > 5:
        add_strassen_block(A_L, B1_L, C1)
    if _time_left() > 5:
        add_strassen_block(A_R, B1_R, C1)

    # Block 2: B cols 2,3 (4x2)
    B2_L = [(0,2),(0,3),(1,2),(1,3)]
    B2_R = [(2,2),(2,3),(3,2),(3,3)]
    C2   = [(0,2),(0,3),(1,2),(1,3)]

    if _time_left() > 5:
        add_strassen_block(A_L, B2_L, C2)
    if _time_left() > 5:
        add_strassen_block(A_R, B2_R, C2)

    # Block 3: B cols 4,5,6 (4x3)
    # 3a: cols 4,5 (4x2)
    B3a_L = [(0,4),(0,5),(1,4),(1,5)]
    B3a_R = [(2,4),(2,5),(3,4),(3,5)]
    C3a   = [(0,4),(0,5),(1,4),(1,5)]

    if _time_left() > 5:
        add_strassen_block(A_L, B3a_L, C3a)
    if _time_left() > 5:
        add_strassen_block(A_R, B3a_R, C3a)

    # 3b: col 6 (4x1)
    B3b_L = [(0,6),(1,6)]
    B3b_R = [(2,6),(3,6)]
    C3b   = [(0,6),(1,6)]

    if _time_left() > 5:
        add_schoolbook_221(A_L, B3b_L, C3b)
    if _time_left() > 5:
        add_schoolbook_221(A_R, B3b_R, C3b)

    return {"U": U, "V": V, "W": W}

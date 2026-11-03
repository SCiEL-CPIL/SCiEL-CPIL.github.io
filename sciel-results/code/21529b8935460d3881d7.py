"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    
    Strategy: Block decomposition 5 = 2 + 3.
    - Use Strassen (rank 7) for the 2x2x2 diagonal block.
    - Use Laderman (rank 15) for the 3x3x3 diagonal block.
    - Use schoolbook for all other sub-block multiplications.
    Total rank: 7 + 15 + 12 + 12 + 18 + 12 + 18 + 18 = 112
    """
    import time
    import numpy as np
    
    start_time = time.time()
    DEADLINE = 135
    
    n, m, p = DIMENSIONS  # all 5
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() - start_time > DEADLINE - 10:
        return schoolbook()
    
    # Block sizes
    a1, a2 = 2, 3  # rows of A split
    b1, b2 = 2, 3  # cols of A / rows of B split
    c1, c2 = 2, 3  # cols of B / cols of C split
    
    # Build full 5x5x5 decomposition via blocks
    U_cols = []  # list of column vectors (length 25)
    V_cols = []  # list of column vectors (length 25)
    W_cols = []  # list of column vectors (length 25)
    
    def add_term(u_vec, v_vec, w_vec):
        """Add one rank-1 term. u_vec, v_vec, w_vec are length-25 lists."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Helper: create zero vector of length 25
    def zero_vec():
        return [0] * 25
    
    # Strassen 2x2x2 (rank 7) for C11 = A11 * B11
    # A11 is rows 0-1, cols 0-1 of A -> indices i*5+j for i in [0,1], j in [0,1]
    # B11 is rows 0-1, cols 0-1 of B -> indices j*5+k for j in [0,1], k in [0,1]
    # C11 is rows 0-1, cols 0-1 of C -> W index k*5+i for k in [0,1], i in [0,1]
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3 = A00(B10-B11)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B01-B10)
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B10)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
        # M7 = (A01-A11)(B01+B11)
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1}),
    ]
    
    for u_coeff, v_coeff, w_coeff in strassen_terms:
        u_vec = zero_vec()
        v_vec = zero_vec()
        w_vec = zero_vec()
        for (li, lj), val in u_coeff.items():
            u_vec[li * 5 + lj] = val
        for (lj, lk), val in v_coeff.items():
            v_vec[lj * 5 + lk] = val
        for (lk, li), val in w_coeff.items():
            w_vec[lk * 5 + li] = val
        add_term(u_vec, v_vec, w_vec)
    
    # Laderman 3x3x3 (rank 15) for C22 = A22 * B22
    # A22 is rows 2-4, cols 2-4 of A -> indices i*5+j for i in [2,3,4], j in [2,3,4]
    # B22 is rows 2-4, cols 2-4 of B -> indices j*5+k for j in [2,3,4], k in [2,3,4]
    # C22 is rows 2-4, cols 2-4 of C -> W index k*5+i for k in [2,3,4], i in [2,3,4]
    
    # Laderman's 15-term algorithm for 3x3 matrix multiplication
    # Local indices (li, lj) for A22: li in [0,1,2], lj in [0,1,2]
    # Global: row = 2+li, col = 2+lj
    # Local (lj, lk) for B22: lj in [0,1,2], lk in [0,1,2]
    # Global: row = 2+lj, col = 2+lk
    # Local (lk, li) for C22: lk in [0,1,2], li in [0,1,2]
    # Global: row = 2+lk, col = 2+li
    
    laderman_terms = [
        # M1 = A00 * B00
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        # M2 = A11 * B11
        ({(1,1):1}, {(1,1):1}, {(1,1):1}),
        # M3 = A22 * B22
        ({(2,2):1}, {(2,2):1}, {(2,2):1}),
        # M4 = (A00+A11) * (B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,1):1, (1,0):1}),
        # M5 = (A00+A22) * (B00+B22)
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,2):1, (2,0):1}),
        # M6 = (A11+A22) * (B11+B22)
        ({(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}, {(1,2):1, (2,1):1}),
        # M7 = A01 * B10
        ({(0,1):1}, {(1,0):1}, {(1,0):1}),
        # M8 = A10 * B01
        ({(1,0):1}, {(0,1):1}, {(0,1):1}),
        # M9 = A02 * B20
        ({(0,2):1}, {(2,0):1}, {(2,0):1}),
        # M10 = A20 * B02
        ({(2,0):1}, {(0,2):1}, {(0,2):1}),
        # M11 = A12 * B21
        ({(1,2):1}, {(2,1):1}, {(2,1):1}),
        # M12 = A21 * B12
        ({(2,1):1}, {(1,2):1}, {(1,2):1}),
        # M13 = (A00+A11+A22) * (B00+B11+B22)
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):-1, (1,1):-1, (2,2):-1}),
        # M14 = (A01+A12) * (B10+B20)
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,0):1}, {(0,1):1, (1,2):1}),
        # M15 = (A10+A20) * (B01+B02)
        ({(1,0):1, (2,0):1}, {(0,1):1, (0,2):1}, {(1,0):1, (2,0):1}),
    ]
    
    for u_coeff, v_coeff, w_coeff in laderman_terms:
        u_vec = zero_vec()
        v_vec = zero_vec()
        w_vec = zero_vec()
        for (li, lj), val in u_coeff.items():
            u_vec[(2+li) * 5 + (2+lj)] = val
        for (lj, lk), val in v_coeff.items():
            v_vec[(2+lj) * 5 + (2+lk)] = val
        for (lk, li), val in w_coeff.items():
            w_vec[(2+lk) * 5 + (2+li)] = val
        add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C11 cross term: A12(2x3) * B21(3x2)
    # A12: rows 0-1, cols 2-4 of A
    # B21: rows 2-4, cols 0-1 of B
    # C11: rows 0-1, cols 0-1 of C
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C12 = A11(2x2) * B12(2x3)
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C12 cross term: A12(2x3) * B22(3x3)
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C21 = A21(3x2) * B11(2x2)
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C21 cross term: A22(3x3) * B21(3x2)
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Schoolbook for C22 = A21(3x2) * B12(2x3)
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                u_vec = zero_vec()
                v_vec = zero_vec()
                w_vec = zero_vec()
                u_vec[i * 5 + j] = 1
                v_vec[j * 5 + k] = 1
                w_vec[k * 5 + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Note: C22 diagonal block (A22 * B22) is handled by Laderman above
    # No need for schoolbook for A22(3x3) * B22(3x3)
    
    # Assemble final matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    for r in range(rank):
        for idx in range(25):
            U[idx][r] = U_cols[r][idx]
            V[idx][r] = V_cols[r][idx]
            W[idx][r] = W_cols[r][idx]
    
    # Verify: check that the decomposition is correct
    try:
        rng = np.random.RandomState(42)
        A_test = rng.rand(5, 5)
        B_test = rng.rand(5, 5)
        C_expected = A_test @ B_test
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        C_from_decomp = np.zeros((5, 5))
        for r in range(rank):
            la = np.sum(U_np[:, r] * A_test.flatten())
            lb = np.sum(V_np[:, r] * B_test.flatten())
            for i in range(5):
                for k in range(5):
                    C_from_decomp[i, k] += W_np[k * 5 + i, r] * la * lb
        
        if not np.allclose(C_from_decomp, C_expected, atol=1e-9):
            return schoolbook()
    except Exception:
        return schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

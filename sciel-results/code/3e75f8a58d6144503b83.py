import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 125

    n, m, p = DIMENSIONS
    try:
        U, V, W = _strassen_4x4(n, m, p)
        _verify(U, V, W, n, m, p, deadline)
        return {"U": U, "V": V, "W": W}
    except Exception as e:
        # Fallback: rank-64 trivial decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, deadline):
    """Verify the decomposition with multiple random tests."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for seed in [42, 123, 999]:
        if time.time() > deadline - 10:
            break
        rng = np.random.RandomState(seed)
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        
        for r in range(rank):
            if time.time() > deadline - 5:
                break
            u_col = U_arr[:, r].reshape(n, m)
            v_col = V_arr[:, r].reshape(m, p)
            
            uA = u_col * A
            vB = v_col * B
            
            term = uA @ vB
            
            w_mat = W_arr[:, r].reshape(p, n)
            C_computed += term * w_mat.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            diff = np.max(np.abs(C_computed - C_expected))
            raise ValueError(f"Decomposition verification failed for seed {seed}, max diff: {diff}")

def _strassen_4x4(n, m, p):
    """
    Recursive Strassen: 4x4 = 2x2 blocks, each 2x2 block mult uses 7 products.
    Total rank = 7 (outer Strassen) * 7 (inner Strassen) = 49
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 formula:
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1 = (A10+A11)B00       -> C10 += M1, C11 -= M1
    # M2 = A00(B01-B11)       -> C01 += M2, C11 += M2
    # M3 = A11(B10-B00)       -> C00 += M3, C10 += M3
    # M4 = (A00+A01)B11       -> C01 += M4, C00 -= M4
    # M5 = (A10-A00)(B00+B01) -> C11 += M5
    # M6 = (A01-A11)(B10+B11) -> C00 += M6

    # For each outer Strassen product s (0..6), we need inner Strassen products t (0..6)
    # The inner product t computes a 2x2 block multiplication using 7 scalar mults
    # Each inner product has specific U, V, W patterns

    # Inner Strassen 2x2 decomposition (rank 7):
    # For multiplying 2x2 X * Y = Z:
    # X rows: 0,1 ; cols: 0,1
    # Y rows: 0,1 ; cols: 0,1
    # Z rows: 0,1 ; cols: 0,1

    inner_patterns = []
    # Each pattern: (u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, w_sign)
    # where u_rows/cols are which X entries are summed (with signs),
    # v_rows/cols are which Y entries are summed (with signs),
    # w_rows/cols is which Z entry this contributes to, w_sign is the coefficient

    # M0: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
    # M1: (X10+X11)Y00       -> Z10 += 1, Z11 -= 1
    # M2: X00(Y01-Y11)       -> Z01 += 1, Z11 += 1
    # M3: X11(Y10-Y00)       -> Z00 += 1, Z10 += 1
    # M4: (X00+X01)Y11       -> Z01 += 1, Z00 -= 1
    # M5: (X10-X00)(Y00+Y01) -> Z11 += 1
    # M6: (X01-X11)(Y10+Y11) -> Z00 += 1

    # For each inner product t, define:
    # u_terms: list of (row, col, sign) for X
    # v_terms: list of (row, col, sign) for Y
    # w_terms: list of (row, col, sign) for Z

    inner_defs = [
        # t=0: M0
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        # t=1: M1
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1),(1,1,-1)]},
        # t=2: M2
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(0,1,1),(1,1,1)]},
        # t=3: M3
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,0,1)]},
        # t=4: M4
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1),(0,0,-1)]},
        # t=5: M5
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        # t=6: M6
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1)]},
    ]

    # Outer Strassen for 4x4 (treating as 2x2 blocks of size 2x2):
    # Block indices: (P,Q) where P,Q in {0,1}
    # A[P][Q] is a 2x2 block, B[P][Q] is a 2x2 block
    # C[P][Q] is a 2x2 block

    # Outer M0: (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # Outer M1: (A10+A11)B00       -> C10 += M1, C11 -= M1
    # Outer M2: A00(B01-B11)       -> C01 += M2, C11 += M2
    # Outer M3: A11(B10-B00)       -> C00 += M3, C10 += M3
    # Outer M4: (A00+A01)B11       -> C01 += M4, C00 -= M4
    # Outer M5: (A10-A00)(B00+B01) -> C11 += M5
    # Outer M6: (A01-A11)(B10+B11) -> C00 += M6

    outer_defs = [
        # s=0: M0
        {'a_blocks': [(0,0,1),(1,1,1)], 'b_blocks': [(0,0,1),(1,1,1)], 'c_blocks': [(0,0,1),(1,1,1)]},
        # s=1: M1
        {'a_blocks': [(1,0,1),(1,1,1)], 'b_blocks': [(0,0,1)], 'c_blocks': [(1,0,1),(1,1,-1)]},
        # s=2: M2
        {'a_blocks': [(0,0,1)], 'b_blocks': [(0,1,1),(1,1,-1)], 'c_blocks': [(0,1,1),(1,1,1)]},
        # s=3: M3
        {'a_blocks': [(1,1,1)], 'b_blocks': [(1,0,1),(0,0,-1)], 'c_blocks': [(0,0,1),(1,0,1)]},
        # s=4: M4
        {'a_blocks': [(0,0,1),(0,1,1)], 'b_blocks': [(1,1,1)], 'c_blocks': [(0,1,1),(0,0,-1)]},
        # s=5: M5
        {'a_blocks': [(1,0,1),(0,0,-1)], 'b_blocks': [(0,0,1),(0,1,1)], 'c_blocks': [(1,1,1)]},
        # s=6: M6
        {'a_blocks': [(0,1,1),(1,1,-1)], 'b_blocks': [(1,0,1),(1,1,1)], 'c_blocks': [(0,0,1)]},
    ]

    # For each outer product s and inner product t, we create one rank-1 term
    # The U matrix entry: for each A block in outer_a_blocks and each X entry in inner_u
    #   A block (P,Q) has elements at A[2P+ir][2Q+jc] for ir,jc in {0,1}
    #   X entry (ir,jc) with sign_su
    #   So A element at row=2P+ir, col=2Q+jc gets contribution sign_su * sign_au
    #   U index = row * m + col = (2P+ir)*4 + (2Q+jc)

    # Similarly for V: B block (P,Q) has elements at B[2P+ir][2Q+jc]
    #   Y entry (ir,jc) with sign_sv
    #   V index = row * p + col = (2P+ir)*4 + (2Q+jc)

    # For W: C block (P,Q) has elements at C[2P+ir][2Q+jc]
    #   Z entry (ir,jc) with sign_sw
    #   C element at row=2P+ir, col=2Q+jc
    #   W index = col * n + row = (2Q+jc)*4 + (2P+ir)

    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t

            # Build U contributions
            for (P, Q, sign_a) in outer['a_blocks']:
                for (ir, jc, sign_x) in inner['u']:
                    coeff = sign_a * sign_x
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        u_idx = row * m + col
                        U[u_idx][r] += coeff

            # Build V contributions
            for (P, Q, sign_b) in outer['b_blocks']:
                for (ir, jc, sign_y) in inner['v']:
                    coeff = sign_b * sign_y
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        v_idx = row * p + col
                        V[v_idx][r] += coeff

            # Build W contributions
            for (P, Q, sign_c) in outer['c_blocks']:
                for (ir, jc, sign_z) in inner['w']:
                    coeff = sign_c * sign_z
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        w_idx = col * n + row
                        W[w_idx][r] += coeff

    return U, V, W



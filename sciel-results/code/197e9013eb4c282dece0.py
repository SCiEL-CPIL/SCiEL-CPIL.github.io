import time
import numpy as np

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Decomposition:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    - C1 = A * B1 using recursive Strassen (rank 49).
    - C2 = A * B2 using recursive Strassen (rank 49).
    - C3 = A * B3 using schoolbook (rank 16).
    - Total rank: 114.
    
    Interface:
    U: (16, rank) where U[i*4+j, r] is coeff for A[i,j]
    V: (36, rank) where V[j*9+k, r] is coeff for B[j,k]
    W: (36, rank) where W[k*4+i, r] is coeff for C[i,k]
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = 4, 4, 9
    # U dim: n*m = 16
    # V dim: m*p = 36
    # W dim: p*n = 36
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Strassen 2x2x2: 7 terms
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # Keys are (row, col) in 0..1
    s2_terms = [
        # 1. (A00+A11)(B00+B11) -> C00 +, C11 +
        (({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})),
        # 2. (A10+A11)B00 -> C10 +, C11 -
        (({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})),
        # 3. A00(B01-B11) -> C01 +, C11 +
        (({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1})),
        # 4. A11(B10-B00) -> C00 +, C10 +
        (({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1})),
        # 5. (A00+A01)B11 -> C00 -, C01 +
        (({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})),
        # 6. (A10-A00)(B00+B01) -> C11 +
        (({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})),
        # 7. (A01-A11)(B10+B11) -> C00 +
        (({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})),
    ]

    def add_strassen_4x4(b_col_off):
        """
        Add 49 terms for A(4x4) * B_block(4x4) -> C_block(4x4).
        b_col_off: starting column index in B (0 or 4).
        C columns will be at the same offset.
        """
        for oa, ob, oc in s2_terms:
            for ia, ib, ic in s2_terms:
                u = np.zeros(16, dtype=np.float64)
                v = np.zeros(36, dtype=np.float64)
                w = np.zeros(36, dtype=np.float64)
                
                # U vector: A indices
                for (oor, ooc), c_ao in oa.items():
                    for (ior, ioc), c_ia in ia.items():
                        r = oor * 2 + ior
                        c = ooc * 2 + ioc
                        u[r * 4 + c] += c_ao * c_ia
                
                # V vector: B indices
                for (oor, ooc), c_bo in ob.items():
                    for (ior, ioc), c_ib in ib.items():
                        r = oor * 2 + ior
                        c = b_col_off + ooc * 2 + ioc
                        v[r * 9 + c] += c_bo * c_ib
                
                # W vector: C indices
                for (oor, ooc), c_co in oc.items():
                    for (ior, ioc), c_ic in ic.items():
                        r = oor * 2 + ior
                        c = b_col_off + ooc * 2 + ioc
                        w[c * 4 + r] += c_co * c_ic

                add_term(u.tolist(), v.tolist(), w.tolist())

    def add_schoolbook_4x1(b_col):
        """
        Add 16 terms for A(4x4) * B_col(4x1) -> C_col(4x1).
        b_col: column index in B (8).
        """
        for i in range(4):
            for j in range(4):
                u = np.zeros(16, dtype=np.float64)
                v = np.zeros(36, dtype=np.float64)
                w = np.zeros(36, dtype=np.float64)
                
                u[i * 4 + j] = 1.0
                v[j * 9 + b_col] = 1.0
                w[b_col * 4 + i] = 1.0
                
                add_term(u.tolist(), v.tolist(), w.tolist())

    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook 144 if somehow we run out of time (should not happen)
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    u = [0.0] * 16
                    v = [0.0] * 36
                    w = [0.0] * 36
                    u[i * 4 + j] = 1.0
                    v[j * 9 + k] = 1.0
                    w[k * 4 + i] = 1.0
                    add_term(u, v, w)
    else:
        # Add Strassen for B cols 0-3
        add_strassen_4x4(0)
        
        # Add Strassen for B cols 4-7
        add_strassen_4x4(4)
        
        # Add Schoolbook for B col 8
        add_schoolbook_4x1(8)

    rank = len(U_cols)
    
    # Build output matrices
    # U: 16 x rank
    U = [[U_cols[r][i] for r in range(rank)] for i in range(16)]
    # V: 36 x rank
    V = [[V_cols[r][i] for r in range(rank)] for i in range(36)]
    # W: 36 x rank
    W = [[W_cols[r][i] for r in range(rank)] for i in range(36)]
    
    return {"U": U, "V": V, "W": W}

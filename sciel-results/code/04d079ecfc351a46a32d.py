import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2 Strassen)
    # Block 2: A[0:2, 2:3] * B[2:3, 0:2] (2x1x2 Schoolbook)
    # Block 3: A[0:2, 2:3] * B[2:3, 2:5] (2x1x3 Schoolbook)
    # Block 4: A[0:2, 0:2] * B[0:2, 2:5] (2x2x3 Strassen+Schoolbook)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(u_rows, v_rows, w_rows, u_cols, v_cols, w_cols, u_vals, v_vals, w_vals):
        # u_rows: list of global row indices for U
        # v_rows: list of global row indices for V
        # w_rows: list of global row indices for W
        # u_cols, v_cols, w_cols: not used directly, we just append to lists
        # u_vals, v_vals, w_vals: lists of values for this rank-1 term
        pass

    # It's easier to just accumulate the full matrices.
    # Let's estimate the total rank first to allocate memory.
    rank1 = 7
    rank2 = 4
    rank3 = 6
    rank4 = 11
    total_rank = rank1 + rank2 + rank3 + rank4
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Block 1: 2x2x2 Strassen
    # A indices: i in 0,1; j in 0,1
    # B indices: j in 0,1; k in 0,1
    # C indices: i in 0,1; k in 0,1
    # Strassen terms:
    # m1 = (a+b)(e+h) -> U: a,b; V: e,h; W: C00, C01, C10, C11
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], h=B[1,0], i=B[1,1]
    # Global indices:
    # A[i, j] -> U[i*m + j]
    # B[j, k] -> V[j*p + k]
    # C[i, k] -> W[k*n + i]
    
    # m1
    add_term(0*m+0, 0*p+0, 0*n+0, 1, 1, 1) # a*e -> C00
    add_term(0*m+0, 0*p+1, 1*n+0, 1, 1, 1) # a*f -> C10? No, C[i,k]. C[0,1] is k=1, i=0. W[1*n+0]
    # Wait, Strassen formula:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Let's just use the standard Strassen construction for 2x2x2.
    # It's easier to hardcode the 7 terms.
    
    # Term 1: (A00+A01)(B00+B10)
    # A00+A01: U[0], U[1]
    # B00+B10: V[0], V[3] (B[0,0], B[1,0])
    # C00 += 1, C10 += 1, C01 += -1, C11 += 1?
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Mapping to 0-indexed:
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
    
    # M1
    add_term(0*m+0, 0*p+0, 0*n+0, 1, 1, 1) # A00 B00 -> C00
    add_term(0*m+0, 1*p+0, 0*n+0, 1, 1, 1) # A00 B10 -> C00
    add_term(0*m+1, 0*p+0, 0*n+0, 1, 1, 1) # A01 B00 -> C00
    add_term(0*m+1, 1*p+0, 0*n+0, 1, 1, 1) # A01 B10 -> C00
    # This is expanding the product. It's better to keep the sum in U and V.
    # U col: A00=1, A01=1
    # V col: B00=1, B10=1
    # W col: C00=1, C10=1, C01=-1, C11=1? No, M1 contributes to C00, C10, C01, C11.
    # C00 += M1
    # C10 += M1
    # C01 += -M1 ? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes:
    # W[0*n+0] += 1 (C00)
    # W[1*n+0] += 1 (C10)
    # W[0*n+1] += 0 (C01)
    # W[1*n+1] += 1 (C11)
    
    # Let's implement a helper to add a Strassen term.
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {global_u_idx: val}
        # v_coeffs: dict {global_v_idx: val}
        # w_coeffs: dict {global_w_idx: val}
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    # M1: (A00+A01)(B00+B10)
    add_strassen_term(
        {0*m+0: 1, 0*m+1: 1},
        {0*p+0: 1, 1*p+0: 1},
        {0*n+0: 1, 1*n+0: 1, 1*n+1: 1} # C00, C10, C11
    )
    
    # M2: (A10+A11)B00
    add_strassen_term(
        {1*m+0: 1, 1*m+1: 1},
        {0*p+0: 1},
        {1*n+0: 1} # C10
    )
    
    # M3: A00(B10-B11)
    add_strassen_term(
        {0*m+0: 1},
        {1*p+0: 1, 1*p+1: -1},
        {0*n+1: 1, 1*n+1: 1} # C01, C11
    )
    
    # M4: A11(B01-B00)
    add_strassen_term(
        {1*m+1: 1},
        {0*p+1: 1, 0*p+0: -1},
        {0*n+0: 1, 1*n+0: 1} # C00, C10
    )
    
    # M5: (A00+A01)B11
    add_strassen_term(
        {0*m+0: 1, 0*m+1: 1},
        {1*p+1: 1},
        {0*n+0: -1, 0*n+1: 1} # C00, C01
    )
    
    # M6: (A10-A00)(B00+B01)
    add_strassen_term(
        {1*m+0: 1, 0*m+0: -1},
        {0*p+0: 1, 0*p+1: 1},
        {1*n+1: 1} # C11
    )
    
    # M7: (A01-A11)(B10+B11)
    add_strassen_term(
        {0*m+1: 1, 1*m+1: -1},
        {1*p+0: 1, 1*p+1: 1},
        {0*n+0: 1, 1*n+1: -1} # C00, C11
    )
    
    # Block 2: 2x1x2 Schoolbook
    # A[0:2, 2:3] -> U[0*m+2], U[1*m+2]
    # B[2:3, 0:2] -> V[2*p+0], V[2*p+1]
    # C[0:2, 0:2] -> W[0*n+0], W[1*n+0], W[0*n+1], W[1*n+1]
    
    # A02 B20 -> C00
    add_strassen_term({0*m+2: 1}, {2*p+0: 1}, {0*n+0: 1})
    # A02 B21 -> C01
    add_strassen_term({0*m+2: 1}, {2*p+1: 1}, {0*n+1: 1})
    # A12 B20 -> C10
    add_strassen_term({1*m+2: 1}, {2*p+0: 1}, {1*n+0: 1})
    # A12 B21 -> C11
    add_strassen_term({1*m+2: 1}, {2*p+1: 1}, {1*n+1: 1})
    
    # Block 3: 2x1x3 Schoolbook
    # A[0:2, 2:3] -> U[0*m+2], U[1*m+2]
    # B[2:3, 2:5] -> V[2*p+2], V[2*p+3], V[2*p+4]
    # C[0:2, 2:5] -> W[2*n+0], W[3*n+0], W[4*n+0], W[2*n+1], W[3*n+1], W[4*n+1]
    
    # A02 B22 -> C02
    add_strassen_term({0*m+2: 1}, {2*p+2: 1}, {2*n+0: 1})
    # A02 B23 -> C03
    add_strassen_term({0*m+2: 1}, {2*p+3: 1}, {3*n+0: 1})
    # A02 B24 -> C04
    add_strassen_term({0*m+2: 1}, {2*p+4: 1}, {4*n+0: 1})
    # A12 B22 -> C12
    add_strassen_term({1*m+2: 1}, {2*p+2: 1}, {2*n+1: 1})
    # A12 B23 -> C13
    add_strassen_term({1*m+2: 1}, {2*p+3: 1}, {3*n+1: 1})
    # A12 B24 -> C14
    add_strassen_term({1*m+2: 1}, {2*p+4: 1}, {4*n+1: 1})
    
    # Block 4: 2x2x3 (Strassen + Schoolbook)
    # A[0:2, 0:2]
    # B[0:2, 2:5]
    # C[0:2, 2:5]
    
    # Part 4a: 2x2x2 (B[0:2, 2:4])
    # B indices: j in 0,1; k in 2,3
    # B02=B[0,2], B03=B[0,3], B12=B[1,2], B13=B[1,3]
    # C02=C[0,2], C03=C[0,3], C12=C[1,2], C13=C[1,3]
    
    # M1: (A00+A01)(B02+B12)
    add_strassen_term(
        {0*m+0: 1, 0*m+1: 1},
        {0*p+2: 1, 1*p+2: 1},
        {2*n+0: 1, 2*n+1: 1, 3*n+1: 1} # C02, C12, C13
    )
    # M2: (A10+A11)B02
    add_strassen_term(
        {1*m+0: 1, 1*m+1: 1},
        {0*p+2: 1},
        {2*n+1: 1} # C12
    )
    # M3: A00(B12-B13)
    add_strassen_term(
        {0*m+0: 1},
        {1*p+2: 1, 1*p+3: -1},
        {3*n+0: 1, 3*n+1: 1} # C03, C13
    )
    # M4: A11(B03-B02)
    add_strassen_term(
        {1*m+1: 1},
        {0*p+3: 1, 0*p+2: -1},
        {2*n+0: 1, 2*n+1: 1} # C02, C12
    )
    # M5: (A00+A01)B13
    add_strassen_term(
        {0*m+0: 1, 0*m+1: 1},
        {1*p+3: 1},
        {2*n+0: -1, 3*n+0: 1} # C02, C03
    )
    # M6: (A10-A00)(B02+B03)
    add_strassen_term(
        {1*m+0: 1, 0*m+0: -1},
        {0*p+2: 1, 0*p+3: 1},
        {3*n+1: 1} # C13
    )
    # M7: (A01-A11)(B12+B13)
    add_strassen_term(
        {0*m+1: 1, 1*m+1: -1},
        {1*p+2: 1, 1*p+3: 1},
        {2*n+0: 1, 3*n+1: -1} # C02, C13
    )
    
    # Part 4b: 2x2x1 (B[0:2, 4:5])
    # B04=B[0,4], B14=B[1,4]
    # C04=C[0,4], C14=C[1,4]
    # Schoolbook
    # A00 B04 -> C04
    add_strassen_term({0*m+0: 1}, {0*p+4: 1}, {4*n+0: 1})
    # A01 B14 -> C04
    add_strassen_term({0*m+1: 1}, {1*p+4: 1}, {4*n+0: 1})
    # A10 B04 -> C14
    add_strassen_term({1*m+0: 1}, {0*p+4: 1}, {4*n+1: 1})
    # A11 B14 -> C14
    add_strassen_term({1*m+1: 1}, {1*p+4: 1}, {4*n+1: 1})
    
    return {"U": U, "V": V, "W": W}

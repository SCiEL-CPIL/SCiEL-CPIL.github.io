import time

def construct():
    n, m, p = 3, 4, 5
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a column
    def add_col(u_indices, v_indices, w_indices):
        nonlocal col
        for i, idx in u_indices.items():
            U[idx][col] += i
        for j, idx in v_indices.items():
            V[idx][col] += j
        for k, idx in w_indices.items():
            W[idx][col] += k
        col += 1

    # Strassen 2x2x2
    def emit_strassen(i0, j0, k0):
        # Local indices 0,1 for i, j, k
        # U rows: i0*2+0, i0*2+1, (i0+1)*2+0, (i0+1)*2+1
        # Wait, global U index is i*m + j.
        # i is in {i0, i0+1}, j is in {j0, j0+1}
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + j0 + 1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k0 + 1
        v10 = (j0 + 1) * p + k0
        v11 = (j0 + 1) * p + k0 + 1
        
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        w10 = k0 * n + i0 + 1
        w11 = (k0 + 1) * n + i0 + 1
        
        # M1: (a+d)(e+h)
        add_col({u00: 1, u11: 1}, {v00: 1, v11: 1}, {w00: 1, w11: 1})
        # M2: (c+d)e
        add_col({u10: 1, u11: 1}, {v00: 1}, {w10: 1, w11: -1})
        # M3: a(f-h)
        add_col({u00: 1}, {v01: 1, v11: -1}, {w01: 1, w11: 1})
        # M4: d(g-e)
        add_col({u11: 1}, {v10: 1, v00: -1}, {w00: 1, w10: 1})
        # M5: (a+b)h
        add_col({u00: 1, u01: 1}, {v11: 1}, {w00: -1, w01: 1})
        # M6: (c-a)(e+f)
        add_col({u10: 1, u00: -1}, {v00: 1, v01: 1}, {w11: 1})
        # M7: (b-d)(g+h)
        add_col({u01: 1, u11: -1}, {v10: 1, v11: 1}, {w00: 1})

    # Schoolbook 2x2x1
    def emit_schoolbook_221(i0, j0, k0):
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + j0 + 1
        
        v00 = j0 * p + k0
        v10 = (j0 + 1) * p + k0
        
        w00 = k0 * n + i0
        w10 = k0 * n + i0 + 1
        
        # C00 = A00 B00
        add_col({u00: 1}, {v00: 1}, {w00: 1})
        # C00 += A01 B10
        add_col({u01: 1}, {v10: 1}, {w00: 1})
        # C10 = A10 B00
        add_col({u10: 1}, {v00: 1}, {w10: 1})
        # C10 += A11 B10
        add_col({u11: 1}, {v10: 1}, {w10: 1})

    # Schoolbook 1x2x2
    def emit_schoolbook_122(i0, j0, k0):
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k0 + 1
        v10 = (j0 + 1) * p + k0
        v11 = (j0 + 1) * p + k0 + 1
        
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        
        # C00 = A00 B00
        add_col({u00: 1}, {v00: 1}, {w00: 1})
        # C00 += A01 B10
        add_col({u01: 1}, {v10: 1}, {w00: 1})
        # C01 = A00 B01
        add_col({u00: 1}, {v01: 1}, {w01: 1})
        # C01 += A01 B11
        add_col({u01: 1}, {v11: 1}, {w01: 1})

    # Schoolbook 1x2x1
    def emit_schoolbook_121(i0, j0, k0):
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        
        v00 = j0 * p + k0
        v10 = (j0 + 1) * p + k0
        
        w00 = k0 * n + i0
        
        # C00 = A00 B00
        add_col({u00: 1}, {v00: 1}, {w00: 1})
        # C00 += A01 B10
        add_col({u01: 1}, {v10: 1}, {w00: 1})

    # Block 1: j in {0, 1}
    # Sub-block 1.1: i in {0, 1}
    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_schoolbook_221(0, 0, 4)
    
    # Sub-block 1.2: i in {2}
    emit_schoolbook_122(2, 0, 0)
    emit_schoolbook_122(2, 0, 2)
    emit_schoolbook_121(2, 0, 4)

    # Block 2: j in {2, 3}
    # Sub-block 2.1: i in {0, 1}
    emit_strassen(0, 2, 0)
    emit_strassen(0, 2, 2)
    emit_schoolbook_221(0, 2, 4)
    
    # Sub-block 2.2: i in {2}
    emit_schoolbook_122(2, 2, 0)
    emit_schoolbook_122(2, 2, 2)
    emit_schoolbook_121(2, 2, 4)

    return {"U": U, "V": V, "W": W}

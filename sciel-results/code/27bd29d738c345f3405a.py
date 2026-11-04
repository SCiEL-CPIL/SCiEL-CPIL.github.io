import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 140.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def construct():
    n, m, p = 2, 5, 5
    # U: (n*m, rank) = (10, R)
    # V: (m*p, rank) = (25, R)
    # W: (p*n, rank) = (10, R)
    
    # We will build lists of terms (u, v, w)
    # u is a dict or list of non-zero entries for A
    # v is a dict or list of non-zero entries for B
    # w is a dict or list of non-zero entries for C
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
        
    # Helper to create Strassen decomposition for 2x2 * 2x2
    # A indices: (ai, aj), B indices: (bj, bk), C indices: (ci, ck)
    # We map local indices to global indices
    def add_strassen(a_off, b_off, c_off):
        # a_off: (i_start, j_start) for A (2x2)
        # b_off: (j_start, k_start) for B (2x2)
        # c_off: (i_start, k_start) for C (2x2)
        
        # Local indices 0,1
        # A[0,0] -> A[i0, j0]
        # A[0,1] -> A[i0, j1]
        # A[1,0] -> A[i1, j0]
        # A[1,1] -> A[i1, j1]
        
        i0, j0 = a_off
        k0, l0 = b_off # b is 2x2, so rows j, cols k. b_off is (row_start, col_start)
        ci0, ck0 = c_off
        
        # Strassen formulas
        # m0 = (a00 + a11)(b00 + b11)
        # m1 = (a10 + a11)b00
        # m2 = a00(b01 - b11)
        # m3 = a11(b10 - b00)
        # m4 = (a00 + a01)b11
        # m5 = (a10 - a00)(b00 + b01)
        # m6 = (a01 - a11)(b10 + b11)
        
        # m0
        add_term(
            {(i0, j0): 1, (i0+1, j0+1): 1},
            {(k0, l0): 1, (k0+1, l0+1): 1},
            {(ci0, ck0): 1, (ci0+1, ck0+1): 1}
        )
        # m1
        add_term(
            {(i0+1, j0): 1, (i0+1, j0+1): 1},
            {(k0, l0): 1},
            {(ci0+1, ck0): 1, (ci0, ck0+1): 1} # C10 = m1+m3, C01 = m2+m4. Wait.
            # C10 = a10b00 + a11b10.
            # m1 = (a10+a11)b00 = a10b00 + a11b00.
            # This term contributes to C10 and C01?
            # Let's re-verify Strassen output mapping.
            # C00 = m0 + m3 - m4 + m6
            # C01 = m2 + m4
            # C10 = m1 + m3
            # C11 = m0 - m1 + m2 + m5
            #
            # So for m1:
            # W entries: C10 gets 1. C11 gets -1.
            # V entries: b00 gets 1.
            # U entries: a10, a11 get 1.
        )
        # Correction: The add_term above was a placeholder. I need to be precise.
        # Let's rewrite the Strassen addition carefully.
        
        # m0
        add_term(
            {(i0, j0): 1, (i0+1, j0+1): 1},
            {(k0, l0): 1, (k0+1, l0+1): 1},
            {(ci0, ck0): 1, (ci0+1, ck0+1): 1}
        )
        # m1
        add_term(
            {(i0+1, j0): 1, (i0+1, j0+1): 1},
            {(k0, l0): 1},
            {(ci0+1, ck0): 1, (ci0+1, ck0+1): -1}
        )
        # m2
        add_term(
            {(i0, j0): 1},
            {(k0, l0+1): 1, (k0+1, l0+1): -1},
            {(ci0, ck0+1): 1, (ci0+1, ck0+1): 1}
        )
        # m3
        add_term(
            {(i0+1, j0+1): 1},
            {(k0+1, l0): 1, (k0, l0): -1},
            {(ci0, ck0): 1, (ci0+1, ck0): 1}
        )
        # m4
        add_term(
            {(i0, j0): 1, (i0, j0+1): 1},
            {(k0+1, l0+1): 1},
            {(ci0, ck0): -1, (ci0, ck0+1): 1}
        )
        # m5
        add_term(
            {(i0+1, j0): 1, (i0, j0): -1},
            {(k0, l0): 1, (k0, l0+1): 1},
            {(ci0+1, ck0+1): 1}
        )
        # m6
        add_term(
            {(i0, j0+1): 1, (i0+1, j0+1): -1},
            {(k0+1, l0): 1, (k0+1, l0+1): 1},
            {(ci0, ck0): 1}
        )

    # Helper to add schoolbook 2x2 * 2x1
    def add_2x2_2x1(a_off, b_off, c_off):
        # A (2x2) at (i0, j0)
        # B (2x1) at (k0, l0) -- l0 is col index, only 1 col
        # C (2x1) at (ci0, ck0)
        
        i0, j0 = a_off
        k0, l0 = b_off
        ci0, ck0 = c_off
        
        # C0 = a00b0 + a01b1
        # C1 = a10b0 + a11b1
        
        # Term 1: a00 b0
        add_term(
            {(i0, j0): 1},
            {(k0, l0): 1},
            {(ci0, ck0): 1}
        )
        # Term 2: a01 b1
        add_term(
            {(i0, j0+1): 1},
            {(k0+1, l0): 1},
            {(ci0, ck0): 1}
        )
        # Term 3: a10 b0
        add_term(
            {(i0+1, j0): 1},
            {(k0, l0): 1},
            {(ci0+1, ck0): 1}
        )
        # Term 4: a11 b1
        add_term(
            {(i0+1, j0+1): 1},
            {(k0+1, l0): 1},
            {(ci0+1, ck0): 1}
        )

    # Helper to add schoolbook 2x1 * 1x5
    def add_2x1_1x5(a_off, b_off, c_off):
        # A (2x1) at (i0, j0)
        # B (1x5) at (k0, l0) -- k0 is row index (only 1 row), l0 is col start
        # C (2x5) at (ci0, ck0)
        
        i0, j0 = a_off
        k0, l0 = b_off
        ci0, ck0 = c_off
        
        for k in range(5):
            # C0k = a0 b_k
            add_term(
                {(i0, j0): 1},
                {(k0, l0+k): 1},
                {(ci0, ck0+k): 1}
            )
            # C1k = a1 b_k
            add_term(
                {(i0+1, j0): 1},
                {(k0, l0+k): 1},
                {(ci0+1, ck0+k): 1}
            )

    # Part 1: j in {0, 1}
    # A cols 0,1. B rows 0,1.
    # C cols 0,1 (from B cols 0,1) -> Strassen
    add_strassen((0, 0), (0, 0), (0, 0))
    # C cols 2,3 (from B cols 2,3) -> Strassen
    add_strassen((0, 0), (0, 2), (0, 2))
    # C col 4 (from B col 4) -> 2x2 * 2x1
    add_2x2_2x1((0, 0), (0, 4), (0, 4))
    
    # Part 2: j in {2, 3}
    # A cols 2,3. B rows 2,3.
    # C cols 0,1 (from B cols 0,1) -> Strassen
    add_strassen((0, 2), (2, 0), (0, 0))
    # C cols 2,3 (from B cols 2,3) -> Strassen
    add_strassen((0, 2), (2, 2), (0, 2))
    # C col 4 (from B col 4) -> 2x2 * 2x1
    add_2x2_2x1((0, 2), (2, 4), (0, 4))
    
    # Part 3: j = 4
    # A col 4. B row 4.
    # C cols 0..4
    add_2x1_1x5((0, 4), (4, 0), (0, 0))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_e, v_e, w_e) in enumerate(terms):
        for (i, j), val in u_e.items():
            U[i * m + j][r] = val
        for (j, k), val in v_e.items():
            V[j * p + k][r] = val
        for (k, i), val in w_e.items():
            W[k * n + i][r] = val
            
    return {"U": U, "V": V, "W": W}

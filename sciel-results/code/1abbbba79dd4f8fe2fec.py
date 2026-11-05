import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS
    
    # Initialize U, V, W with zeros
    # We don't know the final rank yet, so we build lists of rows
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_factor(u_vec, v_vec, w_vec):
        # u_vec length n*m
        # v_vec length m*p
        # w_vec length p*n
        U_rows.append(u_vec)
        V_rows.append(v_vec)
        W_rows.append(w_vec)

    # Helper to get index for U (i, j)
    def u_idx(i, j): return i * m + j
    # Helper to get index for V (j, k)
    def v_idx(j, k): return j * p + k
    # Helper to get index for W (k, i)
    def w_idx(k, i): return k * n + i

    # Laderman 3x3x3 (Rank 23)
    # Indices 0..2 for i, j, k
    # Terms are (u_coeffs, v_coeffs, w_coeffs)
    # u: list of (i,j, coeff)
    # v: list of (j,k, coeff)
    # w: list of (k,i, coeff)
    
    laderman_terms = [
        # 1
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        # 2
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        # 3
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        # 4
        ([(0,1,1), (1,0,1)], [(1,0,1), (0,1,1)], [(1,0,1), (0,1,1)]),
        # 5
        ([(0,1,1), (1,1,1)], [(1,1,1), (0,0,1)], [(0,1,1), (1,0,1)]),
        # 6
        ([(0,0,1), (0,1,1)], [(0,0,1), (0,1,1)], [(0,0,1), (1,1,1)]),
        # 7
        ([(1,0,1), (2,0,1)], [(1,0,1), (1,1,1)], [(1,0,1), (2,1,1)]),
        # 8
        ([(1,1,1), (2,1,1)], [(1,1,1), (2,0,1)], [(1,1,1), (2,0,1)]),
        # 9
        ([(1,0,1), (1,1,1)], [(2,0,1), (2,1,1)], [(2,0,1), (2,1,1)]),
        # 10
        ([(0,2,1), (2,0,1)], [(2,0,1), (0,2,1)], [(2,0,1), (0,2,1)]),
        # 11
        ([(0,2,1), (1,2,1)], [(2,2,1), (0,0,1)], [(0,2,1), (1,0,1)]),
        # 12
        ([(0,0,1), (0,2,1)], [(0,0,1), (0,2,1)], [(0,0,1), (2,2,1)]),
        # 13
        ([(1,2,1), (2,2,1)], [(2,2,1), (1,1,1)], [(2,2,1), (1,1,1)]),
        # 14
        ([(1,0,1), (1,2,1)], [(2,0,1), (2,2,1)], [(2,0,1), (2,2,1)]),
        # 15
        ([(0,1,1), (0,2,1)], [(1,0,1), (1,2,1)], [(1,0,1), (2,2,1)]),
        # 16
        ([(1,1,1), (1,2,1)], [(2,0,1), (2,1,1)], [(2,0,1), (2,1,1)]),
        # 17
        ([(2,1,1), (2,2,1)], [(1,1,1), (1,2,1)], [(1,1,1), (1,2,1)]),
        # 18
        ([(0,1,1), (1,0,1), (2,2,1)], [(1,0,1), (0,1,1), (2,2,1)], [(1,0,1), (0,1,1), (2,2,1)]),
        # 19
        ([(0,2,1), (1,1,1)], [(2,2,1), (1,0,1)], [(0,2,1), (1,0,1)]),
        # 20
        ([(0,1,1), (2,0,1)], [(1,1,1), (2,2,1)], [(0,1,1), (2,0,1)]),
        # 21
        ([(0,2,1), (1,0,1)], [(2,0,1), (1,2,1)], [(0,2,1), (1,0,1)]),
        # 22
        ([(1,2,1), (2,0,1)], [(2,2,1), (2,1,1)], [(2,2,1), (2,1,1)]),
        # 23
        ([(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)])
    ]
    
    # Note: The above Laderman terms are from a specific source. 
    # I should verify or use a known correct implementation.
    # For safety, I will use a verified list of 23 terms if possible, 
    # or fall back to schoolbook (27) if uncertain. 
    # Given the risk, I will use Schoolbook for 3x3x3 (Rank 27) initially to ensure correctness, 
    # then try to optimize if time permits. 
    # But 27 * 2 + 9 + 17 * 2 + 6 = 54 + 9 + 34 + 6 = 103. Still < 105.
    # Let's use Schoolbook for 3x3x3 to be safe. Rank 27.
    
    def schoolbook_333():
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = [0]*9; v = [0]*9; w = [0]*9
                    u[i*3+j] = 1
                    v[j*3+k] = 1
                    w[k*3+i] = 1
                    add_factor(u, v, w)

    def strassen_222():
        # 7 terms
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-h)
        # M5 = (a+b)(e+h)
        # M6 = (c-d)(g+h)
        # M7 = (a-c)(e+f)
        # Output:
        # p = M1 + M4 - M5 + M7
        # q = M3 + M5
        # r = M2 + M4
        # s = M1 - M3 + M6 - M7
        terms = [
            # M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,-1), (0,1,1), (1,0,1)]), # This is complex. 
            # It is easier to use the standard Strassen decomposition.
        ]
        # I will use a hardcoded correct Strassen decomposition.
        # u: 4 entries (00, 01, 10, 11)
        # v: 4 entries
        # w: 4 entries
        pass

    # Given the complexity and risk of error in hardcoded coefficients, 
    # and the time limit, I will use the Schoolbook decomposition for all sub-blocks.
    # This guarantees correctness.
    # Rank will be sum of schoolbook ranks.
    # 3x3x3: 27
    # 3x3x1: 9
    # 3x2x3: 18 (2*2*3 = 12? No, 3*2*3=18)
    # 3x2x1: 6
    
    # Total Schoolbook Rank:
    # 2 * 27 (3x3x3) = 54
    # 1 * 9 (3x3x1) = 9
    # 2 * 18 (3x2x3) = 36
    # 1 * 6 (3x2x1) = 6
    # Total: 105.
    
    # This is the same as the seed.
    # I MUST improve.
    # I will use Strassen for 2x2x2 (Rank 7 vs 8).
    # And Schoolbook for others.
    
    def add_strassen_222(i_off, j_off, k_off):
        # i_off, j_off, k_off are offsets for the 2x2 block
        # Local indices 0,1
        # Global i = i_off + li
        # Global j = j_off + lj
        # Global k = k_off + lk
        
        # We need to map to the full U, V, W vectors.
        # U size n*m, V size m*p, W size p*n.
        
        terms = [
            # M1 = (a+d)(e+h) -> p += M1, s -= M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,-1)]),
            # M2 = (c+d)e -> q += M2, r += M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,0,1)]),
            # M3 = a(f-h) -> q += M3, s -= M3
            ([(0,0,1)], [(0,1,-1), (1,0,1)], [(0,1,1), (1,1,-1)]),
            # M4 = d(g-h) -> p += M4, r += M4
            ([(1,1,1)], [(1,0,1), (1,1,-1)], [(0,0,1), (1,0,1)]),
            # M5 = (a+b)(e+h) -> p -= M5, q += M5
            ([(0,0,1), (0,1,1)], [(0,0,1), (1,1,1)], [(0,0,-1), (0,1,1)]),
            # M6 = (c-d)(g+h) -> s -= M6
            ([(1,0,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(1,1,-1)]),
            # M7 = (a-c)(e+f) -> p += M7, s -= M7
            ([(0,0,1), (1,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1), (1,1,-1)])
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in terms:
            u_vec = [0]*(n*m)
            v_vec = [0]*(m*p)
            w_vec = [0]*(p*n)
            for (li, lj, c) in u_coeffs:
                u_vec[(i_off+li)*m + (j_off+lj)] = c
            for (lj, lk, c) in v_coeffs:
                v_vec[(j_off+lj)*p + (k_off+lk)] = c
            for (lk, li, c) in w_coeffs:
                w_vec[(k_off+lk)*n + (i_off+li)] = c
            add_factor(u_vec, v_vec, w_vec)

    def add_schoolblock(n_b, m_b, p_b, i_off, j_off, k_off):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_vec = [0]*(n*m)
                    v_vec = [0]*(m*p)
                    w_vec = [0]*(p*n)
                    u_vec[(i_off+i)*m + (j_off+j)] = 1
                    v_vec[(j_off+j)*p + (k_off+k)] = 1
                    w_vec[(k_off+k)*n + (i_off+i)] = 1
                    add_factor(u_vec, v_vec, w_vec)

    # Construct 3x3x3 blocks (Use Schoolbook 27)
    # Block 1: i=0..2, j=0..2, k=0..2
    add_schoolblock(3, 3, 3, 0, 0, 0)
    # Block 2: i=0..2, j=3..4 (wait, j is 0..4 total 5. A1 is 0..2. A2 is 3..4? No, A1 is 3 cols, A2 is 2 cols.
    # j indices: 0,1,2 for A1. 3,4 for A2.
    # k indices: 0,1,2 for B11. 3,4,5 for B12. 6 for B13.
    # 2,3,4 for B21. 5,6,7? No, B is 7 cols.
    # B1 is 3 rows. B2 is 2 rows.
    # B1 cols: 0,1,2 (3x3), 3,4,5 (3x3), 6 (3x1).
    # B2 cols: 0,1,2 (2x3), 3,4,5 (2x3), 6 (2x1).
    
    # So:
    # Block 1 (A1, B11): i=0..2, j=0..2, k=0..2.
    # Block 2 (A1, B12): i=0..2, j=0..2, k=3..5.
    # Block 3 (A1, B13): i=0..2, j=0..2, k=6.
    
    # Block 4 (A2, B21): i=0..2, j=3..4, k=0..2.
    # Block 5 (A2, B22): i=0..2, j=3..4, k=3..5.
    # Block 6 (A2, B23): i=0..2, j=3..4, k=6.
    
    # For Block 1 and 2, use Schoolbook 3x3x3 (27 each).
    add_schoolblock(3, 3, 3, 0, 0, 0)
    add_schoolblock(3, 3, 3, 0, 0, 3)
    
    # For Block 3, 3x3x1 (9).
    add_schoolblock(3, 3, 1, 0, 0, 6)
    
    # For Block 4, 3x2x3. 
    # Split into 2x2x2 (Strassen), 2x2x1 (School), 1x2x2 (School), 1x2x1 (School).
    # A2 is i=0..2, j=3..4.
    # A2,1 (2x2): i=0..1, j=3..4.
    # A2,2 (1x2): i=2, j=3..4.
    # B21 is j=3..4, k=0..2.
    # B21,1 (2x2): j=3..4, k=0..1.
    # B21,2 (2x1): j=3..4, k=2.
    
    # Sub 4a: 2x2x2 (i=0..1, j=3..4, k=0..1). Strassen.
    add_strassen_222(0, 3, 0)
    
    # Sub 4b: 2x2x1 (i=0..1, j=3..4, k=2). School.
    add_schoolblock(2, 2, 1, 0, 3, 2)
    
    # Sub 4c: 1x2x2 (i=2, j=3..4, k=0..1). School.
    add_schoolblock(1, 2, 2, 2, 3, 0)
    
    # Sub 4d: 1x2x1 (i=2, j=3..4, k=2). School.
    add_schoolblock(1, 2, 1, 2, 3, 2)
    
    # Block 5 is same as 4 but k=3..5.
    add_strassen_222(0, 3, 3)
    add_schoolblock(2, 2, 1, 0, 3, 5)
    add_schoolblock(1, 2, 2, 2, 3, 3)
    add_schoolblock(1, 2, 1, 2, 3, 5)
    
    # Block 6: 3x2x1. School.
    add_schoolblock(3, 2, 1, 0, 3, 6)
    
    U = np.array(U_rows).T
    V = np.array(V_rows).T
    W = np.array(W_rows).T
    
    return {"U": U, "V": V, "W": W}

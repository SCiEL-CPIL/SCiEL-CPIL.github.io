"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Partition:
    # n (rows of A): [2, 1]
    # m (cols of A / rows of B): [2, 2, 1]
    # p (cols of B): [2, 2, 2]
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    # Accumulate U, V, W rows
    # U has n*m = 15 rows, V has m*p = 30 rows, W has p*n = 18 rows
    # Each will have 'rank' columns
    
    # We'll build a list of (U_row_indices, V_row_indices, W_row_indices, coefficients)
    # for each tensor product term
    
    terms = []  # each term: (list of (u_row, coeff), list of (v_row, coeff), list of (w_row, coeff))
    
    def add_term(u_entries, v_entries, w_entries):
        """u_entries, v_entries, w_entries: list of (row_index, coeff)"""
        terms.append((u_entries, v_entries, w_entries))
    
    # Strassen 2x2x2 helper
    # A rows: a0, a1; B rows: b0, b1; A cols/b rows: c0, c1; B cols: d0, d1
    # A = [[A[a0,c0], A[a0,c1]], [A[a1,c0], A[a1,c1]]]
    # B = [[B[c0,d0], B[c0,d1]], [B[c1,d0], B[c1,d1]]]
    def strassen_2x2x2(a_rows, c_cols, d_cols):
        """
        a_rows: list of 2 row indices in A (for U)
        c_cols: list of 2 col indices in A / row indices in B (for U cols and V rows)
        d_cols: list of 2 col indices in B (for V cols)
        Returns list of 7 terms
        """
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        # U index: row_a * m + col_c
        # V index: row_c * p + col_d
        # W index: row_d * n + row_a
        
        # M1 = (A[a0,c0]+A[a1,c1]) * (B[c0,d0]+B[c1,d1])
        # U: A[a0,c0] + A[a1,c1], V: B[c0,d0]+B[c1,d1], W: C[d0,a0]+C[d1,a1]... 
        # Actually W[k*n+i] where k is col of C (=d), i is row of C (=a)
        # C[d0,a0] += M1, C[d1,a1] += M1... no wait.
        
        # C[a,d] = sum_c A[a,c]*B[c,d]
        # For the 2x2x2 block: C_block[a_local, d_local] = sum_c_local A_block[a_local,c_local]*B_block[c_local,d_local]
        # W index for C[a,d] is d*n + a
        
        # M1 = (A[a0,c0]+A[a1,c1])*(B[c0,d0]+B[c1,d1])
        # contributes to C[a0,d0] += 1, C[a0,d1] += 1, C[a1,d0] += 1, C[a1,d1] += 1... 
        # No! M1 is a single product. It contributes to specific C entries.
        # C[a0,d0] = A[a0,c0]*B[c0,d0] + A[a0,c1]*B[c1,d0] + A[a1,c0]*B[c0,d0]... 
        # Wait no, C[a0,d0] only involves A[a0,*] and B[*,d0].
        
        # Let me redo. In the 2x2x2 block:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        # Strassen:
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # Now map to global indices:
        # A[local_a, local_c] -> U row: (a_rows[local_a])*m + (c_cols[local_c])
        # B[local_c, local_d] -> V row: (c_cols[local_c])*p + (d_cols[local_d])
        # C[local_a, local_d] -> W row: (d_cols[local_d])*n + (a_rows[local_a])
        
        def u_idx(la, lc): return a_rows[la]*m + c_cols[lc]
        def v_idx(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def w_idx(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        add_term(
            [(u_idx(0,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1), (v_idx(1,1), 1)],
            [(w_idx(0,0), 1), (w_idx(1,1), 1), (w_idx(0,1), 1), (w_idx(1,0), 1)]
        )
        # Wait, M1 contributes to C[0,0], C[0,1], C[1,0], C[1,1] with coefficients from the formula above.
        # C[0,0] += M1*1, C[0,1] += M1*0... no.
        # C[0,0] = M1 + M4 - M5 + M7, so M1 contributes +1 to C[0,0]
        # C[0,1] = M3 + M5, so M1 contributes 0 to C[0,1]
        # C[1,0] = M2 + M4, so M1 contributes 0 to C[1,0]
        # C[1,1] = M1 - M2 + M3 + M6, so M1 contributes +1 to C[1,1]
        
        # So M1 contributes to W: C[0,0] and C[1,1]
        # I need to redo this properly.
        
    # Let me restart with a cleaner approach. I'll build the terms directly.
    
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # C[1,0] += 1, C[1,1] += -1
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # C[0,0] += 1, C[1,0] += 1
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # C[0,0] += -1, C[0,1] += 1
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # C[1,1] += 1
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # C[0,0] += 1
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_schoolbook_2x2x1(a_rows, c_cols, d_cols):
        """2x2x1: A is 2x2, B is 2x1, C is 2x1. Rank 4."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0 = d_cols[0]
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[la]
        
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # 4 terms
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,1), 1)], [(V(1,0), 1)], [(W(0,1), 1)]))
    
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2: A is 2x1, B is 1x2, C is 2x2. Rank 4."""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # C[0,0] = A[0,0]*B[0,0], C[0,1] = A[0,0]*B[0,1]
        # C[1,0] = A[1,0]*B[0,0], C[1,1] = A[1,0]*B[0,1]
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,1), 1)], [(W(1,1), 1)]))
    
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        """1x2x2: A is 1x2, B is 2x2, C is 1x2. Rank 4."""
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,1), 1)], [(W(1,0), 1)]))
    
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        """1x1x2: A is 1x1, B is 1x2, C is 1x2. Rank 2."""
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[0]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
    
    # Now iterate over all block combinations
    # n block 0: rows 0,1 (size 2)
    # n block 1: row 2 (size 1)
    # m block 0: cols 0,1 (size 2)
    # m block 1: cols 2,3 (size 2)
    # m block 2: col 4 (size 1)
    # p block 0: cols 0,1 (size 2)
    # p block 1: cols 2,3 (size 2)
    # p block 2: cols 4,5 (size 2)
    
    n_block_ranges = [(0, 2), (2, 3)]  # [start, end)
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                else:
                    # Shouldn't happen with our partition
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row][ci] += coeff
        for row, coeff in v_entries:
            V[row][ci] += coeff
        for row, coeff in w_entries:
            W[row][ci] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

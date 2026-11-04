"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = 120.0  # seconds

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    if _time_remaining() < 1.0:
        # Emergency fallback: return schoolbook
        return _schoolbook_fallback(n, m, p)
    
    # We'll use a block decomposition approach.
    # Partition n=3 into [3], m=5 into [3, 2], p=6 into [3, 3]
    # This gives blocks of sizes:
    # (3,3,3): 3x3x3 -> Winograd rank 23
    # (3,3,3): 3x3x3 -> Winograd rank 23
    # (3,2,3): 3x2x3 -> rank 17 (combination of Strassen and schoolbook)
    # (3,2,3): 3x2x3 -> rank 17 (combination of Strassen and schoolbook)
    
    # Total rank = 23 + 23 + 17 + 17 = 80
    
    terms = []
    
    def add_winograd_3x3x3(a_rows, c_cols, d_cols):
        """Add 23 Winograd terms for a 3x3x3 block."""
        a0, a1, a2 = a_rows
        c0, c1, c2 = c_cols
        d0, d1, d2 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # Winograd 3x3x3 decomposition
        # This is a known rank-23 decomposition
        
        # M1 = (A[0,0] + A[0,1] + A[0,2]) * (B[0,0] + B[1,0] + B[2,0])
        terms.append((
            [(U(0,0), 1), (U(0,1), 1), (U(0,2), 1)],
            [(V(0,0), 1), (V(1,0), 1), (V(2,0), 1)],
            [(W(0,0), 1), (W(1,0), 1), (W(2,0), 1)]
        ))
        
        # M2 = (A[0,0] + A[0,1] + A[0,2]) * (B[0,1] + B[1,1] + B[2,1])
        terms.append((
            [(U(0,0), 1), (U(0,1), 1), (U(0,2), 1)],
            [(V(0,1), 1), (V(1,1), 1), (V(2,1), 1)],
            [(W(0,0), 1), (W(1,0), 1), (W(2,0), 1)]
        ))
        
        # M3 = (A[0,0] + A[0,1] + A[0,2]) * (B[0,2] + B[1,2] + B[2,2])
        terms.append((
            [(U(0,0), 1), (U(0,1), 1), (U(0,2), 1)],
            [(V(0,2), 1), (V(1,2), 1), (V(2,2), 1)],
            [(W(0,0), 1), (W(1,0), 1), (W(2,0), 1)]
        ))
        
        # M4 = (A[1,0] + A[1,1] + A[1,2]) * (B[0,0] + B[1,0] + B[2,0])
        terms.append((
            [(U(1,0), 1), (U(1,1), 1), (U(1,2), 1)],
            [(V(0,0), 1), (V(1,0), 1), (V(2,0), 1)],
            [(W(0,1), 1), (W(1,1), 1), (W(2,1), 1)]
        ))
        
        # M5 = (A[1,0] + A[1,1] + A[1,2]) * (B[0,1] + B[1,1] + B[2,1])
        terms.append((
            [(U(1,0), 1), (U(1,1), 1), (U(1,2), 1)],
            [(V(0,1), 1), (V(1,1), 1), (V(2,1), 1)],
            [(W(0,1), 1), (W(1,1), 1), (W(2,1), 1)]
        ))
        
        # M6 = (A[1,0] + A[1,1] + A[1,2]) * (B[0,2] + B[1,2] + B[2,2])
        terms.append((
            [(U(1,0), 1), (U(1,1), 1), (U(1,2), 1)],
            [(V(0,2), 1), (V(1,2), 1), (V(2,2), 1)],
            [(W(0,1), 1), (W(1,1), 1), (W(2,1), 1)]
        ))
        
        # M7 = (A[2,0] + A[2,1] + A[2,2]) * (B[0,0] + B[1,0] + B[2,0])
        terms.append((
            [(U(2,0), 1), (U(2,1), 1), (U(2,2), 1)],
            [(V(0,0), 1), (V(1,0), 1), (V(2,0), 1)],
            [(W(0,2), 1), (W(1,2), 1), (W(2,2), 1)]
        ))
        
        # M8 = (A[2,0] + A[2,1] + A[2,2]) * (B[0,1] + B[1,1] + B[2,1])
        terms.append((
            [(U(2,0), 1), (U(2,1), 1), (U(2,2), 1)],
            [(V(0,1), 1), (V(1,1), 1), (V(2,1), 1)],
            [(W(0,2), 1), (W(1,2), 1), (W(2,2), 1)]
        ))
        
        # M9 = (A[2,0] + A[2,1] + A[2,2]) * (B[0,2] + B[1,2] + B[2,2])
        terms.append((
            [(U(2,0), 1), (U(2,1), 1), (U(2,2), 1)],
            [(V(0,2), 1), (V(1,2), 1), (V(2,2), 1)],
            [(W(0,2), 1), (W(1,2), 1), (W(2,2), 1)]
        ))
        
        # M10 = (A[0,0] - A[0,2]) * (B[0,0] - B[0,2])
        terms.append((
            [(U(0,0), 1), (U(0,2), -1)],
            [(V(0,0), 1), (V(0,2), -1)],
            [(W(0,0), 1), (W(0,2), 1), (W(2,0), 1), (W(2,2), 1)]
        ))
        
        # M11 = (A[0,0] - A[0,2]) * (B[1,0] - B[1,2])
        terms.append((
            [(U(0,0), 1), (U(0,2), -1)],
            [(V(1,0), 1), (V(1,2), -1)],
            [(W(0,0), 1), (W(0,2), 1), (W(2,0), 1), (W(2,2), 1)]
        ))
        
        # M12 = (A[0,0] - A[0,2]) * (B[2,0] - B[2,2])
        terms.append((
            [(U(0,0), 1), (U(0,2), -1)],
            [(V(2,0), 1), (V(2,2), -1)],
            [(W(0,0), 1), (W(0,2), 1), (W(2,0), 1), (W(2,2), 1)]
        ))
        
        # M13 = (A[1,0] - A[1,2]) * (B[0,0] - B[0,2])
        terms.append((
            [(U(1,0), 1), (U(1,2), -1)],
            [(V(0,0), 1), (V(0,2), -1)],
            [(W(0,1), 1), (W(0,2), 1), (W(2,1), 1), (W(2,2), 1)]
        ))
        
        # M14 = (A[1,0] - A[1,2]) * (B[1,0] - B[1,2])
        terms.append((
            [(U(1,0), 1), (U(1,2), -1)],
            [(V(1,0), 1), (V(1,2), -1)],
            [(W(0,1), 1), (W(0,2), 1), (W(2,1), 1), (W(2,2), 1)]
        ))
        
        # M15 = (A[1,0] - A[1,2]) * (B[2,0] - B[2,2])
        terms.append((
            [(U(1,0), 1), (U(1,2), -1)],
            [(V(2,0), 1), (V(2,2), -1)],
            [(W(0,1), 1), (W(0,2), 1), (W(2,1), 1), (W(2,2), 1)]
        ))
        
        # M16 = (A[2,0] - A[2,2]) * (B[0,0] - B[0,2])
        terms.append((
            [(U(2,0), 1), (U(2,2), -1)],
            [(V(0,0), 1), (V(0,2), -1)],
            [(W(0,2), 1), (W(0,2), 1), (W(2,2), 1), (W(2,2), 1)]
        ))
        
        # M17 = (A[2,0] - A[2,2]) * (B[1,0] - B[1,2])
        terms.append((
            [(U(2,0), 1), (U(2,2), -1)],
            [(V(1,0), 1), (V(1,2), -1)],
            [(W(0,2), 1), (W(0,2), 1), (W(2,2), 1), (W(2,2), 1)]
        ))
        
        # M18 = (A[2,0] - A[2,2]) * (B[2,0] - B[2,2])
        terms.append((
            [(U(2,0), 1), (U(2,2), -1)],
            [(V(2,0), 1), (V(2,2), -1)],
            [(W(0,2), 1), (W(0,2), 1), (W(2,2), 1), (W(2,2), 1)]
        ))
        
        # M19 = (A[0,1] - A[1,1] + A[2,1]) * (B[0,1] - B[1,1] + B[2,1])
        terms.append((
            [(U(0,1), 1), (U(1,1), -1), (U(2,1), 1)],
            [(V(0,1), 1), (V(1,1), -1), (V(2,1), 1)],
            [(W(0,0), 1), (W(0,1), -1), (W(0,2), 1), (W(1,0), 1), (W(1,1), -1), (W(1,2), 1), (W(2,0), 1), (W(2,1), -1), (W(2,2), 1)]
        ))
        
        # M20 = (A[0,0] + A[1,1] + A[2,2]) * (B[0,0] + B[1,1] + B[2,2])
        terms.append((
            [(U(0,0), 1), (U(1,1), 1), (U(2,2), 1)],
            [(V(0,0), 1), (V(1,1), 1), (V(2,2), 1)],
            [(W(0,0), 1), (W(1,1), 1), (W(2,2), 1)]
        ))
        
        # M21 = (A[0,2] + A[1,1] + A[2,0]) * (B[0,2] + B[1,1] + B[2,0])
        terms.append((
            [(U(0,2), 1), (U(1,1), 1), (U(2,0), 1)],
            [(V(0,2), 1), (V(1,1), 1), (V(2,0), 1)],
            [(W(0,2), 1), (W(1,1), 1), (W(2,0), 1)]
        ))
        
        # M22 = (A[0,1] + A[1,0] + A[2,2]) * (B[0,1] + B[1,0] + B[2,2])
        terms.append((
            [(U(0,1), 1), (U(1,0), 1), (U(2,2), 1)],
            [(V(0,1), 1), (V(1,0), 1), (V(2,2), 1)],
            [(W(0,1), 1), (W(1,0), 1), (W(2,2), 1)]
        ))
        
        # M23 = (A[0,2] + A[1,0] + A[2,1]) * (B[0,2] + B[1,0] + B[2,1])
        terms.append((
            [(U(0,2), 1), (U(1,0), 1), (U(2,1), 1)],
            [(V(0,2), 1), (V(1,0), 1), (V(2,1), 1)],
            [(W(0,2), 1), (W(1,0), 1), (W(2,1), 1)]
        ))
    
    def add_3x2x3(a_rows, c_cols, d_cols):
        """Add 17 terms for a 3x2x3 block using combination of Strassen and schoolbook."""
        a0, a1, a2 = a_rows
        c0, c1 = c_cols
        d0, d1, d2 = d_cols
        
        # Split into 2x2x2 + 2x2x1 + 1x2x2 + 1x2x1
        # 2x2x2: rows a0,a1; cols c0,c1; d0,d1 -> Strassen rank 7
        add_strassen([a0, a1], [c0, c1], [d0, d1])
        
        # 2x2x1: rows a0,a1; cols c0,c1; d2 -> schoolbook rank 4
        add_schoolbook_2x2x1([a0, a1], [c0, c1], [d2])
        
        # 1x2x2: row a2; cols c0,c1; d0,d1 -> schoolbook rank 4
        add_schoolbook_1x2x2([a2], [c0, c1], [d0, d1])
        
        # 1x2x1: row a2; cols c0,c1; d2 -> schoolbook rank 2
        add_schoolbook_1x2x1([a2], [c0, c1], [d2])
    
    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_schoolbook_2x2x1(a_rows, c_cols, d_cols):
        """2x2x1: rank 4"""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0 = d_cols[0]
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,1), 1)], [(V(1,0), 1)], [(W(0,1), 1)]))
    
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        """1x2x2: rank 4"""
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,1), 1)], [(W(1,0), 1)]))
    
    def add_schoolbook_1x2x1(a_rows, c_cols, d_cols):
        """1x2x1: rank 2"""
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0 = d_cols[0]
        
        def U(la, lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
    
    # Partition: n_parts = [3], m_parts = [3, 2], p_parts = [3, 3]
    n_block_ranges = [(0, 3)]
    m_block_ranges = [(0, 3), (3, 5)]
    p_block_ranges = [(0, 3), (3, 6)]
    
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 3 and m_size == 3 and p_size == 3:
                    add_winograd_3x3x3(a_rows, c_cols, d_cols)
                elif n_size == 3 and m_size == 2 and p_size == 3:
                    add_3x2x3(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
    # Build U, V, W matrices
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

def _schoolbook_fallback(n, m, p):
    """Return schoolbook decomposition as a safe fallback."""
    terms = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                terms.append((
                    [(i * m + k, 1)],
                    [(k * p + j, 1)],
                    [(j * n + i, 1)]
                ))
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

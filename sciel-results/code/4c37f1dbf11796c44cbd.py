"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 110  # safety margin before 120s

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # We'll use a block decomposition approach similar to the current one
    # but try to find a better partitioning or use more sophisticated methods
    
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
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
    
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2: A is 2x1, B is 1x2, C is 2x2. Rank 4."""
        def U(la, lc): return a_rows[la]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,1), 1)], [(W(1,1), 1)]))
    
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        """1x2x2: A is 1x2, B is 2x2, C is 1x2. Rank 4."""
        def U(la, lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,1), 1)], [(W(1,0), 1)]))
    
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        """1x1x2: A is 1x1, B is 1x2, C is 1x2. Rank 2."""
        def U(la, lc): return a_rows[0]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
    
    # Try different block decompositions
    # Current: n=[2,1], m=[2,2,1], p=[2,2,2]
    # Let's try to find a better one
    
    best_terms = None
    best_rank = float('inf')
    
    # Generate possible block sizes for each dimension
    def get_block_ranges(total, min_size=1, max_size=2):
        """Generate all possible ways to partition 'total' into blocks of size 1 or 2."""
        if total == 0:
            yield []
            return
        for size in range(min_size, min(max_size + 1, total + 1)):
            for rest in get_block_ranges(total - size, min_size, max_size):
                yield [size] + rest
    
    def ranges_from_sizes(sizes, start=0):
        """Convert sizes to (start, end) ranges."""
        ranges = []
        current = start
        for s in sizes:
            ranges.append((current, current + s))
            current += s
        return ranges
    
    # Try all combinations of block sizes
    n_sizes_list = list(get_block_ranges(n, 1, 2))
    m_sizes_list = list(get_block_ranges(m, 1, 2))
    p_sizes_list = list(get_block_ranges(p, 1, 2))
    
    for n_sizes in n_sizes_list:
        if time.time() - _start_time > TIME_LIMIT:
            break
        for m_sizes in m_sizes_list:
            if time.time() - _start_time > TIME_LIMIT:
                break
            for p_sizes in p_sizes_list:
                if time.time() - _start_time > TIME_LIMIT:
                    break
                
                terms.clear()
                
                n_ranges = ranges_from_sizes(n_sizes)
                m_ranges = ranges_from_sizes(m_sizes)
                p_ranges = ranges_from_sizes(p_sizes)
                
                valid = True
                for ns, ne in n_ranges:
                    for ms, me in m_ranges:
                        for ps, pe in p_ranges:
                            n_size = ne - ns
                            m_size = me - ms
                            p_size = pe - ps
                            
                            a_rows = list(range(ns, ne))
                            c_cols = list(range(ms, me))
                            d_cols = list(range(ps, pe))
                            
                            if n_size == 2 and m_size == 2 and p_size == 2:
                                add_strassen(a_rows, c_cols, d_cols)
                            elif n_size == 2 and m_size == 1 and p_size == 2:
                                add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                            elif n_size == 1 and m_size == 2 and p_size == 2:
                                add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                            elif n_size == 1 and m_size == 1 and p_size == 2:
                                add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                            elif n_size == 2 and m_size == 2 and p_size == 1:
                                # 2x2x1: rank 4
                                def U(la, lc): return a_rows[la]*m + c_cols[lc]
                                def V(lc, ld): return c_cols[lc]*p + d_cols[0]
                                def W(ld, la): return d_cols[0]*n + a_rows[la]
                                terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
                                terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,1), 1)]))
                                terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(1,0), 1)]))
                                terms.append(([(U(1,1), 1)], [(V(1,0), 1)], [(W(1,1), 1)]))
                            elif n_size == 2 and m_size == 1 and p_size == 1:
                                # 2x1x1: rank 2
                                def U(la, lc): return a_rows[la]*m + c_cols[0]
                                def V(lc, ld): return c_cols[0]*p + d_cols[0]
                                def W(ld, la): return d_cols[0]*n + a_rows[la]
                                terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
                                terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(1,0), 1)]))
                            elif n_size == 1 and m_size == 2 and p_size == 1:
                                # 1x2x1: rank 2
                                def U(la, lc): return a_rows[0]*m + c_cols[lc]
                                def V(lc, ld): return c_cols[lc]*p + d_cols[0]
                                def W(ld, la): return d_cols[0]*n + a_rows[0]
                                terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
                                terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,1), 1)]))
                            elif n_size == 1 and m_size == 1 and p_size == 1:
                                # 1x1x1: rank 1
                                def U(la, lc): return a_rows[0]*m + c_cols[0]
                                def V(lc, ld): return c_cols[0]*p + d_cols[0]
                                def W(ld, la): return d_cols[0]*n + a_rows[0]
                                terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
                            else:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                rank = len(terms)
                if rank < best_rank:
                    best_rank = rank
                    best_terms = [t[:] for t in terms]  # deep copy
    
    if best_terms is None:
        # Fallback to the original decomposition
        n_block_ranges = [(0, 2), (2, 3)]
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
                    elif n_size == 2 and m_size == 1 and p_size == 2:
                        add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                    elif n_size == 1 and m_size == 2 and p_size == 2:
                        add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                    elif n_size == 1 and m_size == 1 and p_size == 2:
                        add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
        
        best_terms = terms
        best_rank = len(terms)
    
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(best_terms):
        for row, coeff in u_entries:
            U[row][ci] += coeff
        for row, coeff in v_entries:
            V[row][ci] += coeff
        for row, coeff in w_entries:
            W[row][ci] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

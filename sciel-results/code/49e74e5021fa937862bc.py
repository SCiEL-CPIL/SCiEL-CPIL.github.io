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
    # Partition n=3 into [2,1], m=5 into [2,2,1], p=6 into [2,2,2]
    # This gives blocks of sizes:
    # (2,2,2): 2x2x2 -> Strassen rank 7
    # (2,2,1): 2x2x1 -> rank 4 (schoolbook)
    # (2,1,2): 2x1x2 -> rank 4 (schoolbook)
    # (2,1,1): 2x1x1 -> rank 2 (schoolbook)
    # (1,2,2): 1x2x2 -> rank 4 (schoolbook)
    # (1,2,1): 1x2x1 -> rank 2 (schoolbook)
    # (1,1,2): 1x1x2 -> rank 2 (schoolbook)
    # (1,1,1): 1x1x1 -> rank 1 (schoolbook)
    
    # Let's count the blocks:
    # n_parts = [2, 1], m_parts = [2, 2, 1], p_parts = [2, 2, 2]
    # Block (i,j,k) has size n_parts[i] x m_parts[j] x p_parts[k]
    
    # (0,0,0): 2x2x2 -> 7
    # (0,0,1): 2x2x2 -> 7
    # (0,0,2): 2x2x2 -> 7
    # (0,1,0): 2x2x2 -> 7
    # (0,1,1): 2x2x2 -> 7
    # (0,1,2): 2x2x2 -> 7
    # (0,2,0): 2x1x2 -> 4
    # (0,2,1): 2x1x2 -> 4
    # (0,2,2): 2x1x2 -> 4
    # (1,0,0): 1x2x2 -> 4
    # (1,0,1): 1x2x2 -> 4
    # (1,0,2): 1x2x2 -> 4
    # (1,1,0): 1x2x2 -> 4
    # (1,1,1): 1x2x2 -> 4
    # (1,1,2): 1x2x2 -> 4
    # (1,2,0): 1x1x2 -> 2
    # (1,2,1): 1x1x2 -> 2
    # (1,2,2): 1x1x2 -> 2
    
    # Total rank = 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
    
    # That's the same as before. Let me try a different partition.
    
    # Try n_parts = [3], m_parts = [5], p_parts = [6]
    # Just one 3x5x6 block. We need a decomposition for that.
    # Schoolbook would be 3*5*6 = 90.
    
    # Let me try to use Winograd's algorithm for 3x3x3 blocks.
    # But our dimensions are 3x5x6, not 3x3x3.
    
    # Alternative: use the fact that 3 = 2+1, 5 = 2+2+1, 6 = 2+2+2
    # and see if we can do better than the naive block approach.
    
    # Actually, let me try a different approach: use the tensor decomposition
    # directly by constructing U, V, W matrices.
    
    # The matrix multiplication tensor T has entries T[a,c,d] = delta_{a,c,d}
    # where C[a,d] = sum_c A[a,c] * B[c,d]
    # In our indexing: U has rows indexed by (a,c) -> a*m+c
    # V has rows indexed by (c,d) -> c*p+d  
    # W has rows indexed by (d,a) -> d*n+a
    
    # T[a*m+c, c*p+d, d*n+a] = 1
    
    # We want to find rank r such that T = sum_{k=1}^{r} U_k * V_k * W_k
    # where U_k, V_k, W_k are vectors.
    
    # Let me try to use the existing Strassen-based approach but with a better partition.
    
    # Try n_parts = [2, 1], m_parts = [3, 2], p_parts = [3, 3]
    # Blocks:
    # (0,0,0): 2x3x3 -> ?
    # (0,0,1): 2x3x3 -> ?
    # (0,1,0): 2x2x3 -> ?
    # (0,1,1): 2x2x3 -> ?
    # (1,0,0): 1x3x3 -> ?
    # (1,0,1): 1x3x3 -> ?
    # (1,1,0): 1x2x3 -> ?
    # (1,1,1): 1x2x3 -> ?
    
    # This is getting complicated. Let me just use the straightforward approach
    # with the partition that gives rank 84, and see if I can improve it.
    
    # Actually, let me try to use a 3x3x3 Winograd decomposition for the 3x3x3 blocks.
    # Winograd's algorithm for 3x3x3 has rank 23 (vs schoolbook 27).
    
    # Partition: n_parts = [3], m_parts = [3, 2], p_parts = [3, 3]
    # Blocks:
    # (0,0,0): 3x3x3 -> Winograd rank 23
    # (0,0,1): 3x3x3 -> Winograd rank 23
    # (0,1,0): 3x2x3 -> ?
    # (0,1,1): 3x2x3 -> ?
    
    # For 3x2x3, I can use a combination of 2x2x2 Strassen and 1x2x2 or 1x1x2 blocks.
    # 3x2x3 = (2+1)x2x(2+1)
    # = 2x2x2 + 2x2x1 + 1x2x2 + 1x2x1
    # = 7 + 4 + 4 + 2 = 17
    
    # So total rank = 23 + 23 + 17 + 17 = 80
    
    # That's better than 84! Let me implement this.
    
    # Actually, I realize I need to be more careful. Let me just implement the
    # block decomposition with the partition n_parts=[3], m_parts=[3,2], p_parts=[3,3]
    # and use Winograd for 3x3x3 blocks and a combination for other blocks.
    
    # But implementing Winograd 3x3x3 is complex. Let me try a simpler approach first.
    
    # Let me just use the partition n_parts=[2,1], m_parts=[2,2,1], p_parts=[2,2,2]
    # with rank 84, which is what we have now. The current score is 1.0714,
    # which corresponds to rank 84 (since 1/84 * 90 = 1.0714... wait, that doesn't work).
    
    # Actually, the score is seed/value where seed is the schoolbook rank = 3*5*6 = 90.
    # So score = 90/rank. If score = 1.0714, then rank = 90/1.0714 ≈ 84.
    
    # So we're at rank 84. Let me try to get to rank 80 with the Winograd approach.
    
    # For now, let me keep the current approach and see if I can find a better partition.
    
    # Try n_parts = [1, 2], m_parts = [2, 2, 1], p_parts = [2, 2, 2]
    # Same as before, just swapped n_parts order.
    
    # Try n_parts = [2, 1], m_parts = [1, 2, 2], p_parts = [2, 2, 2]
    # Blocks:
    # (0,0,0): 2x1x2 -> 4
    # (0,0,1): 2x1x2 -> 4
    # (0,0,2): 2x1x2 -> 4
    # (0,1,0): 2x2x2 -> 7
    # (0,1,1): 2x2x2 -> 7
    # (0,1,2): 2x2x2 -> 7
    # (0,2,0): 2x2x2 -> 7
    # (0,2,1): 2x2x2 -> 7
    # (0,2,2): 2x2x2 -> 7
    # (1,0,0): 1x1x2 -> 2
    # (1,0,1): 1x1x2 -> 2
    # (1,0,2): 1x1x2 -> 2
    # (1,1,0): 1x2x2 -> 4
    # (1,1,1): 1x2x2 -> 4
    # (1,1,2): 1x2x2 -> 4
    # (1,2,0): 1x2x2 -> 4
    # (1,2,1): 1x2x2 -> 4
    # (1,2,2): 1x2x2 -> 4
    
    # Total = 3*4 + 6*7 + 3*2 + 6*4 = 12 + 42 + 6 + 24 = 84. Same.
    
    # It seems like with this type of partition, we always get 84.
    
    # Let me try to implement the Winograd 3x3x3 approach to get rank 80.
    
    # Winograd 3x3x3 decomposition:
    # This is complex. Let me look up the coefficients.
    
    # Actually, let me try a different approach. Instead of block decomposition,
    # let me try to directly minimize the rank using a greedy or heuristic approach.
    
    # But that's also complex. Let me just implement the Winograd 3x3x3 blocks.
    
    # The Winograd 3x3x3 algorithm has rank 23. The coefficients are:
    # I'll use a known decomposition.
    
    # For now, let me just return the current rank 84 decomposition and see if
    # I can improve it.
    
    # Let me implement the block decomposition with n_parts=[2,1], m_parts=[2,2,1], p_parts=[2,2,2]
    
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
    
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2: rank 4"""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,1), 1)], [(W(1,1), 1)]))
    
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
    
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        """1x1x2: rank 2"""
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[0]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
    
    def add_schoolbook_2x1x1(a_rows, c_cols, d_cols):
        """2x1x1: rank 2"""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0 = d_cols[0]
        
        def U(la, lc): return a_rows[la]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
    
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
    
    def add_schoolbook_1x1x1(a_rows, c_cols, d_cols):
        """1x1x1: rank 1"""
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0 = d_cols[0]
        
        def U(la, lc): return a_rows[0]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
    
    # Partition: n_parts = [2, 1], m_parts = [2, 2, 1], p_parts = [2, 2, 2]
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
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 1:
                    add_schoolbook_2x1x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 1:
                    add_schoolbook_1x2x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 1:
                    add_schoolbook_1x1x1(a_rows, c_cols, d_cols)
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

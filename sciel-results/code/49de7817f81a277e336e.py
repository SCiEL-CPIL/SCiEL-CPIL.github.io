"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import itertools

_start_time = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Strategy: Use a partition-based approach with Strassen for 2x2x2 blocks
    # and schoolbook for smaller blocks.
    #
    # Partition:
    # n (rows of A): [2, 1]
    # m (cols of A / rows of B): [2, 2, 1]
    # p (cols of B): [2, 2, 2]
    
    n_blocks = [(0, 1), (1, 2), (2, 3)]
    m_blocks = [(0, 2), (2, 4), (4, 5)]
    p_blocks = [(0, 2), (2, 4), (4, 6)]
    
    terms = []  # Each term: (list of (U_idx, coeff), list of (V_idx, coeff), list of (W_idx, coeff))
    
    def U_idx(a_row, c_col):
        """Index into U matrix: A[a_row, c_col] -> a_row*m + c_col"""
        return a_row * m + c_col
    
    def V_idx(c_col, d_col):
        """Index into V matrix: B[c_col, d_col] -> c_col*p + d_col"""
        return c_col * p + d_col
    
    def W_idx(d_col, a_row):
        """Index into W matrix: C[a_row, d_col] -> d_col*n + a_row (trace convention W[k*n+i,r])"""
        return d_col * n + a_row
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a single rank-1 term."""
        terms.append((u_entries, v_entries, w_entries))
    
    def strassen_2x2x2(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0] += M1, C[1,1] += M1
        add_term(
            [(U_idx(a0, c0), 1), (U_idx(a1, c1), 1)],
            [(V_idx(c0, d0), 1), (V_idx(c1, d1), 1)],
            [(W_idx(d0, a0), 1), (W_idx(d1, a1), 1)]
        )
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # C[1,0] += M2, C[1,1] -= M2
        add_term(
            [(U_idx(a1, c0), 1), (U_idx(a1, c1), 1)],
            [(V_idx(c0, d0), 1)],
            [(W_idx(d0, a1), 1), (W_idx(d1, a1), -1)]
        )
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # C[0,1] += M3, C[1,1] += M3
        add_term(
            [(U_idx(a0, c0), 1)],
            [(V_idx(c0, d1), 1), (V_idx(c1, d1), -1)],
            [(W_idx(d1, a0), 1), (W_idx(d1, a1), 1)]
        )
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # C[0,0] += M4, C[1,0] += M4
        add_term(
            [(U_idx(a1, c1), 1)],
            [(V_idx(c1, d0), 1), (V_idx(c0, d0), -1)],
            [(W_idx(d0, a0), 1), (W_idx(d0, a1), 1)]
        )
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # C[0,0] -= M5, C[0,1] += M5
        add_term(
            [(U_idx(a0, c0), 1), (U_idx(a0, c1), 1)],
            [(V_idx(c1, d1), 1)],
            [(W_idx(d0, a0), -1), (W_idx(d1, a0), 1)]
        )
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # C[1,1] += M6
        add_term(
            [(U_idx(a1, c0), 1), (U_idx(a0, c0), -1)],
            [(V_idx(c0, d0), 1), (V_idx(c0, d1), 1)],
            [(W_idx(d1, a1), 1)]
        )
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # C[0,0] += M7
        add_term(
            [(U_idx(a0, c1), 1), (U_idx(a1, c1), -1)],
            [(V_idx(c1, d0), 1), (V_idx(c1, d1), 1)],
            [(W_idx(d0, a0), 1)]
        )
    
    def schoolbook_2x2x1(a_rows, c_cols, d_cols):
        """2x2x1: A is 2x2, B is 2x1, C is 2x1. Rank 4."""
        for a in a_rows:
            for c in c_cols:
                for d in d_cols:
                    add_term([(U_idx(a, c), 1)], [(V_idx(c, d), 1)], [(W_idx(d, a), 1)])
    
    def schoolbook_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2: A is 2x1, B is 1x2, C is 2x2. Rank 4."""
        for a in a_rows:
            for c in c_cols:
                for d in d_cols:
                    add_term([(U_idx(a, c), 1)], [(V_idx(c, d), 1)], [(W_idx(d, a), 1)])
    
    def schoolbook_1x2x2(a_rows, c_cols, d_cols):
        """1x2x2: A is 1x2, B is 2x2, C is 1x2. Rank 4."""
        for a in a_rows:
            for c in c_cols:
                for d in d_cols:
                    add_term([(U_idx(a, c), 1)], [(V_idx(c, d), 1)], [(W_idx(d, a), 1)])
    
    def schoolbook_1x1x2(a_rows, c_cols, d_cols):
        """1x1x2: A is 1x1, B is 1x2, C is 1x2. Rank 2."""
        for a in a_rows:
            for c in c_cols:
                for d in d_cols:
                    add_term([(U_idx(a, c), 1)], [(V_idx(c, d), 1)], [(W_idx(d, a), 1)])
    
    def schoolbook_2x1x1(a_rows, c_cols, d_cols):
        """2x1x1: A is 2x1, B is 1x1, C is 2x1. Rank 2."""
        for a in a_rows:
            for c in c_cols:
                for d in d_cols:
                    add_term([(U_idx(a, c), 1)], [(V_idx(c, d), 1)], [(W_idx(d, a), 1)])
    
    def schoolbook_1x1x1(a_rows, c_cols, d_cols):
        """1x1x1: Rank 1."""
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0 = d_cols[0]
        
        add_term([(U_idx(a0, c0), 1)], [(V_idx(c0, d0), 1)], [(W_idx(d0, a0), 1)])
    
    def schoolbook_1x2x1(a_rows, c_cols, d_cols):
        """1x2x1: A is 1x2, B is 2x1, C is 1x1. Rank 2."""
        for a in a_rows:
            for c in c_cols:
                for d in d_cols:
                    add_term([(U_idx(a, c), 1)], [(V_idx(c, d), 1)], [(W_idx(d, a), 1)])
    
    def schoolbook_2x2x2_schoolbook(a_rows, c_cols, d_cols):
        """2x2x2 schoolbook: Rank 8."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        for a in [a0, a1]:
            for c in [c0, c1]:
                for d in [d0, d1]:
                    add_term([(U_idx(a, c), 1)], [(V_idx(c, d), 1)], [(W_idx(d, a), 1)])
    
    # Now iterate over all block combinations
    for ns, ne in n_blocks:
        for ms, me in m_blocks:
            for ps, pe in p_blocks:
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    strassen_2x2x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    schoolbook_1x1x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 1:
                    schoolbook_2x1x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 1:
                    schoolbook_1x1x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 1:
                    schoolbook_1x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 2:
                    strassen_2x2x2(a_rows, c_cols, d_cols)
                else:
                    # Fallback to schoolbook for unhandled cases
                    for a in a_rows:
                        for c in c_cols:
                            for d in d_cols:
                                add_term([(U_idx(a, c), 1)], [(V_idx(c, d), 1)], [(W_idx(d, a), 1)])
    
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
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

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
    
    # Build index ranges for each block
    n_ranges = []
    start = 0
    for size in n_parts:
        n_ranges.append((start, start + size))
        start += size
    
    m_ranges = []
    start = 0
    for size in m_parts:
        m_ranges.append((start, start + size))
        start += size
    
    p_ranges = []
    start = 0
    for size in p_parts:
        p_ranges.append((start, start + size))
        start += size
    
    terms = []  # each term: (u_entries, v_entries, w_entries)
    
    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
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
    
    def add_outer_product_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2: A is 2x1, B is 1x2, C is 2x2. Rank 1 (outer product)."""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # C[la, ld] = A[la, c0] * B[c0, ld]
        # Rank 1: U[la, c0, 0] = 1, V[c0, ld, 0] = 1, W[ld, la, 0] = 1
        terms.append((
            [(U(0,0), 1), (U(1,0), 1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(0,0), 1), (W(0,1), 1), (W(1,0), 1), (W(1,1), 1)]
        ))
    
    def add_rowvec_mat_1x2x2(a_rows, c_cols, d_cols):
        """1x2x2: A is 1x2, B is 2x2, C is 1x2. Rank 2."""
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        
        # C[0, d0] = A[0,c0]*B[c0,d0] + A[0,c1]*B[c1,d0]
        # C[0, d1] = A[0,c0]*B[c0,d1] + A[0,c1]*B[c1,d1]
        # Rank 2: one term for c0, one for c1
        terms.append((
            [(U(0,0), 1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(0,0), 1), (W(1,0), 1)]
        ))
        terms.append((
            [(U(0,1), 1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,0), 1)]
        ))
    
    def add_mat_colvec_2x2x1(a_rows, c_cols, d_cols):
        """2x2x1: A is 2x2, B is 2x1, C is 2x1. Rank 2."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0 = d_cols[0]
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[la]
        
        # C[a0, d0] = A[a0,c0]*B[c0,d0] + A[a0,c1]*B[c1,d0]
        # C[a1, d0] = A[a1,c0]*B[c0,d0] + A[a1,c1]*B[c1,d0]
        # Rank 2: one term for c0, one for c1
        terms.append((
            [(U(0,0), 1), (U(1,0), 1)],
            [(V(0,0), 1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        terms.append((
            [(U(0,1), 1), (U(1,1), 1)],
            [(V(1,0), 1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
    
    def add_scalar_row_1x1x2(a_rows, c_cols, d_cols):
        """1x1x2: A is 1x1, B is 1x2, C is 1x2. Rank 1."""
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[0]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        
        # C[0, d0] = A[0,c0]*B[c0,d0]
        # C[0, d1] = A[0,c0]*B[c0,d1]
        terms.append((
            [(U(0,0), 1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(0,0), 1), (W(1,0), 1)]
        ))
    
    def add_scalar_col_2x1x1(a_rows, c_cols, d_cols):
        """2x1x1: A is 2x1, B is 1x1, C is 2x1. Rank 1."""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0 = d_cols[0]
        
        def U(la, lc): return a_rows[la]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[la]
        
        # C[a0, d0] = A[a0,c0]*B[c0,d0]
        # C[a1, d0] = A[a1,c0]*B[c0,d0]
        terms.append((
            [(U(0,0), 1), (U(1,0), 1)],
            [(V(0,0), 1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
    
    def add_scalar_row_1x2x1(a_rows, c_cols, d_cols):
        """1x2x1: A is 1x2, B is 2x1, C is 1x1. Rank 2."""
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0 = d_cols[0]
        
        def U(la, lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[0]
        
        # C[0, d0] = A[0,c0]*B[c0,d0] + A[0,c1]*B[c1,d0]
        terms.append((
            [(U(0,0), 1)],
            [(V(0,0), 1)],
            [(W(0,0), 1)]
        ))
        terms.append((
            [(U(0,1), 1)],
            [(V(1,0), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_scalar_1x1x1(a_rows, c_cols, d_cols):
        """1x1x1: Rank 1."""
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0 = d_cols[0]
        
        def U(la, lc): return a_rows[0]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[0]
        
        terms.append((
            [(U(0,0), 1)],
            [(V(0,0), 1)],
            [(W(0,0), 1)]
        ))
    
    # Iterate over all block combinations
    for ni, (ns, ne) in enumerate(n_ranges):
        for mi, (ms, me) in enumerate(m_ranges):
            for pi, (ps, pe) in enumerate(p_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_outer_product_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_rowvec_mat_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_mat_colvec_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_scalar_row_1x1x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 1:
                    add_scalar_col_2x1x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 1:
                    add_scalar_row_1x2x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 1:
                    add_scalar_1x1x1(a_rows, c_cols, d_cols)
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

# Verify correctness
def _verify():
    n, m, p = DIMENSIONS
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-10:
                    return False, (i, j, k, total)
    return True, None

_ok, _err = _verify()
if not _ok:
    print(f"Verification failed: {_err}")
# EVOLVE-BLOCK-END

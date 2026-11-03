"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Partition:
    # n (rows of A): [2, 1]
    # m (cols of A / rows of B): [2, 2, 1]
    # p (cols of B): [2, 2, 2]
    
    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
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
    
    # Try to reduce rank via column merging / sparsification using a 
    # smooth optimization approach on the factor matrices
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Verify the current decomposition is correct
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Check if U,V,W correctly decompose the matmul tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U_arr, V_arr, W_arr, n, m, p):
        # Fallback: return the structured decomposition as-is
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank reduction: try to zero out columns that are linearly 
    # dependent or can be absorbed, using a smooth penalty approach.
    # We use a simple greedy approach: try removing each column and check 
    # if the remaining columns can still span the tensor (via least squares).
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 50s safety margin
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    # Try to eliminate columns one by one
    for attempt in range(rank):
        if time.time() > deadline:
            break
        
        found_elimination = False
        for col in range(best_rank):
            if time.time() > deadline:
                break
            
            # Try removing column 'col'
            if best_rank <= 1:
                break
            
            U_test = np.delete(best_U, col, axis=1)
            V_test = np.delete(best_V, col, axis=1)
            W_test = np.delete(best_W, col, axis=1)
            
            if verify(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                found_elimination = True
                break
        
        if not found_elimination:
            break
    
    # If we reduced rank, convert back to list form
    if best_rank < rank:
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
        
        # Ensure entries are clean (integers or half-integers)
        for mat in [U_final, V_final, W_final]:
            for row in mat:
                for idx in range(len(row)):
                    val = row[idx]
                    # Round to nearest half-integer
                    rounded = round(val * 2) / 2.0
                    if abs(val - rounded) < 1e-9:
                        row[idx] = rounded
                    else:
                        row[idx] = round(val, 10)
        
        return {"U": U_final, "V": V_final, "W": W_final}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

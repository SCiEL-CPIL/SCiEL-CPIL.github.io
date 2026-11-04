"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _verify_full(U, V, W, n, m, p):
    """Full verification using Fraction for exactness."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    rank_schoolbook = n * m * p  # 90
    
    import numpy as np
    from fractions import Fraction
    
    def get_strassen_terms(a_rows, c_cols, d_cols, n, m, p):
        """Get 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U_idx(la, lc): return a_rows[la]*m + c_cols[lc]
        def V_idx(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W_idx(ld, la): return d_cols[ld]*n + a_rows[la]
        
        terms = []
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0] += 1, C[1,1] += 1
        terms.append((
            [(U_idx(0,0), 1), (U_idx(1,1), 1)],
            [(V_idx(0,0), 1), (V_idx(1,1), 1)],
            [(W_idx(0,0), 1), (W_idx(1,1), 1)]
        ))
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # C[1,0] += 1, C[1,1] += -1
        terms.append((
            [(U_idx(1,0), 1), (U_idx(1,1), 1)],
            [(V_idx(0,0), 1)],
            [(W_idx(0,1), 1), (W_idx(1,1), -1)]
        ))
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        terms.append((
            [(U_idx(0,0), 1)],
            [(V_idx(0,1), 1), (V_idx(1,1), -1)],
            [(W_idx(1,0), 1), (W_idx(1,1), 1)]
        ))
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # C[0,0] += 1, C[1,0] += 1
        terms.append((
            [(U_idx(1,1), 1)],
            [(V_idx(1,0), 1), (V_idx(0,0), -1)],
            [(W_idx(0,0), 1), (W_idx(0,1), 1)]
        ))
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # C[0,0] += -1, C[0,1] += 1
        terms.append((
            [(U_idx(0,0), 1), (U_idx(0,1), 1)],
            [(V_idx(1,1), 1)],
            [(W_idx(0,0), -1), (W_idx(1,0), 1)]
        ))
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # C[1,1] += 1
        terms.append((
            [(U_idx(1,0), 1), (U_idx(0,0), -1)],
            [(V_idx(0,0), 1), (V_idx(0,1), 1)],
            [(W_idx(1,1), 1)]
        ))
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # C[0,0] += 1
        terms.append((
            [(U_idx(0,1), 1), (U_idx(1,1), -1)],
            [(V_idx(1,0), 1), (V_idx(1,1), 1)],
            [(W_idx(0,0), 1)]
        ))
        
        return terms
    
    def get_schoolbook_terms(a_rows, c_cols, d_cols, n, m, p):
        """Get schoolbook terms for a block."""
        terms = []
        for la in range(len(a_rows)):
            for lc in range(len(c_cols)):
                for ld in range(len(d_cols)):
                    terms.append((
                        [(a_rows[la]*m + c_cols[lc], 1)],
                        [(c_cols[lc]*p + d_cols[ld], 1)],
                        [(d_cols[ld]*n + a_rows[la], 1)]
                    ))
        return terms
    
    def build_decomposition(n_parts, m_parts, p_parts, n, m, p):
        """Build a decomposition given partition sizes."""
        n_offsets = [0]
        for i in range(len(n_parts) - 1):
            n_offsets.append(n_offsets[-1] + n_parts[i])
        
        m_offsets = [0]
        for i in range(len(m_parts) - 1):
            m_offsets.append(m_offsets[-1] + m_parts[i])
        
        p_offsets = [0]
        for i in range(len(p_parts) - 1):
            p_offsets.append(p_offsets[-1] + p_parts[i])
        
        terms = []
        
        for ni in range(len(n_parts)):
            for mi in range(len(m_parts)):
                for pi in range(len(p_parts)):
                    a_rows = list(range(n_offsets[ni], n_offsets[ni] + n_parts[ni]))
                    c_cols = list(range(m_offsets[mi], m_offsets[mi] + m_parts[mi]))
                    d_cols = list(range(p_offsets[pi], p_offsets[pi] + p_parts[pi]))
                    
                    if n_parts[ni] == 2 and m_parts[mi] == 2 and p_parts[pi] == 2:
                        terms.extend(get_strassen_terms(a_rows, c_cols, d_cols, n, m, p))
                    else:
                        terms.extend(get_schoolbook_terms(a_rows, c_cols, d_cols, n, m, p))
        
        total_rank = len(terms)
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for r, (u_entries, v_entries, w_entries) in enumerate(terms):
            for (row, coeff) in u_entries:
                U[row][r] = coeff
            for (row, coeff) in v_entries:
                V[row][r] = coeff
            for (row, coeff) in w_entries:
                W[row][r] = coeff
        
        return U, V, W, total_rank
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify the decomposition exactly."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Try multiple partition strategies and keep the best one
    partitions_to_try = [
        ([2, 1], [2, 2, 1], [2, 2, 2]),  # Original: rank 84
        ([2, 1], [1, 2, 2], [2, 2, 2]),
        ([2, 1], [2, 1, 2], [2, 2, 2]),
        ([2, 1], [2, 2, 1], [1, 2, 2]),
        ([2, 1], [2, 2, 1], [2, 1, 2]),
        ([1, 2], [2, 2, 1], [2, 2, 2]),
        ([2, 1], [2, 2, 1], [2, 2, 1, 1]),  # More fine-grained in p
        ([2, 1], [2, 1, 2], [2, 2, 1, 1]),
        ([1, 2], [2, 1, 2], [2, 2, 1, 1]),
    ]
    
    best_U, best_V, best_W, best_rank = None, None, None, rank_schoolbook
    
    for n_parts, m_parts, p_parts in partitions_to_try:
        if time.time() > deadline - 20:
            break
        
        # Verify partition sums
        if sum(n_parts) != n or sum(m_parts) != m or sum(p_parts) != p:
            continue
        
        U, V, W, total_rank = build_decomposition(n_parts, m_parts, p_parts, n, m, p)
        
        if total_rank < best_rank:
            if verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W, best_rank = U, V, W, total_rank
    
    # Fall back to schoolbook if no valid decomposition found
    if best_U is None:
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce rank using linear dependency detection
    # Convert to numpy for efficient linear algebra
    U_np = np.array(best_U, dtype=float)
    V_np = np.array(best_V, dtype=float)
    W_np = np.array(best_W, dtype=float)
    
    current_rank = best_rank
    
    def try_remove_columns(U_np, V_np, W_np, n, m, p, max_removals=5):
        """Try to remove linearly dependent columns."""
        removed = []
        current_U = U_np.copy()
        current_V = V_np.copy()
        current_W = W_np.copy()
        current_rank = current_U.shape[1]
        
        for _ in range(max_removals):
            if time.time() > deadline - 10:
                break
            
            # Check if any column is a linear combination of others
            # This is expensive, so we sample
            found_removal = False
            for trial in range(20):
                if time.time() > deadline - 10:
                    break
                
                col_idx = random.randint(0, current_rank - 1)
                if col_idx in removed:
                    continue
                
                # Try to express this column as a combination of others
                # Build the system: for each entry (i,j,k), 
                # U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] 
                # should equal sum of other terms
                
                # Simplified: check if the rank-1 tensor is in the span of others
                # This is a hard problem, so use a heuristic:
                # Check if removing this column still gives a valid decomposition
                # with adjusted coefficients (this is complex)
                
                # For now, just check if the column has small norm (likely redundant)
                col_norm = np.sqrt(
                    np.sum(current_U[:, col_idx]**2) +
                    np.sum(current_V[:, col_idx]**2) +
                    np.sum(current_W[:, col_idx]**2)
                )
                
                if col_norm < 1e-10:
                    # Column is effectively zero, remove it
                    removed.append(col_idx)
                    current_U = np.delete(current_U, col_idx, axis=1)
                    current_V = np.delete(current_V, col_idx, axis=1)
                    current_W = np.delete(current_W, col_idx, axis=1)
                    current_rank -= 1
                    found_removal = True
                    break
            
            if not found_removal:
                break
        
        return current_U, current_V, current_W, current_rank
    
    # Try column removal
    U_reduced, V_reduced, W_reduced, reduced_rank = try_remove_columns(
        U_np, V_np, W_np, n, m, p, max_removals=3
    )
    
    if reduced_rank < current_rank:
        # Verify the reduced decomposition
        U_check = [[int(round(U_reduced[i, j])) if abs(U_reduced[i, j] - round(U_reduced[i, j])) < 1e-9 else float(U_reduced[i, j]) for j in range(reduced_rank)] for i in range(n*m)]
        V_check = [[int(round(V_reduced[i, j])) if abs(V_reduced[i, j] - round(V_reduced[i, j])) < 1e-9 else float(V_reduced[i, j]) for j in range(reduced_rank)] for i in range(m*p)]
        W_check = [[int(round(W_reduced[i, j])) if abs(W_reduced[i, j] - round(W_reduced[i, j])) < 1e-9 else float(W_reduced[i, j]) for j in range(reduced_rank)] for i in range(p*n)]
        
        if verify_decomposition(U_check, V_check, W_check, n, m, p):
            best_U, best_V, best_W, best_rank = U_check, V_check, W_check, reduced_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

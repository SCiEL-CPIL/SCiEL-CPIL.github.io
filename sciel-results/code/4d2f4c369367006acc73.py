"""
Strategy: 
1. Base: 44-term decomposition (4 Strassen blocks for 2x2x8 + schoolbook for 2x1x8).
2. Structural change: Use a UNIFIED rank-reduction search. Instead of fixing the 
   Strassen structure, parameterize all 44 terms and use iterative Gaussian 
   elimination to find if any term can be expressed as a linear combination of 
   others (in the tensor space). If so, eliminate it and reduce rank.
3. Additionally, try a GREEDY TERM SWAP: for each pair of terms, check if their 
   sum can be replaced by a single term (i.e., if the sum of two rank-1 tensors 
   has border rank 1, which is rare, but we check via SVD).
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build 44-term decomposition
    rank_44 = 44
    U_44 = [[0] * rank_44 for _ in range(nm)]
    V_44 = [[0] * rank_44 for _ in range(mp)]
    W_44 = [[0] * rank_44 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(n):
        for k in range(p):
            U_44[i * m + 2][col] = 1
            V_44[2 * p + k][col] = 1
            W_44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Verify 44-term
    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    if not _validate(candidate_44):
        return _schoolbook()

    best_rank = 44
    best_candidate = candidate_44

    # Structural change: Try to reduce rank by eliminating linearly dependent terms.
    # Use exact Fraction arithmetic to find if any term is a linear combination of others.
    
    # For each candidate term to eliminate, try to express it as a linear combination
    # of a subset of other terms. Use a greedy approach: try eliminating terms one at a time,
    # checking exact linear dependence via Gaussian elimination on a sufficient set of entries.
    
    def _try_eliminate_term(terms, U_mat, V_mat, W_mat, term_to_eliminate):
        """Try to eliminate term_to_eliminate by expressing it as combo of others.
        Returns (coefficients, success) where coefficients maps term_index -> Fraction."""
        if not _elapsed_ok():
            return None, False
        
        remaining = [t for t in range(len(terms)) if t != term_to_eliminate and t not in terms]
        if not remaining:
            return None, False
        
        # We need to find coefficients c_s such that for all (i,a,k):
        # U[i,a,term]*V[a,k,term]*W[k,i,term] = sum_s c_s * U[i,a,s]*V[a,k,s]*W[k,i,s]
        # 
        # This is a linear system in the c_s. Use a subset of (i,a,k) entries to form equations.
        # We need at least len(remaining) independent equations.
        
        # Collect equations: for each (i,a,k), we get one linear equation in c_s
        # coefficient of c_s is U[i,a,s]*V[a,k,s]*W[k,i,s]
        # RHS is U[i,a,term]*V[a,k,term]*W[k,i,term]
        
        # Build the system using Fraction for exactness
        # Use all possible (i,a,k) but stop when we have enough rows
        all_ijk = [(i, a, k) for i in range(n) for a in range(m) for k in range(p)]
        
        # Build augmented matrix as list of lists of Fractions
        # Each row: [coeff_s1, coeff_s2, ..., coeff_sk, rhs]
        rows = []
        for (i, a, k) in all_ijk:
            rhs = Fraction(U_mat[i * m + a][term_to_eliminate]) * \
                   Fraction(V_mat[a * p + k][term_to_eliminate]) * \
                   Fraction(W_mat[k * n + i][term_to_eliminate])
            if rhs == 0:
                # This entry doesn't constrain the solution (0 = sum c_s * 0)
                # Only useful if all coeffs are also 0
                continue
            row = []
            for s in remaining:
                coeff = Fraction(U_mat[i * m + a][s]) * \
                         Fraction(V_mat[a * p + k][s]) * \
                         Fraction(W_mat[k * n + i][s])
                row.append(coeff)
            row.append(rhs)
            rows.append(row)
            
            if len(rows) >= len(remaining) + 5:  # Have enough equations
                break
        
        if not rows:
            return None, False
        
        # Solve the linear system using Gaussian elimination with Fractions
        # Matrix is rows x (len(remaining) + 1)
        num_vars = len(remaining)
        num_eqs = len(rows)
        
        # Convert to 2D list for Gaussian elimination
        mat = [row[:] for row in rows]
        
        # Gaussian elimination
        pivot_rows = []
        for col in range(num_vars):
            # Find pivot
            pivot = -1
            for row_idx in range(len(pivot_rows), num_eqs):
                if mat[row_idx][col] != 0:
                    pivot = row_idx
                    break
            if pivot == -1:
                continue  # Skip this column (free variable or inconsistent)
            
            # Swap to pivot position
            if pivot != len(pivot_rows):
                mat[len(pivot_rows)], mat[pivot] = mat[pivot], mat[len(pivot_rows)]
            
            pivot_rows.append(col)
            pivot_val = mat[len(pivot_rows) - 1][col]
            
            # Eliminate below
            for row_idx in range(len(pivot_rows), num_eqs):
                factor = mat[row_idx][col] / pivot_val
                if factor != 0:
                    for j in range(col, num_vars + 1):
                        mat[row_idx][j] -= factor * mat[len(pivot_rows) - 1][j]
        
        # Check consistency: any row with all zeros in coeffs but non-zero rhs?
        for row_idx in range(len(pivot_rows), num_eqs):
            if all(mat[row_idx][j] == 0 for j in range(num_vars)) and mat[row_idx][num_vars] != 0:
                return None, False  # Inconsistent
        
        # Extract solution (assume unique solution for simplicity)
        if len(pivot_rows) < num_vars:
            # Underdetermined - try setting free variables to 0
            # This is a heuristic
            pass
        
        # Back substitution
        coeffs = [Fraction(0)] * num_vars
        pivot_row_map = {pivot_rows[i]: i for i in range(len(pivot_rows))}
        
        for i in range(len(pivot_rows) - 1, -1, -1):
            col = pivot_rows[i]
            row = mat[i]
            rhs = row[num_vars]
            for j in range(col + 1, num_vars):
                if row[j] != 0:
                    rhs -= row[j] * coeffs[j]
            if row[col] != 0:
                coeffs[col] = rhs / row[col]
        
        # Verify the solution
        for (i, a, k) in all_ijk:
            lhs = Fraction(U_mat[i * m + a][term_to_eliminate]) * \
                   Fraction(V_mat[a * p + k][term_to_eliminate]) * \
                   Fraction(W_mat[k * n + i][term_to_eliminate])
            rhs = Fraction(0)
            for idx, s in enumerate(remaining):
                if coeffs[idx] != 0:
                    rhs += coeffs[idx] * Fraction(U_mat[i * m + a][s]) * \
                           Fraction(V_mat[a * p + k][s]) * \
                           Fraction(W_mat[k * n + i][s])
            if lhs != rhs:
                return None, False
        
        # Check if coefficients are half-integers (denominator divides 2)
        for c in coeffs:
            if c.denominator > 2:
                return None, False
        
        # Success! Build coefficient mapping
        coeff_map = {}
        for idx, s in enumerate(remaining):
            if coeffs[idx] != 0:
                coeff_map[s] = coeffs[idx]
        
        return coeff_map, True
    
    # Try to eliminate terms one by one
    eliminated_set = set()
    U_cur = [row[:] for row in U_44]
    V_cur = [row[:] for row in V_44]
    W_cur = [row[:] for row in W_44]
    
    # Try each term in order of "simplicity" (fewer non-zero entries first)
    term_complexity = []
    for r in range(rank_44):
        nz = sum(1 for row in U_cur for x in row if x != 0) + \
             sum(1 for row in V_cur for x in row if x != 0) + \
             sum(1 for row in W_cur for x in row if x != 0)
        term_complexity.append((nz, r))
    term_complexity.sort()
    
    for _, r in term_complexity:
        if not _elapsed_ok():
            break
        if r in eliminated_set:
            continue
        
        coeff_map, success = _try_eliminate_term(
            eliminated_set, U_cur, V_cur, W_cur, r
        )
        
        if success and coeff_map:
            # Eliminate term r: for each remaining term s, add coeff_map[s] * term_s to term r
            # Actually, we're saying term_r = sum coeff_map[s] * term_s
            # So we can just remove term r from the decomposition
            eliminated_set.add(r)
            best_rank = rank_44 - len(eliminated_set)
            
            # Rebuild the candidate without eliminated terms
            active_terms = [t for t in range(rank_44) if t not in eliminated_set]
            U_new = [[U_cur[i * m + a][t] for t in active_terms] for i in range(n) for a in range(m)]
            V_new = [[V_cur[a * p + k][t] for t in active_terms] for a in range(m) for k in range(p)]
            W_new = [[W_cur[k * n + i][t] for t in active_terms] for k in range(p) for i in range(n)]
            
            candidate = {"U": U_new, "V": V_new, "W": W_new}
            if _validate(candidate):
                best_candidate = candidate
            
            if not _elapsed_ok():
                break
    
    if not _elapsed_ok():
        return best_candidate
    
    return best_candidate

# EVOLVE-BLOCK-END

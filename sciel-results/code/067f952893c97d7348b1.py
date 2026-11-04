"""
Strategy: 
1. Base: 44-term decomposition (4 Strassen blocks for 2x2x8 + schoolbook for 2x1x8).
2. Rank reduction: Use exact Gaussian elimination to find if any term is a linear 
   combination of others in the tensor space. If so, eliminate it.
3. Try multiple elimination orders and verify each candidate.
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

    # Precompute all (i,a,k) indices
    all_ijk = [(i, a, k) for i in range(n) for a in range(m) for k in range(p)]

    def _gauss_solve(mat, num_vars):
        """Solve linear system with Fraction arithmetic. Returns solution or None."""
        num_rows = len(mat)
        if num_rows == 0:
            return None
        mat = [row[:] for row in mat]
        pivot_cols = []
        
        for col in range(num_vars):
            pivot = -1
            for row_idx in range(len(pivot_cols), num_rows):
                if mat[row_idx][col] != 0:
                    pivot = row_idx
                    break
            if pivot == -1:
                continue
            if pivot != len(pivot_cols):
                mat[len(pivot_cols)], mat[pivot] = mat[pivot], mat[len(pivot_cols)]
            pivot_cols.append(col)
            pivot_val = mat[len(pivot_cols) - 1][col]
            for row_idx in range(len(pivot_cols), num_rows):
                factor = mat[row_idx][col] / pivot_val
                if factor != 0:
                    for j in range(col, num_vars + 1):
                        mat[row_idx][j] -= factor * mat[len(pivot_cols) - 1][j]
        
        # Check consistency
        for row_idx in range(len(pivot_cols), num_rows):
            if all(mat[row_idx][j] == 0 for j in range(num_vars)) and mat[row_idx][num_vars] != 0:
                return None
        
        # Back substitution
        coeffs = [Fraction(0)] * num_vars
        for i in range(len(pivot_cols) - 1, -1, -1):
            col = pivot_cols[i]
            row = mat[i]
            rhs = row[num_vars]
            for j in range(col + 1, num_vars):
                if row[j] != 0:
                    rhs -= row[j] * coeffs[j]
            if row[col] != 0:
                coeffs[col] = rhs / row[col]
        return coeffs

    def _try_eliminate(terms_to_eliminate, active_terms, U_mat, V_mat, W_mat):
        """Try to eliminate terms_to_eliminate by expressing each as combo of active_terms.
        Returns (coeff_dict, success) where coeff_dict maps term->list of (active_term, coeff)."""
        if not _elapsed_ok():
            return None, False
        
        # For each term to eliminate, find coefficients
        all_coeffs = {}
        for term in terms_to_eliminate:
            # Build linear system
            rows = []
            for (i, a, k) in all_ijk:
                rhs = Fraction(U_mat[i * m + a][term]) * \
                       Fraction(V_mat[a * p + k][term]) * \
                       Fraction(W_mat[k * n + i][term])
                if rhs == 0:
                    # Check if all active terms also give 0 here
                    all_zero = True
                    for s in active_terms:
                        c = Fraction(U_mat[i * m + a][s]) * \
                             Fraction(V_mat[a * p + k][s]) * \
                             Fraction(W_mat[k * n + i][s])
                        if c != 0:
                            all_zero = False
                            break
                    if not all_zero:
                        # This entry constrains: 0 = sum c_s * ...
                        row = []
                        for s in active_terms:
                            coeff = Fraction(U_mat[i * m + a][s]) * \
                                     Fraction(V_mat[a * p + k][s]) * \
                                     Fraction(W_mat[k * n + i][s])
                            row.append(coeff)
                        row.append(Fraction(0))
                        rows.append(row)
                    continue
                row = []
                for s in active_terms:
                    coeff = Fraction(U_mat[i * m + a][s]) * \
                             Fraction(V_mat[a * p + k][s]) * \
                             Fraction(W_mat[k * n + i][s])
                    row.append(coeff)
                row.append(rhs)
                rows.append(row)
                if len(rows) >= len(active_terms) + 3:
                    break
            
            if not rows:
                return None, False
            
            coeffs = _gauss_solve(rows, len(active_terms))
            if coeffs is None:
                return None, False
            
            # Verify solution
            for (i, a, k) in all_ijk:
                lhs = Fraction(U_mat[i * m + a][term]) * \
                       Fraction(V_mat[a * p + k][term]) * \
                       Fraction(W_mat[k * n + i][term])
                rhs = Fraction(0)
                for idx, s in enumerate(active_terms):
                    if coeffs[idx] != 0:
                        rhs += coeffs[idx] * Fraction(U_mat[i * m + a][s]) * \
                               Fraction(V_mat[a * p + k][s]) * \
                               Fraction(W_mat[k * n + i][s])
                if lhs != rhs:
                    return None, False
            
            # Check half-integer constraint
            for c in coeffs:
                if c.denominator > 2:
                    return None, False
            
            all_coeffs[term] = coeffs
        
        return all_coeffs, True

    # Try eliminating single terms first
    U_cur = [row[:] for row in U_44]
    V_cur = [row[:] for row in V_44]
    W_cur = [row[:] for row in W_44]
    
    # Sort terms by number of non-zero entries (simpler terms first)
    term_nz = []
    for r in range(rank_44):
        nz = 0
        for row in U_cur:
            if row[r] != 0:
                nz += 1
        for row in V_cur:
            if row[r] != 0:
                nz += 1
        for row in W_cur:
            if row[r] != 0:
                nz += 1
        term_nz.append((nz, r))
    term_nz.sort()
    
    eliminated = set()
    for _, r in term_nz:
        if not _elapsed_ok():
            break
        if r in eliminated:
            continue
        
        active = [t for t in range(rank_44) if t != r and t not in eliminated]
        if not active:
            continue
        
        _, success = _try_eliminate([r], active, U_cur, V_cur, W_cur)
        if success:
            eliminated.add(r)
            best_rank = rank_44 - len(eliminated)
            
            # Build reduced candidate
            active_terms = [t for t in range(rank_44) if t not in eliminated]
            U_new = [[U_cur[i * m + a][t] for t in active_terms] for i in range(n) for a in range(m)]
            V_new = [[V_cur[a * p + k][t] for t in active_terms] for a in range(m) for k in range(p)]
            W_new = [[W_cur[k * n + i][t] for t in active_terms] for k in range(p) for i in range(n)]
            
            candidate = {"U": U_new, "V": V_new, "W": W_new}
            if _validate(candidate):
                best_candidate = candidate
            else:
                # Rollback
                eliminated.discard(r)
                best_rank = rank_44 - len(eliminated)
    
    if not _elapsed_ok():
        return best_candidate
    
    return best_candidate

# EVOLVE-BLOCK-END

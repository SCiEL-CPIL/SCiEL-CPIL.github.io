"""
Rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.

Strategy:
1. Build a block-Strassen + schoolbook decomposition (rank 99 baseline).
2. Attempt rank reduction via:
   a. Removing zero terms.
   b. Checking if any single term can be removed (linear dependency).
   c. Trying to merge pairs of terms that are linearly dependent in a useful way.
3. Use exact rational arithmetic (Fraction) for correctness.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [3, 5, 7]


def _strassen_block(U, V, W, n, m, p, ia, ja, ka, offset, sign=1):
    """
    Insert 7 Strassen rank-1 terms at column index starting at `offset`
    for 2x2x2 subblock at (ia, ja, ka).
    Returns next free column index.
    """
    col = offset

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[ia*m + ja][col] += sign
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + ka][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[ka*n + ia][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M2 = (A10+A11)B00 -> C10 += M2
    U[(ia+1)*m + ja][col] += sign
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + ka][col] += sign
    W[ka*n + (ia+1)][col] += sign
    col += 1

    # M3 = A00(B10-B11) -> C11 += M3
    U[ia*m + ja][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    V[(ja+1)*p + ka][col] -= sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M4 = A11(B01-B10) -> C00 += M4
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + (ka+1)][col] += sign
    V[ja*p + ka][col] -= sign
    W[ka*n + ia][col] += sign
    col += 1

    # M5 = (A00+A01)B11 -> C01 += M5, C11 -= M5
    U[ia*m + ja][col] += sign
    U[ia*m + (ja+1)][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    W[(ka+1)*n + ia][col] -= sign
    col += 1

    # M6 = (A10-A00)(B00+B10) -> C10 += M6
    U[(ia+1)*m + ja][col] += sign
    U[ia*m + ja][col] -= sign
    V[ja*p + ka][col] += sign
    V[ja*p + (ka+1)][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M7 = (A01-A11)(B01+B11) -> C00 -= M7, C11 += M7
    U[(ia+1)*m + (ja+1)][col] += sign
    U[ia*m + (ja+1)][col] -= sign
    V[(ja+1)*p + ka][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[ka*n + ia][col] -= sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    return col


def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 115
    t0 = time.time()

    def get_block_options(d):
        """Get all valid sets of non-overlapping 2-blocks for dimension d"""
        results = []
        for nb in range(0, d // 2 + 1):
            starts = list(range(d - 1))
            for combo in combinations(starts, nb):
                valid = True
                for i in range(len(combo) - 1):
                    if combo[i] + 1 >= combo[i+1]:
                        valid = False
                        break
                if not valid:
                    continue
                covered = set()
                for s in combo:
                    covered.add(s)
                    covered.add(s + 1)
                uncovered = d - len(covered)
                blocks = [(s, s+1) for s in combo]
                results.append((blocks, covered, uncovered))
        return results

    n_opts = get_block_options(n)
    m_opts = get_block_options(m)
    p_opts = get_block_options(p)

    best_config = None
    best_rank = float('inf')

    for nb, n_cov, n_unc in n_opts:
        for mb, m_cov, m_unc in m_opts:
            for pb, p_cov, p_unc in p_opts:
                num_strassen = len(nb) * len(mb) * len(pb)
                rank_strassen = num_strassen * 7

                num_schoolbook = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i not in n_cov or j not in m_cov or k not in p_cov:
                                num_schoolbook += 1

                total_rank = rank_strassen + num_schoolbook
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_config = (nb, mb, pb, n_cov, m_cov, p_cov)

    if best_config is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    nb, mb, pb, n_cov, m_cov, p_cov = best_config
    num_strassen = len(nb) * len(mb) * len(pb)
    rank_strassen = num_strassen * 7

    num_schoolbook = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i not in n_cov or j not in m_cov or k not in p_cov:
                    num_schoolbook += 1

    total_rank = rank_strassen + num_schoolbook

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    for (a, b) in nb:
        for (c, d) in mb:
            for (e, f) in pb:
                col = _strassen_block(U, V, W, n, m, p, a, c, e, col)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i not in n_cov or j not in m_cov or k not in p_cov:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    def verify_float(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify_float(U, V, W, total_rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Convert to Fraction for exact arithmetic
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    rank = total_rank

    def verify_exact(Uf, Vf, Wf, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for t in range(rank):
                        s += Uf[i*m+j][t] * Vf[j*p+k][t] * Wf[k*n+i][t]
                    if s != 1:
                        return False
        return True

    if not verify_exact(Uf, Vf, Wf, rank):
        return {"U": U, "V": V, "W": W}

    def remove_zero_terms(Uf, Vf, Wf, rank, deadline):
        """Remove terms that are entirely zero in one component."""
        changed = True
        while changed and time.time() < deadline - 20:
            changed = False
            for t in range(rank):
                u_zero = all(Uf[i*m+j][t] == 0 for i in range(n) for j in range(m))
                v_zero = all(Vf[j*p+k][t] == 0 for j in range(m) for k in range(p))
                w_zero = all(Wf[k*n+i][t] == 0 for k in range(p) for i in range(n))
                if u_zero or v_zero or w_zero:
                    Uf = [row[:t] + row[t+1:] for row in Uf]
                    Vf = [row[:t] + row[t+1:] for row in Vf]
                    Wf = [row[:t] + row[t+1:] for row in Wf]
                    rank -= 1
                    changed = True
                    break
        return Uf, Vf, Wf, rank

    def try_remove_single_term(Uf, Vf, Wf, rank, deadline):
        """
        Try to remove a single term if the remaining terms still produce
        the correct tensor. This works if the term is linearly dependent
        on the others in a way that its contribution is already captured.
        """
        for t in range(rank):
            if time.time() > deadline - 15:
                break
            
            # Try removing term t
            Uf_test = [row[:t] + row[t+1:] for row in Uf]
            Vf_test = [row[:t] + row[t+1:] for row in Vf]
            Wf_test = [row[:t] + row[t+1:] for row in Wf]
            
            # Check if the remaining terms still produce the correct tensor
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for tt in range(rank - 1):
                            s += Uf_test[i*m+j][tt] * Vf_test[j*p+k][tt] * Wf_test[k*n+i][tt]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if time.time() > deadline - 15:
                    break
                if not valid:
                    break
            
            if valid:
                return Uf_test, Vf_test, Wf_test, rank - 1
        
        return Uf, Vf, Wf, rank

    def try_merge_proportional_terms(Uf, Vf, Wf, rank, deadline):
        """
        Try to merge two terms that are proportional in all three components.
        If U[t2] = alpha * U[t1], V[t2] = alpha * V[t1], W[t2] = alpha * W[t1],
        then term t2 = alpha^3 * term t1, so we can't just remove it.
        But if we find such pairs, we might be able to restructure.
        For now, this is a placeholder for more sophisticated merging.
        """
        # This is complex; skip for now and rely on single-term removal
        return Uf, Vf, Wf, rank

    # First, remove zero terms
    Uf, Vf, Wf, rank = remove_zero_terms(Uf, Vf, Wf, deadline)
    
    # Then, try to remove single terms (linear dependencies)
    if time.time() < deadline - 30:
        Uf, Vf, Wf, rank = try_remove_single_term(Uf, Vf, Wf, deadline)
    
    # Try again after removing one term
    if time.time() < deadline - 20:
        Uf, Vf, Wf, rank = remove_zero_terms(Uf, Vf, Wf, deadline)
        Uf, Vf, Wf, rank = try_remove_single_term(Uf, Vf, Wf, deadline)
    
    # Final verification
    if verify_exact(Uf, Vf, Wf, rank):
        def to_clean_list(flist):
            result = []
            for row in flist:
                crow = []
                for val in row:
                    if val.denominator == 1:
                        crow.append(int(val))
                    else:
                        crow.append(float(val))
                result.append(crow)
            return result
        
        return {"U": to_clean_list(Uf), "V": to_clean_list(Vf), "W": to_clean_list(Wf)}
    
    return {"U": U, "V": V, "W": W}

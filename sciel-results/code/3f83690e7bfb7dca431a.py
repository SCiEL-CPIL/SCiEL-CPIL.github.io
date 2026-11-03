"""Combined program: sub-problem decomposition + exact randomized solver."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Fallback: schoolbook for the full problem (always valid, rank n*m*p)
    schoolbook = _schoolbook(n, m, p)

    # Sub-problem: M_{n, m, p_sub} where p_sub divides p
    # rank(M_{n,m,p}) = (p/p_sub) * rank(M_{n,m,p_sub})
    p_sub = 2
    if p % p_sub != 0:
        return schoolbook
    repeats = p // p_sub
    sub_schoolbook_rank = n * m * p_sub
    min_possible = max(n * m, m * p_sub, p_sub * n)

    best = None
    best_rank = None
    for r in range(min_possible, sub_schoolbook_rank + 1):
        if time.time() > deadline:
            break
        # more attempts for smaller ranks (harder, but higher payoff)
        attempts = 20000 if r <= 8 else 8000
        sol = _solve_sub(n, m, p_sub, r, attempts, deadline)
        if sol is not None:
            best = sol
            best_rank = r
            break  # smallest r found; stop (ranks tried ascending)

    if best is None:
        return schoolbook

    U_sub, V_sub, W_sub = best
    full_rank = best_rank * repeats
    U_full = [[0] * full_rank for _ in range(n * m)]
    V_full = [[0] * full_rank for _ in range(m * p)]
    W_full = [[0] * full_rank for _ in range(p * n)]

    for rep in range(repeats):
        offset = rep * best_rank
        # U: row index i*m + j (same in sub)
        for i in range(n):
            for j in range(m):
                sidx = i * m + j
                for c in range(best_rank):
                    U_full[sidx][offset + c] = U_sub[sidx][c]
        # V: row index j*p + k ; sub k -> full k = rep*p_sub + k_sub
        for j in range(m):
            for k_sub in range(p_sub):
                sidx = j * p_sub + k_sub
                k_full = rep * p_sub + k_sub
                fidx = j * p + k_full
                for c in range(best_rank):
                    V_full[fidx][offset + c] = V_sub[sidx][c]
        # W: row index k*n + i ; sub k -> full k = rep*p_sub + k_sub
        for k_sub in range(p_sub):
            for i in range(n):
                sidx = k_sub * n + i
                k_full = rep * p_sub + k_sub
                fidx = k_full * n + i
                for c in range(best_rank):
                    W_full[fidx][offset + c] = W_sub[sidx][c]

    result = {"U": U_full, "V": V_full, "W": W_full}
    # Final safety verification of the full decomposition (cheap: 2*3*6*rank terms)
    if not _verify_full(n, m, p, U_full, V_full, W_full):
        return schoolbook
    return result


def _schoolbook(n, m, p):
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


def _verify_full(n, m, p, U, V, W):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                urow = U[i * m + j]
                vrow = V[j * p + k]
                wcol_i = i
                for c in range(len(urow)):
                    s += urow[c] * vrow[c] * W[k * n + wcol_i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _solve_sub(n, m, p, r, attempts, deadline):
    """
    Find U (n*m x r), V (m*p x r), W (p*n x r) with half-integer entries such that
    for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    Strategy: fix random U,V, solve for W per (k,i) via least squares, snap to
    half-integers, verify exactly. Returns (U,V,W) or None.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    # Precompute, for each (k,i), the list of (j, c) coefficient pairs:
    # coeff = U[i*m+j,c] * V[j*p+k,c]; W row index = k*n+i
    # We'll build coefficient matrices lazily per attempt.

    for _ in range(attempts):
        if time.time() > deadline:
            return None
        # Random half-integer-ish U, V. Use small integer entries (will snap W to halves).
        U = np.random.randint(-2, 3, size=(nm, r)).astype(float)
        V = np.random.randint(-2, 3, size=(mp, r)).astype(float)
        # Avoid degenerate columns
        if np.linalg.matrix_rank(U) < r or np.linalg.matrix_rank(V) < r:
            continue

        # Build coefficient matrix for each (k,i): rows = (j,c) pairs, one equation each? No.
        # For fixed (k,i): unknowns are W[k*n+i, c] for c in 0..r-1.
        # Equations: for each j: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # That's m equations in r unknowns. Need r <= m to have a solution generally.
        # m=3, so r must be <= 3 for this to work. That's too restrictive.
        # Instead, treat all (k,i) W rows jointly is over-constrained.
        # Better: solve the full 3-tensor via per-(k,i) but with r <= m.
        # Since r can exceed m, use a different approach: solve for W using the
        # full system across all (k,i,j) as a linear system in all W entries.
        # Number of W unknowns = p*n*r. Equations = n*m*p. For r=6: unknowns=2*6*6=72,
        # equations=36. Underdetermined -> many solutions; pick min-norm then snap.
        # Build A (n*m*p x p*n*r) and b (n*m*p).
        # This is feasible for small sizes.
        num_eq = n * m * p
        num_unk = p * n * r
        if num_unk > 4000 or num_eq > 4000:
            # too large for this approach; skip
            continue
        A = np.zeros((num_eq, num_unk))
        b = np.ones(num_eq)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        coeff = U[i * m + j, c] * V[j * p + k, c]
                        if coeff != 0:
                            unk = (k * n + i) * r + c
                            A[eq, unk] = coeff
                    eq += 1
        # Solve min-norm least squares
        try:
            sol, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # Snap to half-integers
        W_flat = np.round(sol * 2) / 2.0
        # Verify
        W = W_flat.reshape(p * n, r)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return (U_list, V_list, W_list)
    return None
# EVOLVE-BLOCK-END

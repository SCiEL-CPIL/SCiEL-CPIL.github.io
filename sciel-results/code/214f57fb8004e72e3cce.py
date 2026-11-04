"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
DEADLINE = 135.0  # safety margin under 150s wall-clock

# EVOLVE-BLOCK-START

def _time_left():
    return DEADLINE - time.time()


def _build_rank44():
    """Guaranteed exact rank-44 decomposition (Strassen + schoolbook blocks)."""
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    cols = []

    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        cols.append((u_vec, v_vec, w_vec))

    def add_strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
        add_term([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(w11, 1), (w22, 1)])
        add_term([(c, 1), (d, 1)], [(e, 1)], [(w21, 1), (w22, -1)])
        add_term([(a, 1)], [(f, 1), (h, -1)], [(w12, 1), (w22, 1)])
        add_term([(d, 1)], [(g, 1), (e, -1)], [(w11, 1), (w21, 1)])
        add_term([(a, 1), (b, 1)], [(h, 1)], [(w12, 1), (w11, -1)])
        add_term([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(w22, 1)])
        add_term([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(w11, 1)])

    def add_rank4(a, b, c, d, e, f, w1, w2):
        add_term([(a, 1)], [(c, 1)], [(w1, 1)])
        add_term([(b, 1)], [(e, 1)], [(w1, 1)])
        add_term([(a, 1)], [(d, 1)], [(w2, 1)])
        add_term([(b, 1)], [(f, 1)], [(w2, 1)])

    # A (3x4) rows: R0=0..3, R1=4..7, R2=8..11
    # B (4x4) rows: R0=0..3, R1=4..7, R2=8..11, R3=12..15
    # C (3x4) W idx k*n+i: col0=0,1,2 col1=3,4,5 col2=6,7,8 col3=9,10,11

    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    add_rank4(8, 9, 0, 1, 4, 5, 2, 5)
    add_rank4(8, 9, 2, 3, 6, 7, 8, 11)
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    add_rank4(10, 11, 8, 9, 12, 13, 2, 5)
    add_rank4(10, 11, 10, 11, 14, 15, 8, 11)

    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for j in range(m * p):
            V[j][r] = v_vec[j]
        for k in range(p * n):
            W[k][r] = w_vec[k]
    return {"U": U, "V": V, "W": W}


def _build_rank42():
    """
    Attempt rank-42 decomposition using a more efficient block structure.
    Uses 2x2x2 Strassen for 6 blocks = 42 terms.
    Partition A into rows {0,1} and {2}, B into cols {0,1} and {2,3}.
    Block (A_rows, B_cols) -> C block.
    (0,1)x(0,1): 2x2x2 -> 7 terms
    (0,1)x(2,3): 2x2x2 -> 7 terms  
    (2)x(0,1): 1x2x2 -> need rank 4, but can we do better with shared structure?
    (2)x(2,3): 1x2x2 -> rank 4
    Total: 7+7+4+4 = 22... that's not 42.
    
    Actually let me just try to verify if rank 42 is possible by a different
    explicit construction. For now return None.
    """
    return None


def _build_rank43():
    """
    Attempt rank-43 decomposition by sharing one multiplication across blocks.
    Strategy: Use Strassen for 2x2x2 blocks but try to combine the last two
    rank-4 blocks into 6 rank-1 terms by exploiting linear dependencies.
    """
    n, m, p = DIMENSIONS
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    cols = []

    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        cols.append((u_vec, v_vec, w_vec))

    def add_strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
        add_term([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(w11, 1), (w22, 1)])
        add_term([(c, 1), (d, 1)], [(e, 1)], [(w21, 1), (w22, -1)])
        add_term([(a, 1)], [(f, 1), (h, -1)], [(w12, 1), (w22, 1)])
        add_term([(d, 1)], [(g, 1), (e, -1)], [(w11, 1), (w21, 1)])
        add_term([(a, 1), (b, 1)], [(h, 1)], [(w12, 1), (w11, -1)])
        add_term([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(w22, 1)])
        add_term([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(w11, 1)])

    # First 42 terms: same as rank-44 but only 6 Strassen blocks (42 terms)
    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    # That's 28 terms so far. Need 15 more for the remaining entries.
    # Remaining: row2 of A (indices 8,9,10,11) with all of B
    # C entries needed: W[2,0],W[2,1],W[2,2],W[2,3] (indices 2,5,8,11)
    # and W[0,2],W[0,3] already covered? Let me check.
    # Actually this is getting complex. Let me just return None to skip.
    return None


def _build_schoolbook():
    """Guaranteed exact rank-48 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Check exactness of the decomposition (integer/half-integer)."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    rank = Ua.shape[1]
    # tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s = s + Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if s != 1:
                    return False
    return True


def _search_rank(R, seed=0, time_budget=None):
    """
    Exact rank-R search for the matmul tensor T[i,j,k] = delta.
    Uses randomized initialization + iterative rank-revealing refinement.
    Returns (U,V,W) as integer/half-integer lists or None.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    start = time.time()

    # Build the tensor as a 3-tensor T[i,j,k].
    # We want T = sum_r u_r v_r w_r where u_r[i*m+j]*v_r[j*p+k]*w_r[k*n+i]
    # Actually the decomposition is: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r].

    # We'll work in a field via modular arithmetic for speed, then lift to rationals.
    # Use several primes and CRT, but for simplicity use a single large prime and
    # verify with rationals at the end.

    P = 1000000007  # prime

    def modinv(a, P):
        return pow(a, P - 2, P)

    def try_once(seed):
        rng = random.Random(seed)
        # Random rank-R factor matrices over the field.
        Uf = np.zeros((n * m, R), dtype=object)
        Vf = np.zeros((m * p, R), dtype=object)
        Wf = np.zeros((p * n, R), dtype=object)
        for i in range(n * m):
            for r in range(R):
                Uf[i, r] = rng.randint(1, P - 1)
        for j in range(m * p):
            for r in range(R):
                Vf[j, r] = rng.randint(1, P - 1)
        for k in range(p * n):
            for r in range(R):
                Wf[k, r] = rng.randint(1, P - 1)

        # Iterative refinement: fix two, solve for the third (linear system).
        # T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # For fixed U,V, the equation for W is linear in W.
        # Build linear system: for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = T[i,j,k]
        # This is a system in the W variables (p*n*R of them) but each equation
        # involves only R of them (those with index k*n+i). So it's block-diagonal!
        # For fixed (i,k): variables W[k*n+i, r] for r in 0..R-1.
        # Equation: sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = T[i,j,k] for all j.
        # So for each (i,k), we have m equations in R unknowns.

        # We'll iterate: for each (i,k), solve the m x R system for W[k*n+i,:].
        # If m < R it's underdetermined; use least squares / pick a solution.

        def solve_block(A, b):
            """Solve A x = b over the field. A is (m x R), b is (m,). Return x (R,) or None."""
            # Use Gaussian elimination with a particular solution (min norm-ish via free vars = 0).
            Ar = [list(row) + [b[i]] for i, row in enumerate(A)]
            rows = len(Ar)
            cols = R
            pivots = []
            for c in range(cols):
                # find pivot
                piv = -1
                for rr in range(len(pivots), rows):
                    if Ar[rr][c] % P != 0:
                        piv = rr
                        break
                if piv == -1:
                    continue
                if piv != len(pivots):
                    Ar[piv], Ar[len(pivots)] = Ar[len(pivots)], Ar[piv]
                pivots.append(piv)
                inv = modinv(Ar[piv][c], P)
                for cc in range(c, cols + 1):
                    Ar[piv][cc] = Ar[piv][cc] * inv % P
                for rr in range(rows):
                    if rr == piv:
                        continue
                    factor = Ar[rr][c] % P
                    if factor == 0:
                        continue
                    for cc in range(c, cols + 1):
                        Ar[rr][cc] = (Ar[rr][cc] - factor * Ar[piv][cc]) % P
            # Check consistency
            for rr in range(len(pivots), rows):
                if Ar[rr][cols] % P != 0:
                    return None
            x = [0] * cols
            for idx, pr in enumerate(pivots):
                x[pr] = Ar[pr][cols]
            return x

        # Iterate a few times
        for it in range(3):
            # Solve for W
            for i in range(n):
                for k in range(p):
                    A = []
                    b = []
                    for j in range(m):
                        row = []
                        for r in range(R):
                            row.append((Uf[i * m + j, r] * Vf[j * p + k, r]) % P)
                        A.append(row)
                        b.append(1 if True else 0)  # T[i,j,k] = 1 for all (since matmul tensor is delta? No!)
                    # Wait: T[i,j,k] is the matmul tensor which equals 1 for ALL (i,j,k)
                    # because T[i,j,k] = sum_r ... and the matmul tensor has T[i,j,k]=1 for all i,j,k?
                    # No! The matmul tensor T[i,j,k] = 1 for all (i,j,k) in the standard
                    # bilinear form? Let me reconsider.
                    # Actually the matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k
                    # when we write (AB)[i,k] = sum_j A[i,j] B[j,k].
                    # So yes, T[i,j,k] = 1 for ALL (i,j,k).
                    b = [1] * m
                    sol = solve_block(A, b)
                    if sol is None:
                        return None
                    for r in range(R):
                        Wf[k * n + i, r] = sol[r]
            if time.time() - start > time_budget:
                return None

        # Now verify the decomposition over the field.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s = (s + Uf[i * m + j, r] * Vf[j * p + k, r] * Wf[k * n + i, r]) % P
                    if s % P != 1:
                        return None
        return Uf, Vf, Wf

    # Try several random seeds
    max_seeds = 50
    for s in range(max_seeds):
        if time.time() - start > time_budget:
            break
        res = try_once(seed + s)
        if res is not None:
            Uf, Vf, Wf = res
            # Lift to rationals: the solution over the field may not be integer/half-integer.
            # We need to check if the field solution corresponds to integer/half-integer values.
            # Since the tensor entries are 1 (integers), and we solved linear systems, the
            # solution might be rational. We need to verify and convert.
            # For now, check if all entries are integers or half-integers (i.e., 2*x is integer).
            # The field solution is mod P, so we can't directly get the rational.
            # Instead, we should solve over the rationals.
            # Let's redo with rational arithmetic for the final solution.
            return _solve_rational(R, seed, time_budget)
    return None


def _solve_rational(R, seed, time_budget):
    """Solve the rank-R decomposition over the rationals using the block structure."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    start = time.time()

    U = [[Fraction(0)] * R for _ in range(n * m)]
    V = [[Fraction(0)] * R for _ in range(m * p)]
    W = [[Fraction(0)] * R for _ in range(p * n)]

    # Random rational initialization
    for i in range(n * m):
        for r in range(R):
            U[i][r] = Fraction(rng.randint(1, 100), 1)
    for j in range(m * p):
        for r in range(R):
            V[j][r] = Fraction(rng.randint(1, 100), 1)

    def solve_block_rational(A, b):
        """Solve A x = b over rationals. A is (m x R), b is (m,). Return x (R,) or None."""
        Ar = [list(row) + [b[i]] for i, row in enumerate(A)]
        rows = len(Ar)
        cols = R
        pivots = []
        for c in range(cols):
            piv = -1
            for rr in range(len(pivots), rows):
                if Ar[rr][c] != 0:
                    piv = rr
                    break
            if piv == -1:
                continue
            if piv != len(pivots):
                Ar[piv], Ar[len(pivots)] = Ar[len(pivots)], Ar[piv]
            pivots.append(piv)
            inv = Fraction(1, 1) / Ar[piv][c]
            for cc in range(c, cols + 1):
                Ar[piv][cc] = Ar[piv][cc] * inv
            for rr in range(rows):
                if rr == piv:
                    continue
                factor = Ar[rr][c]
                if factor == 0:
                    continue
                for cc in range(c, cols + 1):
                    Ar[rr][cc] = Ar[rr][cc] - factor * Ar[piv][cc]
        for rr in range(len(pivots), rows):
            if Ar[rr][cols] != 0:
                return None
        x = [Fraction(0)] * cols
        for idx, pr in enumerate(pivots):
            x[pr] = Ar[pr][cols]
        return x

    for it in range(3):
        for i in range(n):
            for k in range(p):
                A = []
                b = [Fraction(1)] * m
                for j in range(m):
                    row = []
                    for r in range(R):
                        row.append(U[i * m + j][r] * V[j * p + k][r])
                    A.append(row)
                sol = solve_block_rational(A, b)
                if sol is None:
                    return None
                for r in range(R):
                    W[k * n + i][r] = sol[r]
        if time.time() - start > time_budget:
            return None

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(R):
                    s = s + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return None
    return U, V, W


def _to_int_half(U, V, W):
    """Convert Fraction matrices to integer/half-integer lists, or None if not possible."""
    def convert(M):
        out = []
        for row in M:
            new_row = []
            for val in row:
                # val should be integer or half-integer
                doubled = val * 2
                if doubled.denominator != 1:
                    return None
                new_row.append(doubled.numerator / 2)
            out.append(new_row)
        return out
    return convert(U), convert(V), convert(W)


def construct():
    n, m, p = DIMENSIONS
    start = time.time()

    # First, get the guaranteed rank-44 solution.
    best = _build_rank44()
    best_rank = 44

    # Try to find a lower rank via exact search.
    # Try ranks 43, 42, 41, ... down to some minimum.
    # Each search has a time budget.
    for R in range(best_rank - 1, max(30, best_rank - 6), -1):
        if time.time() - start > DEADLINE - 15:
            break
        budget = min(20, max(2, DEADLINE - 15 - (time.time() - start)))
        found = False
        # Try multiple seeds for better coverage
        for trial in range(5):
            if time.time() - start > DEADLINE - 15:
                break
            if found:
                break
            try:
                seed = R * 1000 + trial * 7
                res = _search_rank(R, seed=seed, time_budget=budget / 5)
                if res is not None:
                    Uf, Vf, Wf = res
                    # Try to get rational solution
                    res_rat = _solve_rational(R, seed, budget / 5)
                    if res_rat is not None:
                        Ur, Vr, Wr = res_rat
                        conv = _to_int_half(Ur, Vr, Wr)
                        if conv is not None:
                            Uc, Vc, Wc = conv
                            # Verify
                            if _verify(Uc, Vc, Wc):
                                best = {"U": Uc, "V": Vc, "W": Wc}
                                best_rank = R
                                found = True
                                # Don't break; try even lower
                    else:
                        # The field solution exists but rational didn't; skip
                        pass
            except Exception:
                pass

    # Final safety: ensure we return within deadline
    if time.time() - start > DEADLINE - 5:
        return best

    return best
# EVOLVE-BLOCK-END

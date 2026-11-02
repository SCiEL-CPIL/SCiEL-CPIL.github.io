import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_exact_fast(U, V, W, n, m, p):
    """
    Vectorized exact verification.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Returns True iff sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all (i,j,k).
    Uses Fraction arithmetic for exactness but vectorizes the accumulation per (i,j,k) tuple.
    """
    r = U[0].__len__() if U and len(U) > 0 and len(U[0]) > 0 else 0
    # For speed, first do a float pre-check; if fails, return False quickly.
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)
    # Build tensor T via einsum
    # U indexed by (i,j,c) -> reshape to (n,m,r)
    U_t = Uf.reshape(n, m, r)
    V_t = Vf.reshape(m, p, r)
    W_t = Wf.reshape(p, n, r)
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    if not np.allclose(T, 1.0, atol=1e-8):
        return False
    # Now do exact Fraction verification (only if float passed)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(r):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_rank_sparse(r, num_attempts=200, max_nonzero=3, seed_offset=0):
        """
        Try to find a decomposition of rank r using sparse V and W with small integer entries.
        Each column of V and W has at most `max_nonzero` non-zero entries chosen from {-1, 1}.
        Solve for U via least squares, round to integers/half-integers, verify exactly.
        """
        if time.time() > deadline:
            return None

        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None

            rng = random.Random(12345 + r * 1000 + attempt + seed_offset)

            # Build sparse V and W
            # V: (mp, r), W: (pn, r)
            # For each column c, place max_nonzero non-zeros at random positions
            V = [[0] * r for _ in range(mp)]
            W = [[0] * r for _ in range(pn)]

            for c in range(r):
                # V column: choose max_nonzero distinct positions out of mp
                v_positions = rng.sample(range(mp), max_nonzero)
                for pos in v_positions:
                    V[pos][c] = rng.choice([-1, 1])

                # W column: choose max_nonzero distinct positions out of pn
                w_positions = rng.sample(range(pn), max_nonzero)
                for pos in w_positions:
                    W[pos][c] = rng.choice([-1, 1])

            # Solve for U block by block (for each i)
            U = [[0.0] * r for _ in range(nm)]
            valid = True

            for i in range(n):
                # System: A * u_i = b
                # u_i is vector of length m*r (flattened U rows for fixed i)
                # A is (m*p) x (m*r)
                A_mat = np.zeros((m*p, m*r), dtype=np.float64)
                b_vec = np.ones(m*p, dtype=np.float64)

                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = float(V[j*p+k][c] * W[k*n+i][c])

                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break

                # Check if solution is close to integers or half-integers
                sol_int = np.round(sol)
                diff_int = np.max(np.abs(sol - sol_int))

                sol_half = np.round(sol * 2.0) / 2.0
                diff_half = np.max(np.abs(sol - sol_half))

                if diff_int < diff_half:
                    sol_rounded = sol_int
                    diff = diff_int
                else:
                    sol_rounded = sol_half
                    diff = diff_half

                if diff > 1e-8:
                    valid = False
                    break

                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])

            if not valid:
                continue

            # Convert to Fractions for exact verification
            U_frac = []
            for row in U:
                frac_row = []
                for x in row:
                    xi = int(round(x))
                    if abs(x - xi) < 1e-10:
                        frac_row.append(Fraction(xi))
                    else:
                        # half-integer
                        frac_row.append(Fraction(int(round(x * 2)), 2))
                U_frac.append(frac_row)

            V_frac = [[Fraction(x) for x in row] for row in V]
            W_frac = [[Fraction(x) for x in row] for row in W]

            if _verify_exact_fast(U_frac, V_frac, W_frac, n, m, p):
                return {
                    "U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]
                }

        return None

    def try_rank_densersparse(r, num_attempts=100, max_nonzero=5, seed_offset=10000):
        """Denser sparse variant: more non-zeros per column."""
        return try_rank_sparse(r, num_attempts=num_attempts, max_nonzero=max_nonzero, seed_offset=seed_offset)

    def try_rank_dense(r, num_attempts=50, seed_offset=20000):
        """
        Denser V and W: each entry is in {-1, 0, 1} with probability of non-zero ~0.3.
        """
        if time.time() > deadline:
            return None

        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None

            rng = random.Random(34567 + r * 1000 + attempt + seed_offset)

            V = [[0] * r for _ in range(mp)]
            W = [[0] * r for _ in range(pn)]

            for c in range(r):
                for pos in range(mp):
                    if rng.random() < 0.3:
                        V[pos][c] = rng.choice([-1, 1])

            for c in range(r):
                for pos in range(pn):
                    if rng.random() < 0.3:
                        W[pos][c] = rng.choice([-1, 1])

            U = [[0.0] * r for _ in range(nm)]
            valid = True

            for i in range(n):
                A_mat = np.zeros((m*p, m*r), dtype=np.float64)
                b_vec = np.ones(m*p, dtype=np.float64)

                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = float(V[j*p+k][c] * W[k*n+i][c])

                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break

                sol_int = np.round(sol)
                diff_int = np.max(np.abs(sol - sol_int))
                sol_half = np.round(sol * 2.0) / 2.0
                diff_half = np.max(np.abs(sol - sol_half))

                if diff_int < diff_half:
                    sol_rounded = sol_int
                    diff = diff_int
                else:
                    sol_rounded = sol_half
                    diff = diff_half

                if diff > 1e-8:
                    valid = False
                    break

                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])

            if not valid:
                continue

            U_frac = []
            for row in U:
                frac_row = []
                for x in row:
                    xi = int(round(x))
                    if abs(x - xi) < 1e-10:
                        frac_row.append(Fraction(xi))
                    else:
                        frac_row.append(Fraction(int(round(x * 2)), 2))
                U_frac.append(frac_row)

            V_frac = [[Fraction(x) for x in row] for row in V]
            W_frac = [[Fraction(x) for x in row] for row in W]

            if _verify_exact_fast(U_frac, V_frac, W_frac, n, m, p):
                return {
                    "U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]
                }

        return None

    best_result = None
    best_rank = n * m * p

    # Time-budgeted search: allocate time to each rank
    # We'll try ranks from a lower bound upward, spending more time on lower ranks.
    # Lower bound: max(mp, pn, nm) = 35.
    # We'll try from 40 to 105, but with decreasing time budget as rank increases.

    total_time = deadline - start_time
    # Allocate time proportionally: lower ranks get more time
    ranks_to_try = list(range(40, 106))
    # Weight: 1/(rank-35) approximately
    weights = [1.0 / (r - 35 + 1) for r in ranks_to_try]
    total_weight = sum(weights)

    for idx, r in enumerate(ranks_to_try):
        if time.time() > deadline - 3:
            break

        time_budget = total_time * (weights[idx] / total_weight)
        # Minimum time per rank: 0.5s, maximum: 15s
        time_budget = max(0.5, min(15.0, time_budget))

        # Choose strategy based on rank
        if r <= 50:
            # Sparse with 3 non-zeros, many attempts
            attempts = int(time_budget / 0.05)  # ~20 attempts/sec
            attempts = max(50, min(500, attempts))
            result = try_rank_sparse(r, num_attempts=attempts, max_nonzero=3, seed_offset=0)
        elif r <= 70:
            # Denser sparse with 5 non-zeros
            attempts = int(time_budget / 0.05)
            attempts = max(30, min(300, attempts))
            result = try_rank_densersparse(r, num_attempts=attempts, max_nonzero=5, seed_offset=10000)
        else:
            # Dense random
            attempts = int(time_budget / 0.05)
            attempts = max(20, min(200, attempts))
            result = try_rank_dense(r, num_attempts=attempts, seed_offset=20000)

        if result is not None:
            best_result = result
            best_rank = r
            # We found a valid decomposition; try to improve by checking lower ranks?
            # But we're going upward, so this is the lowest rank found so far.
            # Continue searching for even lower rank? No, we're going upward.
            # Actually, we should try lower ranks first. Let's break and return.
            break

    if best_result is not None:
        return best_result

    # Fallback to schoolbook
    return schoolbook()

import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def schoolbook():
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

    def try_rank(r, num_attempts, seed_base=0):
        """
        Try to find a decomposition of rank r.
        Strategy: random V, W with structured entries; solve for U per block; round; verify.
        """
        if time.time() > deadline:
            return None

        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None

            random.seed(seed_base + r * 1000 + attempt)

            # Generate random V and W with entries from {-1, 0, 1}
            # Use a sparsity pattern to keep the linear system well-conditioned
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]

            # Ensure no column is all-zeros in V or W (degenerate case)
            for c in range(r):
                if all(V[ic][c] == 0 for ic in range(mp)):
                    V[random.randint(0, mp-1)][c] = 1
                if all(W[ic][c] == 0 for ic in range(pn)):
                    W[random.randint(0, pn-1)][c] = 1

            U = [[0.0]*r for _ in range(nm)]
            valid = True

            for i in range(n):
                # Build linear system for U[i*m+j, c] over j=0..m-1
                # Variables: m*r (the m rows of U for fixed i)
                # Equations: m*p (one per (j,k) pair)
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)

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

                # Round to nearest half-integer
                sol_rounded = np.round(sol * 2.0) / 2.0
                diff = np.max(np.abs(sol - sol_rounded))

                # If half-integers don't work well, try integers
                if diff > 1e-8:
                    sol_int = np.round(sol)
                    diff_int = np.max(np.abs(sol - sol_int))
                    if diff_int < diff:
                        sol_rounded = sol_int
                        diff = diff_int

                if diff > 1e-6:
                    valid = False
                    break

                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])

            if not valid:
                continue

            # Verify numerically first (fast)
            if _verify_np(U, V, W, n, m, p):
                # Convert to fractions for exact verification
                U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]

                if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }

        return None

    # Search for lower ranks, starting from just below schoolbook
    # Use more attempts for lower ranks since they're harder
    for r in range(104, 49, -1):
        if time.time() > deadline:
            break

        # Scale attempts: more attempts for lower ranks
        if r >= 90:
            num_attempts = 30
        elif r >= 70:
            num_attempts = 50
        else:
            num_attempts = 20

        result = try_rank(r, num_attempts, seed_base=0)
        if result is not None:
            return result

    return schoolbook()

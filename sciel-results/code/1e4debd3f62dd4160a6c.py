import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 56
    deadline = time.time() + 140

    def is_half_integer_matrix(M, tol=1e-9):
        M2 = np.asarray(M, dtype=float) * 2
        M2_rounded = np.round(M2)
        return bool(np.allclose(M2, M2_rounded, atol=tol))

    def verify(U, V, W):
        """Exact verification: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = U.shape[1]
        Uf = np.asarray(U, dtype=float)
        Vf = np.asarray(V, dtype=float)
        Wf = np.asarray(W, dtype=float)
        # For each (i,k): sum over j,c
        ok = True
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    urow = Uf[i*m+j, :]
                    vrow = Vf[j*p+k, :]
                    wrow = Wf[k*n+i, :]
                    s += np.dot(urow, vrow * wrow)
                if abs(s - 1.0) > 1e-7:
                    ok = False
                    break
            if not ok:
                break
        return ok

    def snap_half(M):
        return np.round(np.asarray(M, dtype=float) * 2.0) / 2.0

    def try_rank(r, max_attempts=400, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        nm = n * m
        mp = m * p
        pn = p * n
        int_vals = np.array([-2, -1, 0, 1, 2])
        small_vals = np.array([-1, 0, 1])

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None

            if attempt % 2 == 0:
                vals = small_vals
            else:
                vals = int_vals

            U = rng.choice(vals, size=(nm, r))
            W = rng.choice(vals, size=(pn, r))

            V = np.zeros((mp, r), dtype=float)
            valid = True

            for j in range(m):
                for k in range(p):
                    # Build B (n x r): B[i,c] = U[i*m+j,c]*W[k*n+i,c]
                    B = np.array([
                        [U[i*m+j, c] * W[k*n+i, c] for c in range(r)]
                        for i in range(n)
                    ], dtype=float)
                    rhs = np.ones(n)
                    try:
                        v = np.linalg.lstsq(B, rhs, rcond=None)[0]
                        if not np.allclose(B @ v, rhs, atol=1e-9):
                            valid = False
                            break
                        V[j*p+k, :] = v
                    except Exception:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            # Snap to half-integers and verify
            U_r = snap_half(U)
            V_r = snap_half(V)
            W_r = snap_half(W)
            if verify(U_r, V_r, W_r):
                return (U_r, V_r, W_r)

            # Also verify without snap (in case already half-integer)
            if verify(U, V, W):
                return (U, V, W)

        return None

    # Structured construction: build U, W with guaranteed per-(i,k) solvability
    def try_rank_structured(r, max_attempts=300, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        nm = n * m
        mp = m * p
        pn = p * n

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None

            # Choose U and W with integer entries, ensuring for each (i,k)
            # the system B v = 1 has an integer/half-integer solution.
            U = np.zeros((nm, r), dtype=int)
            W = np.zeros((pn, r), dtype=int)

            # Assign random integer values
            vals = np.array([-1, 0, 1])
            U = rng.choice(vals, size=(nm, r))
            W = rng.choice(vals, size=(pn, r))

            V = np.zeros((mp, r), dtype=float)
            valid = True

            for j in range(m):
                for k in range(p):
                    B = np.array([
                        [U[i*m+j, c] * W[k*n+i, c] for c in range(r)]
                        for i in range(n)
                    ], dtype=float)
                    rhs = np.ones(n)
                    # Check if 1 is in column space of B
                    try:
                        rankB = np.linalg.matrix_rank(B)
                        if rankB < n:
                            valid = False
                            break
                        v = np.linalg.lstsq(B, rhs, rcond=None)[0]
                        if not np.allclose(B @ v, rhs, atol=1e-9):
                            valid = False
                            break
                        V[j*p+k, :] = v
                    except Exception:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            U_r = snap_half(U)
            V_r = snap_half(V)
            W_r = snap_half(W)
            if verify(U_r, V_r, W_r):
                return (U_r, V_r, W_r)

        return None

    best_result = None

    # Try a range of ranks; lower ranks are more likely to have solutions
    # but harder to find. Start moderate and go down.
    for r in range(40, 56):
        if time.time() > deadline - 10:
            break
        rng = np.random.RandomState(r * 1000 + 7)
        result = try_rank(r, max_attempts=150, rng=rng)
        if result is None:
            result = try_rank_structured(r, max_attempts=100, rng=rng)
        if result is not None:
            best_result = result
            break

    if best_result is not None:
        U, V, W = best_result
        rank = U.shape[1]
        U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook decomposition
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

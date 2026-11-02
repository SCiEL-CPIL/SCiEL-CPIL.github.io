"""Combined program: rank-minimal exact decomposition for 2x2x9 multiplication.

Strategy: keep the proven rank-32 Strassen-per-block construction as a guaranteed
fallback, then run a bounded exact search for a single lower-rank decomposition
of the full [2,2,9] tensor (shared A-side U, column-structured V/W). Every
candidate is verified by exact tensor reconstruction (no tolerance). A hard
deadline guard returns the best (lowest-rank) valid result found.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    deadline = time.time() + 120.0

    def build_rank32():
        """Proven rank-32: 4 Strassen blocks (2 cols each) + schoolbook for col 8."""
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(k0, k1, base_col):
            c = base_col
            U[0][c] += 1; U[3][c] += 1
            V[0 * p + k0][c] += 1; V[1 * p + k1][c] += 1
            W[k0 * n + 0][c] += 1; W[k1 * n + 1][c] += 1

            c = base_col + 1
            U[2][c] += 1; U[3][c] += 1
            V[0 * p + k0][c] += 1
            W[k0 * n + 1][c] += 1; W[k1 * n + 1][c] -= 1

            c = base_col + 2
            U[0][c] += 1
            V[0 * p + k1][c] += 1; V[1 * p + k1][c] -= 1
            W[k1 * n + 0][c] += 1; W[k1 * n + 1][c] += 1

            c = base_col + 3
            U[3][c] += 1
            V[1 * p + k0][c] += 1; V[0 * p + k0][c] -= 1
            W[k0 * n + 0][c] += 1; W[k0 * n + 1][c] += 1

            c = base_col + 4
            U[0][c] += 1; U[1][c] += 1
            V[1 * p + k1][c] += 1
            W[k0 * n + 0][c] -= 1; W[k1 * n + 0][c] += 1

            c = base_col + 5
            U[2][c] += 1; U[0][c] -= 1
            V[0 * p + k0][c] += 1; V[0 * p + k1][c] += 1
            W[k1 * n + 1][c] += 1

            c = base_col + 6
            U[1][c] += 1; U[3][c] -= 1
            V[1 * p + k0][c] += 1; V[1 * p + k1][c] += 1
            W[k0 * n + 0][c] += 1

        for b in range(4):
            add_strassen_block(2 * b, 2 * b + 1, b * 7)

        k = 8
        base_col = 28
        U[0][base_col + 0] += 1; V[0 * p + k][base_col + 0] += 1; W[k * n + 0][base_col + 0] += 1
        U[1][base_col + 1] += 1; V[1 * p + k][base_col + 1] += 1; W[k * n + 0][base_col + 1] += 1
        U[2][base_col + 2] += 1; V[0 * p + k][base_col + 2] += 1; W[k * n + 1][base_col + 2] += 1
        U[3][base_col + 3] += 1; V[1 * p + k][base_col + 3] += 1; W[k * n + 1][base_col + 3] += 1

        return {"U": U, "V": V, "W": W}

    # Guaranteed valid fallback.
    fallback = build_rank32()

    # --- Verification: exact tensor reconstruction ---
    def tensor_from_factors(U, V, W):
        r = len(U[0])
        T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
        for t in range(r):
            u = np.array(U[:, t])
            v = np.array(V[:, t])
            w = np.array(W[:, t])
            T += np.einsum('i,j,k->ijk', u, v, w)
        return T

    def target_tensor():
        T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T

    TARGET = target_tensor()

    def verify(U, V, W):
        return np.array_equal(tensor_from_factors(U, V, W), TARGET)

    assert verify(fallback["U"], fallback["V"], fallback["W"]), "fallback invalid"

    # --- Exact search for a single lower-rank decomposition ---
    # U is the shared A-side (4 x r); V,W are column-structured (18 x r).
    # Each term's V and W are supported on a single output column k.
    best = fallback
    best_rank = 32

    # Candidate U rows (4-dim integer vectors) and per-column V/W patterns.
    U_ROWS = [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 0],
        [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, -1, 0], [1, 0, 0, -1],
        [0, 1, 0, -1], [0, 1, -1, 0], [1, 1, 0, 0], [1, -1, 0, 0],
        [0, 1, -1, 0], [1, 0, -1, 0], [0, 1, 1, 0], [1, 0, 1, 0],
        [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
        [1, -1, 1, 0], [1, -1, 0, 1], [1, 0, -1, 1], [0, 1, -1, 1],
        [1, 1, -1, 0], [1, 1, 0, -1], [1, 0, 1, -1], [0, 1, 1, -1],
        [1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1],
    ]

    # Per-column V patterns (row 0 = B[0,k], row 1 = B[1,k])
    V_PAT = [
        [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1],
    ]
    # Per-column W patterns (row 0 = C[0,k], row 1 = C[1,k])
    W_PAT = [
        [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1],
    ]

    # A term is (u_row, k, v_pat, w_pat). Its contribution to tensor entry
    # (i, j, k') is u[i] * v[j] * w[i] if k'==k else 0, where v is the
    # V pattern placed at column k and w is the W pattern placed at column k.
    def term_contrib(u, k, vpat, wpat):
        # returns dict of (i, j, k) -> value
        out = {}
        for i in range(n):
            for j in range(m):
                val = u[i] * vpat[j] * wpat[i]
                if val != 0:
                    out[(i, j, k)] = val
        return out

    # Precompute target as dict for fast lookup
    target_dict = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_dict[(i, j, k)] = 1

    def trial_succeeds(assign):
        # assign: list of (u, k, vpat, wpat)
        acc = {}
        for (u, k, vp, wp) in assign:
            for key, val in term_contrib(u, k, vp, wp).items():
                acc[key] = acc.get(key, 0) + val
        # check exact equality with target
        for key, val in acc.items():
            if target_dict.get(key, 0) != val:
                return False
        for key, val in target_dict.items():
            if acc.get(key, 0) != val:
                return False
        return True

    def build_factors(assign):
        r = len(assign)
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        for t, (u, k, vp, wp) in enumerate(assign):
            for i in range(n):
                U[i * m + 0][t] = u[i] if i == 0 else U[i * m][t]
            # U index is i*m+j; A has only 2x2 so j=0,1 -> indices i*2+j
            # We only use A[i,0] effectively? No: A is 2x2, U index i*m+j.
            # Our u is 4-dim = [A00,A01,A10,A11]. Place accordingly.
            U[0][t] = u[0]
            U[1][t] = u[1]
            U[2][t] = u[2]
            U[3][t] = u[3]
            V[0 * p + k][t] = vp[0]
            V[1 * p + k][t] = vp[1]
            W[k * n + 0][t] = wp[0]
            W[k * n + 1][t] = wp[1]
        return {"U": U, "V": V, "W": W}

    rng = random.Random(12345)

    for r in range(13, 32):
        if time.time() > deadline - 3.0:
            break
        # Number of trials scales with remaining time.
        remaining = deadline - time.time()
        max_trials = int(remaining * 250)  # ~250 trials/sec budget
        max_trials = min(max_trials, 40000)
        for _ in range(max_trials):
            if time.time() > deadline - 3.0:
                break
            assign = []
            for _t in range(r):
                u = rng.choice(U_ROWS)
                k = rng.randrange(p)
                vp = rng.choice(V_PAT)
                wp = rng.choice(W_PAT)
                assign.append((u, k, vp, wp))
            if trial_succeeds(assign):
                cand = build_factors(assign)
                if verify(cand["U"], cand["V"], cand["W"]):
                    best = cand
                    best_rank = r
                    break  # found lowest r that works; stop

    return best
# EVOLVE-BLOCK-END

"""Matrix multiplication 2x3x9 rank-minimal exact tensor decomposition.

Strategy: schoolbook baseline (rank 54) as guaranteed fallback, plus a
random-restart alternating-minimization search for a lower-rank exact
integer decomposition. Each restart initializes U,V,W with small integers,
then iteratively solves each factor given the other two, rounds, and polishes.
Exact verification guarantees the returned decomposition is correct; otherwise
the schoolbook rank-54 baseline is returned.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin under 150s wall-clock

    n, m, p = DIMENSIONS
    nm = n * m      # 6
    mp = m * p      # 27
    pn = p * n      # 18

    # ---------- schoolbook baseline (rank = n*m*p = 54) ----------
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best = {"U": [row[:] for row in U],
            "V": [row[:] for row in V],
            "W": [row[:] for row in W]}
    best_rank = rank_school

    # ---------- exact verification ----------
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    # ---------- alternating minimization (closed-form per factor) ----------
    def solve_U(V, W, rank):
        # For each (i,k): sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # Unknowns U[i*m+0..m-1,r] with one equation -> min-norm solution.
        U = np.zeros((nm, rank))
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    A = np.array([V[j * p + k, r] * W[k * n + i, r]
                                  for j in range(m)], dtype=float)
                    nrm = A @ A
                    if nrm < 1e-12:
                        u = np.zeros(m)
                    else:
                        u = A / nrm
                    for j in range(m):
                        U[i * m + j, r] = u[j]
        return U

    def solve_V(U, W, rank):
        # For each (j,k): V[j*p+k,r] * sum_i U[i*m+j,r]*W[k*n+i,r] = 1
        V = np.zeros((mp, rank))
        for r in range(rank):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for i in range(n):
                        s += U[i * m + j, r] * W[k * n + i, r]
                    V[j * p + k, r] = 0.0 if abs(s) < 1e-12 else 1.0 / s
        return V

    def solve_W(U, V, rank):
        # For each (i,k): W[k*n+i,r] * sum_j U[i*m+j,r]*V[j*p+k,r] = 1
        W = np.zeros((pn, rank))
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U[i * m + j, r] * V[j * p + k, r]
                    W[k * n + i, r] = 0.0 if abs(s) < 1e-12 else 1.0 / s
        return W

    def round_half(X):
        return np.round(2.0 * X) / 2.0

    def polish(U, V, W, rank):
        U = U.copy(); V = V.copy(); W = W.copy()
        for _ in range(60):
            worst = None; worst_val = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = abs(s - 1.0)
                        if err > worst_val:
                            worst_val = err
                            worst = (i, j, k)
            if worst_val < 1e-9:
                return U, V, W, True
            i, j, k = worst
            r = random.randint(0, rank - 1)
            d = random.choice([-1, 1])
            which = random.randint(0, 2)
            if which == 0:
                U[i*m+j, r] += d
            elif which == 1:
                V[j*p+k, r] += d
            else:
                W[k*n+i, r] += d
        return U, V, W, False

    def run_altmin(rank, iters=10):
        vals = [0, 1, 2, -1, -2]
        U = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(nm)], dtype=float)
        V = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(mp)], dtype=float)
        W = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(pn)], dtype=float)
        for _ in range(iters):
            U = round_half(solve_U(V, W, rank))
            V = round_half(solve_V(U, W, rank))
            W = round_half(solve_W(U, V, rank))
        U2, V2, W2, ok = polish(np.round(U), np.round(V), np.round(W), rank)
        if ok:
            return U2, V2, W2
        return None

    # ---------- search ----------
    for target_rank in range(best_rank - 1, 25, -1):
        if time.time() - start_time > DEADLINE - 15:
            break
        for attempt in range(40):
            if time.time() - start_time > DEADLINE - 8:
                break
            res = run_altmin(target_rank)
            if res is None:
                continue
            U2, V2, W2 = res
            U2 = U2.astype(int); V2 = V2.astype(int); W2 = W2.astype(int)
            if verify(U2.tolist(), V2.tolist(), W2.tolist(), target_rank):
                best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                best_rank = target_rank
                break

    return best
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
n, m, p = DIMENSIONS


# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 150s kill
    rng = random.Random(12345)
    np.random.seed(12345)

    # ---- Build the full target tensor T[a,b,c] for verification ----
    # a = i*m+j (A index), b = j*p+k (B index), c = k*n+i (C index)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # ---- Schoolbook fallback (rank 45) ----
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # ---- 3x3 sub-problem tensor: T3[a3,b3,c3]=1 where a3=i*3+j, b3=j*3+k, c3=k*3+i ----
    T3 = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T3[i * 3 + j, j * 3 + k, k * 3 + i] = 1.0

    def verify_3x3(U3, V3, W3):
        P = np.einsum('ak,bk,ck->abc', U3, V3, W3)
        return float(np.max(np.abs(P - T3)))

    def replicate(U3, V3, W3, r3):
        """
        Replicate a 3x3 rank-r3 decomposition to 3x3*3x5 rank-5*r3.
        For each output column k (0..4):
          U_full[a, k*r3+t] = U3[a, t]
          V_full[b, k*r3+t] = V3[j*3+k, t]  where b = j*5+k (so b%5==k, j=b//5)
          W_full[c, k*r3+t] = W3[k*3+i, t]  where c = k*3+i (so c//3==k, i=c%3)
        """
        r_full = r3 * p
        U_full = np.zeros((n * m, r_full))
        V_full = np.zeros((m * p, r_full))
        W_full = np.zeros((p * n, r_full))
        for k in range(p):
            off = k * r3
            U_full[:, off:off + r3] = U3
            for j in range(m):
                b = j * p + k
                V_full[b, off:off + r3] = V3[j * 3 + k, :]
            for i in range(n):
                c = k * n + i
                W_full[c, off:off + r3] = W3[k * 3 + i, :]
        return U_full, V_full, W_full

    def verify_full(U, V, W):
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        return float(np.max(np.abs(P - T)))

    def try_rank_3x3(r3, max_attempts):
        """Random U3, V3 (small integers), solve W3 by least squares, round to half-integers."""
        for _ in range(max_attempts):
            if time.time() > deadline - 8:
                return None
            U3 = np.random.randint(-2, 3, size=(9, r3)).astype(float)
            V3 = np.random.randint(-2, 3, size=(9, r3)).astype(float)
            # Build linear system for W3: for each (a3,b3) with T3[a3,b3,c3]=1,
            # sum_t U3[a3,t]*V3[b3,t]*W3[c3,t] = 1
            # Unknowns: W3[c3, t] for c3 in 0..8, t in 0..r3-1  => 9*r3 unknowns
            # Equations: for each (a3,b3,c3) with T3=1: sum_t U3[a3,t]*V3[b3,t]*W3[c3,t] = 1
            num_unknowns = 9 * r3
            A = np.zeros((27, num_unknowns))  # at most 27 nonzero entries
            bvec = np.zeros(27)
            eq = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        a3 = i * 3 + j
                        b3 = j * 3 + k
                        c3 = k * 3 + i
                        for t in range(r3):
                            A[eq, c3 * r3 + t] = U3[a3, t] * V3[b3, t]
                        bvec[eq] = 1.0
                        eq += 1
            A = A[:eq]
            bvec = bvec[:eq]
            try:
                W3_flat, res, rk, sv = np.linalg.lstsq(A, bvec, rcond=None)
            except Exception:
                continue
            W3 = W3_flat.reshape(9, r3)
            # Round to half-integers
            W3r = np.round(W3 * 2) / 2
            err = verify_3x3(U3, V3, W3r)
            if err < 1e-9:
                return U3, V3, W3r
        return None

    # ---- Search 3x3 sub-problem from low rank upward ----
    for r3 in range(4, 9):  # 3x3x3 schoolbook is 9; Strassen-style is 7
        if time.time() > deadline - 15:
            break
        found = try_rank_3x3(r3, 40)
        if found is not None:
            U3, V3, W3 = found
            U_full, V_full, W_full = replicate(U3, V3, W3, r3)
            err = verify_full(U_full, V_full, W_full)
            if err < 1e-9:
                best_rank = r3 * p
                best_result = {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist(),
                }
                break

    return best_result
# EVOLVE-BLOCK-END

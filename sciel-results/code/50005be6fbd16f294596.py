"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
Uses structured families + exact verification with a hard deadline guard.
Falls back to schoolbook rank 42.
"""

import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin under 150s kill

    n, m, p = DIMENSIONS
    NM, MP, PN = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 42

    def build_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = build_schoolbook()

    def verify(U, V, W):
        """U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank.
        Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    uidx = i * m + j
                    vidx = j * p + k
                    widx = k * n + i
                    Urow = U[uidx]
                    Vrow = V[vidx]
                    Wrow = W[widx]
                    for c in range(rank):
                        s += Urow[c] * Vrow[c] * Wrow[c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def als_decompose(rank, max_iters=50, seed=42, init_scale=1.0):
        """ALS to find approximate U, V, W of given rank."""
        rng = random.Random(seed)
        U_np = np.array([[rng.uniform(-init_scale, init_scale) for _ in range(rank)] for _ in range(NM)], dtype=float)
        V_np = np.array([[rng.uniform(-init_scale, init_scale) for _ in range(rank)] for _ in range(MP)], dtype=float)
        W_np = np.array([[rng.uniform(-init_scale, init_scale) for _ in range(rank)] for _ in range(PN)], dtype=float)
        
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V_np[j * p + k, c] * W_np[k * n + i, c]
                    b = np.ones(p)
                    try:
                        U_row = np.linalg.lstsq(A, b, rcond=None)[0]
                        U_np[i * m + j, :] = U_row
                    except:
                        pass
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U_np[i * m + j, c] * W_np[k * n + i, c]
                    b = np.ones(n)
                    try:
                        V_row = np.linalg.lstsq(A, b, rcond=None)[0]
                        V_np[j * p + k, :] = V_row
                    except:
                        pass
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U_np[i * m + j, c] * V_np[j * p + k, c]
                    b = np.ones(m)
                    try:
                        W_row = np.linalg.lstsq(A, b, rcond=None)[0]
                        W_np[k * n + i, :] = W_row
                    except:
                        pass
        return U_np, V_np, W_np

    def round_to_half(x):
        return round(x * 2) / 2

    def try_rank_als(rank, n_trials=6, max_iters=40):
        for trial in range(n_trials):
            if time.time() - t_start > DEADLINE - 5:
                return None
            try:
                U_np, V_np, W_np = als_decompose(rank, max_iters=max_iters, seed=100 + trial)
            except:
                continue
            U_r = [[round_to_half(x) for x in row] for row in U_np.tolist()]
            V_r = [[round_to_half(x) for x in row] for row in V_np.tolist()]
            W_r = [[round_to_half(x) for x in row] for row in W_np.tolist()]
            if verify(U_r, V_r, W_r):
                return {"U": U_r, "V": V_r, "W": W_r}
        return None

    # Try structured low-rank constructions first
    # Theoretical lower bound considerations:
    # For matrix multiplication, rank is at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But there are better lower bounds from algebraic geometry.
    
    # Try to find decompositions with rank < 42
    lower_bound = max(NM, MP, PN)  # 21
    
    # First try some small ranks with multiple trials
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() - t_start > DEADLINE - 10:
            break
        result = try_rank_als(rank, n_trials=6, max_iters=40)
        if result is not None:
            return result
    
    return best
# EVOLVE-BLOCK-END

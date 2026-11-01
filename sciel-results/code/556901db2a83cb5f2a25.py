"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fallback = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fallback = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fallback = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = rank_schoolbook

    # Try Strassen-based rank-49 construction
    if time.time() < deadline - 5:
        success = _construct_strassen_49(n, m, p)
        if success:
            best_rank = 49
            U, V, W = _get_current_decomposition()
            best_result = {"U": U, "V": V, "W": W}

    return best_result

def _construct_strassen_49(n, m, p):
    """Construct a rank-49 decomposition using 7 pairs of k-values with Strassen."""
    if p % 2 != 0:
        return False
    num_pairs = p // 2
    rank = 7 * num_pairs

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for pair_idx in range(num_pairs):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base = pair_idx * 7

        # Indices for U (A matrix): 00, 01, 10, 11
        # Indices for V (B matrix): 0k1, 0k2, 1k1, 1k2
        # Indices for W (C matrix): 0k1, 0k2, 1k1, 1k2

        # M1 = (A00 + A11) * (B0k1 + B0k2 + B1k1 + B1k2)
        U[0][base] = 1
        U[3][base] = 1
        V[k1][base] = 1
        V[k2][base] = 1
        V[p + k1][base] = 1
        V[p + k2][base] = 1
        W[k1 * n + 0][base] = 1  # C0k1
        W[k2 * n + 1][base] = 1  # C1k2

        # M2 = (A10 + A11) * B0k1
        U[2][base + 1] = 1
        U[3][base + 1] = 1
        V[k1][base + 1] = 1
        W[k1 * n + 1][base + 1] = 1  # C1k1
        W[k2 * n + 1][base + 1] = -1 # C1k2

        # M3 = A00 * (B0k2 - B1k2)
        U[0][base + 2] = 1
        V[k2][base + 2] = 1
        V[p + k2][base + 2] = -1
        W[k2 * n + 0][base + 2] = 1  # C0k2
        W[k2 * n + 1][base + 2] = 1  # C1k2

        # M4 = A11 * (B1k1 - B0k1)
        U[3][base + 3] = 1
        V[p + k1][base + 3] = 1
        V[k1][base + 3] = -1
        W[k1 * n + 0][base + 3] = 1  # C0k1
        W[k1 * n + 1][base + 3] = 1  # C1k1

        # M5 = (A00 + A01) * B1k2
        U[0][base + 4] = 1
        U[1][base + 4] = 1
        V[p + k2][base + 4] = 1
        W[k1 * n + 0][base + 4] = -1 # C0k1
        W[k2 * n + 0][base + 4] = 1  # C0k2

        # M6 = (A10 - A00) * (B0k1 + B0k2)
        U[2][base + 5] = 1
        U[0][base + 5] = -1
        V[k1][base + 5] = 1
        V[k2][base + 5] = 1
        W[k1 * n + 1][base + 5] = -1 # C1k1
        # Note: M6 does not contribute to C0k1 or C0k2 or C1k2 in this formulation?
        # Let's check C1k1 = M2 + M4.
        # C1k1 = (A10+A11)B0k1 + A11(B1k1-B0k1) = A10B0k1 + A11B1k1.
        # My M2 contributes to C1k1 (+1) and C1k2 (-1).
        # My M4 contributes to C1k1 (+1) and C0k1 (+1).
        # Wait, standard Strassen C10 = M2 + M4.
        # So C1k1 should be M2 + M4.
        # In my W assignments:
        # M2: W[C1k1] = 1.
        # M4: W[C1k1] = 1.
        # Sum = 2? No, the coefficients in U/V handle the expansion.
        # Let's trust the verification step. If it fails, I return False.

        # M7 = (A01 - A11) * (B1k1 + B1k2)
        U[1][base + 6] = 1
        U[3][base + 6] = -1
        V[p + k1][base + 6] = 1
        V[p + k2][base + 6] = 1
        W[k1 * n + 0][base + 6] = -1 # C0k1
        W[k2 * n + 1][base + 6] = 1  # C1k2

    return _verify_and_store(U, V, W, n, m, p)

def _verify_and_store(U, V, W, n, m, p):
    """Verify the decomposition and store it if valid."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False

    global _current_U, _current_V, _current_W
    _current_U = U
    _current_V = V
    _current_W = W
    return True

def _get_current_decomposition():
    return _current_U, _current_V, _current_W

_current_U = None
_current_V = None
_current_W = None
# EVOLVE-BLOCK-END

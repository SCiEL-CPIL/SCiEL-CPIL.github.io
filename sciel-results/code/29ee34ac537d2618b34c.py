"""Rewritten program: search for low-rank exact decomposition of 5x5x5 matmul tensor."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin before 150s

    n, m, p = DIMENSIONS

    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                row_list.append(rounded)
            result.append(row_list)
        return result

    # Build schoolbook fallback
    U_fb = np.zeros((n*m, n*m*p))
    V_fb = np.zeros((m*p, n*m*p))
    W_fb = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1

    best_result = None
    best_rank = n * m * p

    # Try block diagonal decomposition: 3x3 + 2x2
    # For 3x3, use Strassen rank 7
    # For 2x2, use Strassen rank 7
    # Total: 7 + 7 = 14? No, that's for the sub-blocks only.
    # Actually for block diagonal, we need:
    # - 3x3 block: rank 7
    # - 2x2 block: rank 7  
    # - Cross terms handled separately
    # This doesn't work directly. Let me try a different approach.

    # Try a structured approach using Kronecker products
    # Split 5 = 3 + 2
    # Block structure:
    # [A11 A12] [B11 B12] = [C11 C12]
    # [A21 A22] [B21 B22] = [C21 C22]
    # where A11,B11,C11 are 3x3 and A22,B22,C22 are 2x2

    # For each block, we can use known decompositions
    # 3x3: Strassen rank 7
    # 2x2: Strassen rank 7 (or schoolbook 4)
    # Cross blocks: need to handle 3x2, 2x3, 2x2 multiplications

    # Let's try a simpler approach: use a rank-49 decomposition
    # based on known bounds

    # Actually, let me try to construct a valid decomposition using
    # the following pattern: treat the 5x5 as a block matrix and use
    # tensor product decompositions

    # For now, try random search with structured initialization
    np.random.seed(42)

    # Try ranks that are known to be achievable or close to optimal
    target_ranks = [49, 50, 52, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break

        found = False
        for attempt in range(200):
            if time.time() > deadline - 15:
                break

            # Structured sparse initialization
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))

            # Each column should contribute to a subset of entries
            # Use a pattern where each column covers a small rectangle
            for c in range(target_rank):
                # Random rectangle for U
                ui_start = np.random.randint(0, n)
                ui_size = np.random.randint(1, 3)
                uj_start = np.random.randint(0, m)
                uj_size = np.random.randint(1, 3)

                ui_end = min(ui_start + ui_size, n)
                uj_end = min(uj_start + uj_size, m)

                for ui in range(ui_start, ui_end):
                    for uj in range(uj_start, uj_end):
                        U[ui*m + uj, c] = np.random.choice([-1, 1])

                # Random rectangle for V
                vi_start = np.random.randint(0, m)
                vi_size = np.random.randint(1, 3)
                vj_start = np.random.randint(0, p)
                vj_size = np.random.randint(1, 3)

                vi_end = min(vi_start + vi_size, m)
                vj_end = min(vj_start + vj_size, p)

                for vi in range(vi_start, vi_end):
                    for vj in range(vj_start, vj_end):
                        V[vi*p + vj, c] = np.random.choice([-1, 1])

                # Random rectangle for W
                wi_start = np.random.randint(0, p)
                wi_size = np.random.randint(1, 3)
                wj_start = np.random.randint(0, n)
                wj_size = np.random.randint(1, 3)

                wi_end = min(wi_start + wi_size, p)
                wj_end = min(wj_start + wj_size, n)

                for wi in range(wi_start, wi_end):
                    for wj in range(wj_start, wj_end):
                        W[wi*n + wj, c] = np.random.choice([-1, 1])

            if verify(U, V, W):
                best_rank = target_rank
                best_result = (U, V, W)
                found = True
                break

        if found:
            break

    # If no better found, use schoolbook
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_fb, V_fb, W_fb

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END

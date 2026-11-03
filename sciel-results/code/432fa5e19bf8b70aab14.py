"""Combined program: schoolbook decomposition with verification and time guard."""
import time

DIMENSIONS = [2, 3, 7]

# Hard wall-clock safety margin (seconds). Evaluator kills at 150 s.
_DEADLINE_SEC = 145.0


def _elapsed_since(start):
    return time.monotonic() - start


def _schoolbook_decomposition(n, m, p):
    """Exact rank n*m*p decomposition with 0/1 integer entries.

    U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1 for the c-th
    (i, j, k) triple. Output factor uses the official trace convention
    W[k*n + i, c].
    """
    rank = n * m * p
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


def _verify_exact(n, m, p, U, V, W):
    """Check that sum_c U[a,c]*V[b,c]*W[d,c] == 1 exactly when
    a = i*m+j, b = j*p+k, d = k*n+i, and 0 otherwise.
    Uses pure integer arithmetic, so the check is exact.
    """
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for d in range(p * n):
                total = 0
                for c in range(rank):
                    total += U[a][c] * V[b][c] * W[d][c]
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(d, n)
                expected = 1 if (j == j2 and k == k2 and i == i2) else 0
                if total != expected:
                    return False
    return True


def construct():
    start = time.monotonic()
    n, m, p = DIMENSIONS

    # Return the exact schoolbook decomposition, verified for correctness.
    result = _schoolbook_decomposition(n, m, p)

    # Safety: if we are running out of time, return the known-good result
    # immediately rather than risk a timeout.
    if _elapsed_since(start) > _DEADLINE_SEC:
        return result

    # Exact verification (integer arithmetic). The schoolbook decomposition
    # is guaranteed correct, so this should always pass.
    ok = _verify_exact(n, m, p, result["U"], result["V"], result["W"])
    return result

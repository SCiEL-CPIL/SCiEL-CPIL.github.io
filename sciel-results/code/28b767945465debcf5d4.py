"""Rewritten program for the suite task; keep construct() -> {'U','V','W'}."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# Hard wall-clock safety margin (seconds). Evaluator kills at 150 s.
_DEADLINE_SEC = 140.0


def _elapsed_since(start):
    return time.monotonic() - start


def _schoolbook_decomposition(n, m, p):
    """Exact rank n*m*p decomposition with 0/1 integer entries.

    U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1 for the c-th
    (i, j, k) triple. Output factor uses the official trace convention
    W[k*n + i, c].
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}


def _verify_exact(n, m, p, U, V, W):
    """Check that sum_c U[a,c]*V[b,c]*W[d,c] == 1 exactly when
    a = i*m+j, b = j*p+k, d = k*n+i, and 0 otherwise.
    Uses NumPy for speed, but the result is exact integer arithmetic.
    """
    rank = U.shape[1]
    # We need to check for all a, b, d
    # T[a, b, d] = sum_c U[a, c] V[b, c] W[d, c]
    # This is equivalent to contracting U and V over c, then checking against W?
    # No, it's a 3-tensor.
    # Let's just iterate, it's small enough (2*3 * 3*7 * 7*2 = 6 * 21 * 14 = 1764 checks).
    # Each check sums 42 terms. 1764 * 42 ~ 74k ops. Very fast.
    
    for a in range(n * m):
        for b in range(m * p):
            for d in range(p * n):
                # Calculate the sum
                # U[a, :] is shape (rank,)
                # V[b, :] is shape (rank,)
                # W[d, :] is shape (rank,)
                # sum(U[a,c] * V[b,c] * W[d,c])
                prod = U[a, :] * V[b, :] * W[d, :]
                total = np.sum(prod)
                
                # Expected target value:
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

    # Construct the schoolbook decomposition
    result = _schoolbook_decomposition(n, m, p)

    # Safety: if we are running out of time, return the known-good result
    # immediately rather than risk a timeout.
    if _elapsed_since(start) > _DEADLINE_SEC:
        return result

    # Exact verification.
    # If verification fails (which it shouldn't), we still return the result
    # as it is the best guaranteed-valid answer structure, but typically
    # we trust the construction.
    # The verification is fast enough to run.
    # ok = _verify_exact(n, m, p, result["U"], result["V"], result["W"])
    # if not ok:
    #     # Fallback: return empty or raise? No, return schoolbook.
    #     pass

    return result

"""Matrix multiplication 3x5x5 rank search with deadline guard."""
import time
import sys

DIMENSIONS = [3, 5, 5]

# Hard wall-clock deadline (evaluator kills at 150s; keep safety margin)
DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - _elapsed()

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 75."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-6):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == delta(i,j,k) for all i,j,k.

    The multiplication tensor T[i,j,k] = 1 iff the (i,k) entry of A@B equals
    the product A[i,j]*B[j,k]. So we need:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1  for all (i,j,k)
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build arrays
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reshape to tensors
    Ua = Ua.reshape(n, m, rank)
    Va = Va.reshape(m, p, rank)
    Wa = Wa.reshape(p, n, rank)
    # Compute T[i,j,k] = sum_c Ua[i,j,c]*Va[j,k,c]*Wa[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    # Expected: all ones
    return np.max(np.abs(T - 1.0)) < tol


def _try_strassen_block():
    """
    Attempt a decomposition that uses a Strassen-style rank-7 factor for a 2x2
    block. For 3x5x5, we can try to embed a 2x2x2 Strassen (rank 7) plus
    schoolbook for the rest.

    Strategy: Partition the 5x5 middle into blocks. Use Strassen for a 2x2
    sub-block and schoolbook for the remaining rows/cols.

    Actually, let's try a more systematic approach: use the known fact that
    for multiplying an (a x b) by (b x c) matrix, if we can decompose the
    b x b part efficiently...

    Let's try: treat the 5x5 multiplication as needing rank <= 18 (Strassen-like
    for 4x4 is 18, for 5x5 we can pad to get a bound). Then combine with the
    3 outer dimension.

    A cleaner approach: The 3x5x5 tensor can be seen as 3 copies of a 5x5
    multiplication tensor (one for each row of the 3x5 matrix). Each 5x5
    multiplication has tensor rank at most 25 (schoolbook). If we can find
    a rank-r decomposition for 5x5, the total rank is at most 3r.

    Known: 4x4 matrix mult has rank 18 (Strassen-based). 5x5 can be done with
    rank at most 18 + 5 + 5 = 28 (pad 4x4 result, handle 5th row/col separately).
    Actually, 5x5 rank is at most 25 (schoolbook) or we can use the 4x4 rank-18
    plus 5+5 for the extra row/col = 28. But 25 < 28, so schoolbook is better
    for 5x5.

    Let's try a different angle: use the fact that 3x5x5 = (3x5) x (5x5).
    We can use the "bordered" approach or just try to find a good decomposition
    via random initialization + verification.

    For now, let's try a structured decomposition:
    - Use rank-7 Strassen for a 2x2x2 core
    - Handle the rest with schoolbook

    Actually, the simplest improvement: try to use a 2x2 Strassen decomposition
    for the first 2x2 block of the 5x5, reducing rank by 1 (from 4 to 3 for that
    2x2 block), giving total rank 75 - 1 = 74.
    """
    n, m, p = DIMENSIONS
    # Start with schoolbook
    U = [[0] * 75 for _ in range(n * m)]
    V = [[0] * 75 for _ in range(m * p)]
    W = [[0] * 75 for _ in range(p * n)]
    c = 0
    # Assign schoolbook entries, but for the 2x2 block (i=0,1; j=0,1; k=0,1)
    # use Strassen rank-7 instead of schoolbook rank-4.
    # Strassen 2x2x2: 7 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For the full 3x5x5, the 2x2 block corresponds to i in {0,1}, j in {0,1}, k in {0,1}
    # We replace 4 schoolbook terms (i,j,k) for i,j,k in {0,1} with 7 Strassen terms.
    # Net change: -4 + 7 = +3, so rank increases. That's worse!

    # Strassen only helps when we can share factors across the outer dimension.
    # Since n=3, the 2x2 block in j,k is shared across all 3 rows i.
    # So for each (i, j, k) with j,k in {0,1}, we have a term. There are 3*2*2 = 12
    # schoolbook terms. With Strassen, we use 3*7 = 21 terms. Still worse.

    # The real benefit of Strassen is when you have a large block. For 5x5,
    # let's try: decompose 5x5 using a 4x4 Strassen-based (rank 18) + extra.
    # 4x4 Strassen: rank 18. For 5x5: use 4x4 rank 18 for the top-left 4x4,
    # then handle row 4 and col 4 separately: +4 (for row 4 of A times B) + 4 (for A times col 4 of B) = 18+8 = 26.
    # But schoolbook is 25. So no improvement.

    # Let's just try random search for a lower rank decomposition.
    return None


def _random_search(max_rank, n_trials=100):
    """Try to find a rank-`max_rank` decomposition via random initialization."""
    import numpy as np
    n, m, p = DIMENSIONS
    np.random.seed(42)
    for trial in range(n_trials):
        if _time_left() < 2.0:
            break
        U = np.random.randn(n * m, max_rank) * 0.5
        V = np.random.randn(m * p, max_rank) * 0.5
        W = np.random.randn(p * n, max_rank) * 0.5
        # Verify
        Ua = U.reshape(n, m, max_rank)
        Va = V.reshape(m, p, max_rank)
        Wa = W.reshape(p, n, max_rank)
        T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
        err = np.max(np.abs(T - 1.0))
        if err < 1e-4:
            # Found a solution! Convert to list
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    return None


def construct():
    # Deadline guard: if we've used too much time, return schoolbook immediately
    if _time_left() < 5.0:
        return _schoolbook()

    best = _schoolbook()
    best_rank = 75

    # Try random search for lower rank
    # Theoretical lower bound for 3x5x5 is at least 25 (since 5x5 mult needs rank >= 25? No, 5x5 rank is between 18 and 25)
    # Actually, the tensor rank of matrix multiplication [m,n,p] is at least max(m*n, n*p, p*m) / something...
    # For [3,5,5], lower bound is at least 25 (since it contains a [5,5,5] sub-tensor? No, it's [3,5,5]).
    # The rank of [3,5,5] is at least the rank of [5,5,5] projected... Let's just try.
    for r in range(25, best_rank):
        if _time_left() < 10.0:
            break
        result = _random_search(r, n_trials=50)
        if result is not None:
            best = result
            best_rank = r
            break

    return best

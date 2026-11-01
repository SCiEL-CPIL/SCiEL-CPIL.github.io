"""Combined: Strassen-block core with shared A-side forms + exact verification."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, trials=8, tol=1e-9):
    """Check the decomposition reproduces A@B exactly on random instances."""
    rank = len(U[0]) if U else 0
    rng = np.random.RandomState(12345)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        # C_dec[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        C_dec = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    # dot over c
                    s += np.dot(u_row, np.multiply(v_row, w_row))
                C_dec[i, k] = s
        if not np.allclose(C, C_dec, atol=tol, rtol=0.0):
            return False
    return True


def _build_strassen_blocks_5x2_plus_1():
    """
    Known-good rank-39 decomposition: 5 Strassen 2x2x2 blocks (cols 0-9)
    + 4 schoolbook terms for col 10.
    Returns (U, V, W) as lists of lists, rank 39.
    """
    n, m, p = 2, 2, 11
    rank = 5 * 7 + 4
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    r = 0
    for b in range(5):
        v00 = 2 * b
        v01 = 2 * b + 1
        v10 = 11 + 2 * b
        v11 = 12 + 2 * b

        w00 = 4 * b
        w01 = 4 * b + 2
        w10 = 4 * b + 1
        w11 = 4 * b + 3

        # P1: (a00+a11)(b00+b11)
        U[0][r] += 1; U[3][r] += 1
        V[v00][r] += 1; V[v11][r] += 1
        W[w00][r] += 1; W[w11][r] += 1
        r += 1
        # P2: (a10+a11)b00
        U[2][r] += 1; U[3][r] += 1
        V[v00][r] += 1
        W[w10][r] += 1; W[w11][r] -= 1
        r += 1
        # P3: a00(b01-b11)
        U[0][r] += 1
        V[v01][r] += 1; V[v11][r] -= 1
        W[w01][r] += 1; W[w11][r] += 1
        r += 1
        # P4: a11(b10-b00)
        U[3][r] += 1
        V[v10][r] += 1; V[v00][r] -= 1
        W[w00][r] += 1; W[w10][r] += 1
        r += 1
        # P5: (a00+a01)b11
        U[0][r] += 1; U[1][r] += 1
        V[v11][r] += 1
        W[w00][r] -= 1; W[w01][r] += 1
        r += 1
        # P6: (a10-a00)(b00+b01)
        U[2][r] += 1; U[0][r] -= 1
        V[v00][r] += 1; V[v01][r] += 1
        W[w11][r] += 1
        r += 1
        # P7: (a01-a11)(b10+b11)
        U[1][r] += 1; U[3][r] -= 1
        V[v10][r] += 1; V[v11][r] += 1
        W[w00][r] += 1
        r += 1

    # Col 10 schoolbook
    U[0][r] += 1; V[10][r] += 1; W[20][r] += 1; r += 1
    U[1][r] += 1; V[21][r] += 1; W[20][r] += 1; r += 1
    U[2][r] += 1; V[10][r] += 1; W[21][r] += 1; r += 1
    U[3][r] += 1; V[21][r] += 1; W[21][r] += 1; r += 1

    return U, V, W


def _build_shared_strassen_2x2x11():
    """
    Structural refinement: use a SHARED set of 7 A-side bilinear forms across
    ALL 11 columns. Each form c contributes a product (A_form_c) * (B_linear_c).
    We express C[i,k] = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c} by letting
    U encode the A-side combination, V the B-side linear form (depends on k),
    W the output routing (depends on i,k).

    For 2x2x11, we keep the 7 Strassen A-side forms fixed and let the B-side
    linear forms be column-dependent. This gives rank 7 (one per A-form) IF the
    B-side can be made linear in each column independently. However, Strassen's
    7 forms are specifically designed for 2x2x2 where B has 2 columns. For 11
    columns we need more B-side degrees of freedom.

    A cleaner structural idea: treat the multiplication as 2 independent
    2x1x11 problems (row 0 and row 1 of A) coupled only through the A-side.
    Actually the cleanest exact reduction: the tensor for 2x2x11 has a known
    lower bound. We use a block-circulant / polynomial approach:

    Represent B as a 2x11 matrix. The product C = A*B. We can write
    C[i,k] = sum_j A[i,j] B[j,k]. This is bilinear in (A, B).

    The shared-form approach: define 7 A-side forms (Strassen) and for each
    column k, define B-side linear forms that depend on k. Since each A-form
    is a fixed linear combination of A entries, and each B-side is a fixed
    linear combination of B entries (per column), the total rank is 7 * (number
    of independent B-side patterns). But B-side for column k involves B[0,k]
    and B[1,k], so we need 2 B-side linear forms per column -> 7*2 = 14? No.

    Let me use the verified rank-39 as the safe baseline and attempt a
    lower-rank construction via a known 2x2xp formula. For p=11, a valid
    rank is at most 2*7 + 2*4 = 22? Let's just return the verified 39.
    """
    return _build_strassen_blocks_5x2_plus_1()


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Baseline: verified rank-39 Strassen-block decomposition.
    U, V, W = _build_strassen_blocks_5x2_plus_1()

    # Verify the baseline; if it passes, we have a valid score of 39.
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # If baseline somehow fails (should not), fall back to schoolbook rank 44.
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

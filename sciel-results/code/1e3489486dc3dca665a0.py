import time
import numpy as np

N = 4
M = 4
P = 8
DEADLINE_SEC = 110.0


def _strassen2():
    """Return the 7 Strassen factor triples (U, V, W) for 2x2x2 matrix product.

    Each factor is a length-4 vector indexed by the flattened (row, col) pair
    in the *output tensor* convention.  For the multiplication tensor
    T[a_idx, b_idx, c_idx] with a_idx = i*m + j, b_idx = j*p + k, and
    (official trace convention) c_idx = k*n + i, the rank-1 factor is the
    outer product  U[a_idx] * V[b_idx] * W[c_idx].

    The seven triples are the classical Strassen M1..M7 expressed on the
    (row, col) pairs of A (size 2x2), B (size 2x2) and the trace-convention
    C-index (also 2x2, row = k, col = i).
    """
    S = [
        # (U on (i,j), V on (j,k), W on (k,i))  -- each a list of 4 floats
        # M1 = (A00+A11)*(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)*B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3 = A00*(B01-B11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = A11*(B10-B00)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5 = (A00+A01)*B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A10-A00)*(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)*(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    return S


def _decompose_4x4x4():
    """Build the rank-49 Strassen decomposition of the 4x4x4 multiplication
    tensor using the 2x2x2 building block in a Kronecker-product (tensor)
    sense.  Returns three (16, 49) arrays."""
    S = _strassen2()
    n = 4
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)

    # Precompute the 16x4 Kronecker-basis for A-index (i,j) = (I,ii),(J,jj)
    # For each meta factor entry (bi,bj) with weight w, the contribution to
    # A-index (bi*2+ii, bj*2+jj) is w * inner[ii,jj].
    # We build a 16x4 matrix K_A where K_A[(bi*2+ii)*4 + (bj*2+jj), (bi,bj)] = inner[(ii,jj)].
    # But inner varies per term, so we just loop over the 49 combinations.
    for mt in range(7):
        for it in range(7):
            col = mt * 7 + it
            u_m = np.array(S[mt][0], dtype=np.int64).reshape(2, 2)
            u_i = np.array(S[it][0], dtype=np.int64).reshape(2, 2)
            v_m = np.array(S[mt][1], dtype=np.int64).reshape(2, 2)
            v_i = np.array(S[it][1], dtype=np.int64).reshape(2, 2)
            w_m = np.array(S[mt][2], dtype=np.int64).reshape(2, 2)
            w_i = np.array(S[it][2], dtype=np.int64).reshape(2, 2)

            # U: A[(bi*2+ii), (bj*2+jj)] with weight u_m[bi,bj]*u_i[ii,jj]
            u_full = np.zeros((4, 4), dtype=np.int64)
            for bi in range(2):
                for bj in range(2):
                    u_full[bi*2:bi*2+2, bj*2:bj*2+2] += u_m[bi, bj] * u_i
            U4[:, col] = u_full.reshape(16)

            # V: B[(bj*2+jj), (bk*2+kk)] with weight v_m[bj,bk]*v_i[jj,kk]
            v_full = np.zeros((4, 4), dtype=np.int64)
            for bj in range(2):
                for bk in range(2):
                    v_full[bj*2:bj*2+2, bk*2:bk*2+2] += v_m[bj, bk] * v_i
            V4[:, col] = v_full.reshape(16)

            # W: C-index = k*n + i = (bk*2+kk)*4 + (bi*2+ii)
            # weight w_m[bi,bk] * w_i[ii,kk]
            w_full = np.zeros((4, 4), dtype=np.int64)  # indexed [k, i]
            for bi in range(2):
                for bk in range(2):
                    w_full[bk*2:bk*2+2, bi*2:bi*2+2] += w_m[bi, bk] * w_i
            W4[:, col] = w_full.reshape(16)

    return U4, V4, W4


def _expand_to_4x4x8(U4, V4, W4):
    """Expand the rank-49 (4x4x4) decomposition to (4x4x8) by splitting
    the 8 output columns into two groups of 4.  Rank becomes 98."""
    n, m, p = N, M, P
    U49 = U4  # (16, 49)
    V49 = V4
    W49 = W4

    U = np.zeros((n * m, 98), dtype=np.int64)
    V = np.zeros((m * p, 98), dtype=np.int64)
    W = np.zeros((p * n, 98), dtype=np.int64)

    # First block: output columns k = 0..3
    U[:, 0:49] = U49
    # V: B[j, k] for k in 0..3 -> index j*8 + k
    V[:, 0:49] = 0
    for j in range(m):
        for k in range(4):
            V[j * p + k, 0:49] = V49[j * 4 + k, :]
    # W: C-index = k*4 + i for k in 0..3
    for k in range(4):
        for i in range(n):
            W[k * n + i, 0:49] = W49[k * 4 + i, :]

    # Second block: output columns k = 4..7
    U[:, 49:98] = U49
    for j in range(m):
        for k in range(4, 8):
            V[j * p + k, 49:98] = V49[j * 4 + (k - 4), :]
    for k in range(4, 8):
        for i in range(n):
            W[k * n + i, 49:98] = W49[(k - 4) * 4 + i, :]

    return U, V, W


def _verify(U, V, W):
    """Quick numerical verification with a couple of random inputs."""
    n, m, p = N, M, P
    rng = np.random.RandomState(7)
    for _ in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_true = A @ B
        C_app = np.zeros((n, p), dtype=float)
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            C_app += np.outer((u * A).sum(axis=1), (v * B).sum(axis=0)) * w.T
        if not np.allclose(C_app, C_true, atol=1e-8):
            return False
    return True


def _schoolbook():
    n, m, p = N, M, P
    R = n * m * p
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    t0 = time.time()

    U4, V4, W4 = _decompose_4x4x4()
    U, V, W = _expand_to_4x4x8(U4, V4, W4)

    if _verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist(),
        }

    # Fallback: schoolbook rank 128
    Us, Vs, Ws = _schoolbook()
    return {
        "U": Us.tolist(),
        "V": Vs.tolist(),
        "W": Ws.tolist(),
    }

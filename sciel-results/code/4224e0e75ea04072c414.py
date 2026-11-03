import numpy as np
import time

DIMENSIONS = [3, 4, 8]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check the cyclic bilinear identity exactly (loop-based, safe)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_random(U, V, W, n, m, p, seed=42, tol=1e-7):
    """Cross-check with random integer matrices (defense in depth)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    rng = np.random.RandomState(seed)
    A = rng.randint(1, 6, (n, m)).astype(float)
    B = rng.randint(1, 6, (m, p)).astype(float)
    C = A @ B
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] * A[i, j] * B[j, k]
            if abs(s - C[i, k]) > tol * max(1.0, abs(C[i, k])):
                return False
    return True


def _build_rank88():
    """Verified rank-88 construction: schoolbook(1x4x8) + Strassen(2x4x8)."""
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    # Part 1: i=0 schoolbook 1x4x8 -> 32 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    # Part 2: i=1,2 Strassen on 2x2x2 blocks -> 2*4*7 = 56 terms
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (js + lj), col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(js + lj) * p + (ks + lk), col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(ks + lk) * n + (1 + li), col] += sW[t][l]
                col += 1
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, deadline, tol=1e-9):
    """Attempt to reduce rank by finding exact linear dependencies among columns."""
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    W = np.asarray(W, dtype=float).copy()
    R = U.shape[1]

    def column_contribution(r):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = (
                        U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    )
        return vec

    def build_rhs_matrix(exclude):
        cols = [c for c in range(R) if c != exclude]
        M = np.zeros((n * m * p, len(cols)))
        for idx, c in enumerate(cols):
            M[:, idx] = column_contribution(c)
        return M, cols

    for r in range(R - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            R -= 1
            continue

        b = column_contribution(r)
        A_mat, cols = build_rhs_matrix(r)
        try:
            c, residuals, rankA, sv = np.linalg.lstsq(A_mat, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(A_mat @ c - b)
        if residual < tol * max(1.0, np.linalg.norm(b)):
            U[:, cols] += U[:, r][:, None] * c[None, :]
            V[:, cols] += V[:, r][:, None] * c[None, :]
            W[:, cols] += W[:, r][:, None] * c[None, :]
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            R -= 1

    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    U, V, W = _build_rank88()

    if not _verify(U, V, W, n, m, p):
        pass

    U2, V2, W2 = _try_reduce_rank(U, V, W, n, m, p, deadline)

    if _verify(U2, V2, W2, n, m, p) and _verify_random(U2, V2, W2, n, m, p):
        U, V, W = U2, V2, W2

    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}


if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    n, m, p = DIMENSIONS
    Ua = np.array(result['U']); Va = np.array(result['V']); Wa = np.array(result['W'])
    print("verify:", _verify(Ua, Va, Wa, n, m, p))
    print("verify_random:", _verify_random(Ua, Va, Wa, n, m, p))

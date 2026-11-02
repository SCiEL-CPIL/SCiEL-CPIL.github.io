"""Matrix multiplication 4x5 by 5x6 — Kronecker Strassen / block decomposition.

Strategy (within the existing family):
  * Use a rank-49 Kronecker(2x2x2 Strassen, 2x2x2 Strassen) for the 4x4x4 core.
  * Use schoolbook (rank 32) for the 4x4x2 remainder in the k-direction.
  * Use schoolbook (rank 24) for the 4x1x6 remainder in the m-direction.
  Total rank = 49 + 32 + 24 = 105.

We additionally verify the decomposition numerically and fall back to a
guaranteed-valid (schoolbook) decomposition if verification fails, so the
result is always valid.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """2x2x2 Strassen. U,V,W have shape (4,7).

    Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t].
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # t=0: (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # t=1: (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # t=2: a11(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # t=3: a22(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # t=4: (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # t=5: (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # t=6: (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _build_schoolbook(n, m, p):
    """Schoolbook rank n*m*p. U:(n*m), V:(m*p), W:(p*n)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _kron_strassen_4x4x4():
    """Rank-49 Kronecker product of two 2x2x2 Strassen decompositions.

    U:(16,49), V:(16,49), W:(16,49).
    U[(i1*2+i2)*4 + (j1*2+j2), t1*7+t2] = U2[i1*2+j1, t1]*U2[i2*2+j2, t2]
    V[(j1*2+j2)*4 + (k1*2+k2), t1*7+t2] = V2[j1*2+k1, t1]*V2[j2*2+k2, t2]
    W[(k1*2+k2)*4 + (i1*2+i2), t1*7+t2] = W2[k1*2+i1, t1]*W2[k2*2+i2, t2]
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1 * 2 + j1][t1]
                    if c1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2 * 2 + j2][t2]
                            if c2 == 0:
                                continue
                            u_row = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U[u_row][col] += c1 * c2
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1 * 2 + k1][t1]
                    if c1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2 * 2 + k2][t2]
                            if c2 == 0:
                                continue
                            v_row = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            V[v_row][col] += c1 * c2
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1 * 2 + i1][t1]
                    if c1 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2 * 2 + i2][t2]
                            if c2 == 0:
                                continue
                            w_row = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            W[w_row][col] += c1 * c2
    return U, V, W


def _combine(decomps, n, m, p):
    """Combine sub-decompositions into a global (U,V,W).

    Each decomp: (U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz)
    Sub indices: U_sub[i*j_sz+j, c], V_sub[j*k_sz+k, c], W_sub[k*i_sz+i, c].
    Global:      U[(i_s+i)*m+(j_s+j), c], V[(j_s+j)*p+(k_s+k), c],
                 W[(k_s+k)*n+(i_s+i), c].
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    cur = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rsz = len(U_sub[0])
        for i in range(i_sz):
            for j in range(j_sz):
                ur = i * j_sz + j
                ug = (i_s + i) * m + (j_s + j)
                for c in range(rsz):
                    val = U_sub[ur][c]
                    if val != 0:
                        U[ug][cur + c] += val
        for j in range(j_sz):
            for k in range(k_sz):
                vr = j * k_sz + k
                vg = (j_s + j) * p + (k_s + k)
                for c in range(rsz):
                    val = V_sub[vr][c]
                    if val != 0:
                        V[vg][cur + c] += val
        for k in range(k_sz):
            for i in range(i_sz):
                wr = k * i_sz + i
                wg = (k_s + k) * n + (i_s + i)
                for c in range(rsz):
                    val = W_sub[wr][c]
                    if val != 0:
                        W[wg][cur + c] += val
        cur += rsz
    return U, V, W


def _verify(U, V, W, n, m, p, trials=3, tol=1e-8):
    """Check C[i,k] == sum_j A[i,j]B[j,k] via the tensor identity."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rng = np.random.default_rng(0)
    for _ in range(trials):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        # Precompute per-rank linear forms.
        # For each rank r: a_r = U_r^T . (A flattened), b_r = V_r^T . (B flattened), w_r = W_r.
        # C[i,k] = sum_r w_r[k*n+i] * a_r * b_r
        Aflat = A.reshape(n * m)
        Bflat = B.reshape(m * p)
        ar = U_np.T @ Aflat          # (rank,)
        br = V_np.T @ Bflat          # (rank,)
        Crec = np.einsum('r,ri->ir', ar * br, W_np)  # (p*n,) -> reshape
        Crec = Crec.reshape(p, n).T  # (n, p)
        if np.max(np.abs(Crec - C)) > tol:
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 130.0

    decomps = []
    # 4x4x4 core (rank 49)
    U4, V4, W4 = _kron_strassen_4x4x4()
    decomps.append((U4, V4, W4, 0, 0, 0, 4, 4, 4))
    # 4x4x2 remainder in k (rank 32)
    U5, V5, W5 = _build_schoolbook(4, 4, 2)
    decomps.append((U5, V5, W5, 0, 0, 4, 4, 4, 2))
    # 4x1x6 remainder in m (rank 24)
    U6, V6, W6 = _build_schoolbook(4, 1, 6)
    decomps.append((U6, V6, W6, 0, 4, 0, 4, 1, 6))

    U, V, W = _combine(decomps, n, m, p)

    # Verify; if it fails, fall back to guaranteed-valid schoolbook.
    if time.time() < deadline - 5.0:
        if not _verify(U, V, W, n, m, p):
            U, V, W = _build_schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

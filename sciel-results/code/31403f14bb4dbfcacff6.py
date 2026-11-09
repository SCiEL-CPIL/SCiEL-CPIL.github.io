import time
import numpy as np

DIMENSIONS = [4, 4, 5]

DEADLINE_FRAC = 0.90
START = time.time()

def _deadline():
    return time.time() - START > 150.0 * DEADLINE_FRAC

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    A = np.random.default_rng(42).random((n, m))
    B = np.random.default_rng(123).random((m, p))
    Ctrue = A @ B
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)

def _make_2x2x2_strassen():
    """Build U(4,7), V(4,7), W(4,7) for 2x2x2 Strassen."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    # A: 0=a00, 1=a01, 2=a10, 3=a11
    # B: 0=b00, 1=b01, 2=b10, 3=b11
    # C: 0=c00, 1=c01, 2=c10, 3=c11
    # M1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = a00(b01-b11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = a11(b10-b00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[3,3]=-1
    # M5 = (a00+a10)b11
    U[0,4]=1; U[2,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (a01-a00)b10
    U[1,5]=1; U[0,5]=-1; V[2,5]=1; W[1,5]=1
    # M7 = (a11-a10)(b00+b01)
    U[3,6]=1; U[2,6]=-1; V[0,6]=1; V[1,6]=1; W[0,6]=1; W[3,6]=1
    return U, V, W

def _make_4x4x4_strassen():
    """4x4x4 Strassen via 2-level blocks. Rank 49."""
    n = m = p = 4
    U = np.zeros((n * m, 49), dtype=float)
    V = np.zeros((m * p, 49), dtype=float)
    W = np.zeros((p * n, 49), dtype=float)
    inner_U, inner_V, inner_W = _make_2x2x2_strassen()
    # Top-level Strassen factors: (a_block_coefs[2][2], b_block_coefs[2][2])
    top_factors = [
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])),   # M1
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]])),   # M2
        (np.array([[1,0],[0,0]]), np.array([[0,1],[0,-1]])),  # M3
        (np.array([[0,0],[0,1]]), np.array([[0,-1],[1,0]])),  # M4
        (np.array([[1,0],[1,0]]), np.array([[0,0],[0,1]])),   # M5
        (np.array([[0,1],[-1,0]]), np.array([[0,0],[1,0]])),  # M6
        (np.array([[0,0],[1,-1]]), np.array([[1,1],[0,0]])),  # M7
    ]
    # C_block = sum_t sign * Mt
    c_comb = {
        (0,0): [(0,1),(3,1),(4,-1),(6,1)],
        (0,1): [(2,1),(5,1)],
        (1,0): [(1,1),(4,1)],
        (1,1): [(0,1),(1,-1),(2,1),(6,1)],
    }
    for t in range(7):
        a_coefs, b_coefs = top_factors[t]
        for tp in range(7):
            col = t * 7 + tp
            for bi in range(2):
                for bj in range(2):
                    coef = a_coefs[bi][bj]
                    if coef == 0:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            gi = bi * 2 + ii
                            gj = bj * 2 + jj
                            U[gi * m + gj, col] += coef * inner_U[ii * 2 + jj, tp]
            for bi in range(2):
                for bj in range(2):
                    coef = b_coefs[bi][bj]
                    if coef == 0:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            gi = bi * 2 + ii
                            gj = bj * 2 + jj
                            V[gi * p + gj, col] += coef * inner_V[ii * 2 + jj, tp]
            for bi in range(2):
                for bj in range(2):
                    for (ft, sign) in c_comb[(bi,bj)]:
                        if ft == t:
                            for ii in range(2):
                                for jj in range(2):
                                    gi = bi * 2 + ii
                                    gj = bj * 2 + jj
                                    W[gj * n + gi, col] += sign * inner_W[jj * 2 + ii, tp]
    return U, V, W

def _build_445():
    """4x4x5: Strassen 4x4x4 (rank 49) + schoolbook k=4 (rank 16) = rank 65."""
    n, m, p = 4, 4, 5
    U4, V4, W4 = _make_4x4x4_strassen()
    R4 = U4.shape[1]
    extra = n * m
    R = R4 + extra
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    U[:, :R4] = U4
    V[:, :R4] = V4
    W[:, :R4] = W4
    k = 4
    c = R4
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            c += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    best = None
    Us, Vs, Ws = _schoolbook(n, m, p)
    if _verify(Us, Vs, Ws, n, m, p):
        best = (Us, Vs, Ws, Us.shape[1])
    if not _deadline():
        try:
            U2, V2, W2 = _build_445()
            if _verify(U2, V2, W2, n, m, p):
                if best is None or U2.shape[1] < best[3]:
                    best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass
    if best is None:
        best = (Us, Vs, Ws, Us.shape[1])
    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

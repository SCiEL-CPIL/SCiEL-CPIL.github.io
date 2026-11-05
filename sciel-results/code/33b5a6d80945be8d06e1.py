import time
import numpy as np
import itertools

_START = time.time()
_DEADLINE = 118.0

def _left():
    return _DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        T += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k] - 1.0) > 1e-7:
                    return False
    return True

def _rank_of(A, tol=1e-8):
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except Exception:
        return 0
    return int(np.sum(s > tol * (s[0] if s.size else 1.0)))

def construct():
    n, m, p = 2, 4, 8
    NM, MP, PN = n*m, m*p, p*n  # 8, 32, 16
    school = n*m*p  # 64

    def build_sch():
        U = np.zeros((NM, school)); V = np.zeros((MP, school)); W = np.zeros((PN, school))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1; V[j*p+k, r] = 1; W[k*n+i, r] = 1; r += 1
        return U, V, W

    def build_strassen():
        U = np.zeros((NM, 56)); V = np.zeros((MP, 56)); W = np.zeros((PN, 56))
        col = 0
        for jb in (0, 2):
            for kb in (0, 2, 4, 6):
                a00, a01 = 0+jb, 1+jb
                a10, a11 = 1+jb, 2+jb
                b00, b01 = 0+jb, 1+jb
                b10, b11 = 1+jb, 2+jb
                k0, k1 = 0+kb, 1+kb
                def term(u, v, w):
                    nonlocal col
                    for (i,j),x in u.items(): U[i*m+j, col] += x
                    for (j,k),x in v.items(): V[j*p+k, col] += x
                    for (k,i),x in w.items(): W[k*n+i, col] += x
                    col += 1
                term({(0,a00):1,(1,a11):1}, {(b00,k0):1,(b11,k1):1}, {(k0,0):1,(k1,1):1})
                term({(1,a10):1,(1,a11):1}, {(b00,k0):1}, {(k0,1):1,(k1,1):-1})
                term({(0,a00):1}, {(b01,k1):1,(b11,k1):-1}, {(k1,0):1,(k1,1):1})
                term({(1,a11):1}, {(b10,k0):1,(b00,k0):-1}, {(k0,0):1,(k0,1):1})
                term({(0,a00):1,(0,a01):1}, {(b11,k1):1}, {(k0,0):-1,(k1,0):1})
                term({(1,a10):1,(0,a00):-1}, {(b00,k0):1,(b01,k1):1}, {(k1,1):1})
                term({(0,a01):1,(1,a11):-1}, {(b10,k0):1,(b11,k1):1}, {(k0,0):1})
        return U, V, W

    U0, V0, W0 = build_strassen()
    if not _verify(U0, V0, W0, n, m, p):
        U0, V0, W0 = build_sch()

    # Measure the true tensor rank lower bound from the three unfoldings.
    # If min unfolding rank < 56, a lower-rank decomposition exists.
    def unfold_mode1(U, V, W):
        # (NM) x (MP * PN)  ; element = sum_r U[i,r] V[j,r] W[k,r] with (j,k) index
        M = np.zeros((NM, MP*PN))
        for r in range(U.shape[1]):
            M += np.outer(U[:, r], (V[:, r] * W[:, r]))
        return M
    def unfold_mode2(U, V, W):
        M = np.zeros((MP, NM*PN))
        for r in range(U.shape[1]):
            M += np.outer(V[:, r], (U[:, r] * W[:, r]))
        return M
    def unfold_mode3(U, V, W):
        M = np.zeros((PN, NM*MP))
        for r in range(U.shape[1]):
            M += np.outer(W[:, r], (U[:, r] * V[:, r]))
        return M

    best = (56, U0, V0, W0)
    try:
        if _left() > 30:
            r1 = _rank_of(unfold_mode1(U0, V0, W0))
            r2 = _rank_of(unfold_mode2(U0, V0, W0))
            r3 = _rank_of(unfold_mode3(U0, V0, W0))
            lb = max(r1, r2, r3)
            # Tensor rank >= each unfolding rank. If lb < 56 there is headroom,
            # but we still need an explicit decomposition; keep the verified one.
    except Exception:
        pass

    return {"U": best[1].tolist(), "V": best[2].tolist(), "W": best[3].tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

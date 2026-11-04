import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0

    # Tensor T[3a+b, 7b+c, 3c+a] = 1
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m+b, b*p+c, c*n+a] = 1.0

    def verify(U, V, W):
        try:
            return np.allclose(np.einsum('ir,jr,kr->ijk', U, V, W), T, atol=1e-6)
        except:
            return False

    def schoolbook():
        R = n*m*p
        U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
        col = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    U[a*m+b, col] = 1.0; V[b*p+c, col] = 1.0; W[c*n+a, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()

    def als(T, R, U0, V0, W0, max_iters=15):
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        for it in range(max_iters):
            if time.time() - start_time > deadline - 10: break
            for i in range(T.shape[0]):
                M = np.outer(V.ravel(), W.ravel()).reshape(T.shape[1]*T.shape[2], R)
                U[i] = np.linalg.lstsq(M, T[i].ravel(), rcond=None)[0]
            for j in range(T.shape[1]):
                M = np.outer(U.ravel(), W.ravel()).reshape(T.shape[0]*T.shape[2], R)
                V[j] = np.linalg.lstsq(M, T[:,j,:].ravel(), rcond=None)[0]
            for k in range(T.shape[2]):
                M = np.outer(U.ravel(), V.ravel()).reshape(T.shape[0]*T.shape[1], R)
                W[k] = np.linalg.lstsq(M, T[:,:,k].ravel(), rcond=None)[0]
        return U, V, W

    best = None
    for R in [55, 50, 45]:
        if time.time() - start_time > deadline - 20: break
        for seed in range(5):
            if time.time() - start_time > deadline - 20: break
            rng = np.random.RandomState(seed)
            cols = rng.choice(n*m*p, R, replace=False)
            U, V, W = als(T, R, U_sb[:,cols], V_sb[:,cols], W_sb[:,cols])
            Ur, Vr, Wr = np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2
            if verify(Ur, Vr, Wr):
                best = (Ur, Vr, Wr)
                break
        if best: break

    if best:
        return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [3, 3, 6]
START = time.time()
DEADLINE = 150.0 - 5.0


def _ok():
    return time.time() < DEADLINE


def _verify_3x6(U, V, W, iters=12, seed=555):
    n, m, p = 3, 3, 6
    R = U.shape[1]
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-4, 5, size=(n, m))
        b = rng.integers(-4, 5, size=(m, p))
        C = a @ b
        Cc = np.zeros((n, p))
        for c in range(R):
            ua = np.array([[U[i * m + k, c] for k in range(m)] for i in range(n)])
            vb = np.array([[V[k * p + j, c] for j in range(p)] for k in range(m)])
            wa = np.array([[W[j * n + i, c] for i in range(n)] for j in range(p)])
            Cc += float(np.sum(ua * a)) * float(np.sum(vb * b)) * wa
        if not np.allclose(Cc, C, atol=1e-9, rtol=0.0):
            return False
    return True


def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def _find_3x3x2_decomp(rank, seed, trials=500):
    """Find rank decomposition for 3x3 * 3x2 matrix multiplication."""
    n, m, p = 3, 3, 2
    rng = np.random.default_rng(seed)
    
    for trial in range(trials):
        if not _ok():
            return None
        
        # Random U and V with small integer/half-integer values
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = rng.choice(vals, size=(rank, n, m))  # U[t, i, k]
        V = rng.choice(vals, size=(rank, m, p))  # V[t, k, j]
        
        # Build linear system for W
        W = np.zeros((rank, n, p))
        valid = True
        
        for i in range(n):
            for j in range(p):
                A = np.zeros((m * m, rank))
                b = np.zeros(m * m)
                for k in range(m):
                    for kp in range(m):
                        A[k * m + kp, :] = U[:, i, k] * V[:, kp, j]
                        b[k * m + kp] = 1.0 if k == kp else 0.0
                
                try:
                    W_col, *_ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                W_col_rounded = np.round(W_col * 2) / 2
                W[:, i, j] = W_col_rounded
                
                if not np.allclose(A @ W_col_rounded, b, atol=1e-9, rtol=0.0):
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Vectorized verification
        check = True
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    for kp in range(m):
                        s = np.sum(U[:, i, k] * V[:, kp, j] * W[:, i, j])
                        expected = 1.0 if k == kp else 0.0
                        if abs(s - expected) > 1e-9:
                            check = False
                            break
                    if not check:
                        break
                if not check:
                    break
            if not check:
                break
        
        if check:
            return U.astype(float), V.astype(float), W.astype(float)
    
    return None


def _build_3x6_from_3x3x2(r, d0, d1, d2):
    n, m, p = 3, 3, 6
    R = 3 * r
    U = np.zeros((9, R))
    V = np.zeros((18, R))
    W = np.zeros((18, R))
    for q, (u, v, w) in enumerate((d0, d1, d2)):
        base = q * r
        for t in range(r):
            col = base + t
            for i in range(3):
                for k in range(3):
                    U[i * 3 + k, col] = u[t, i, k]
            for k in range(3):
                for j in range(2):
                    V[k * 6 + (q * 2 + j), col] = v[t, k, j]
            for i in range(3):
                for j in range(2):
                    W[(q * 2 + j) * 3 + i, col] = w[t, i, j]
    return U, V, W


def construct():
    # The random search for 3x3x2 decompositions is extremely unlikely to succeed
    # with the given parameter values and trial counts. Return schoolbook immediately.
    return _schoolbook()

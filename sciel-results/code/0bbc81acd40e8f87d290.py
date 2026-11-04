import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 130.0

def _strassen_2x2():
    """Strassen coefficients for 2x2 x 2x2 -> 2x2."""
    # M[0] = (A00+A11)(B00+B11)
    # M[1] = (A10+A11)B00
    # M[2] = A00(B01-B11)
    # M[3] = A11(B10-B00)
    # M[4] = (A00+A01)B11
    # M[5] = (A10-A00)(B00+B01)
    # M[6] = (A01-A11)(B10+B11)
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]
    return alpha, beta, recon

def _build_77():
    """77-rank two-level Strassen for 4x4 * 4x6."""
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    alpha, beta, recon = _strassen_2x2()
    
    col = 0
    for s in range(7):
        # Inner Strassen for 2x2 x 2x2 (cols 0,1 of the 2x3 block)
        for si in range(7):
            for a in range(4):
                ca = alpha[si][a]
                if ca == 0:
                    continue
                ar, ac = a // 2, a % 2
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    rr, rc = r // 2, r % 2
                    u_idx = (rr * 2 + ar) * 4 + (rc * 2 + ac)
                    U[u_idx, col] += ca * cr
            
            for b in range(4):
                cb = beta[si][b]
                if cb == 0:
                    continue
                br, bc = b // 2, b % 2
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    cr, cc2 = c // 2, c % 2
                    v_idx = (cr * 2 + br) * 6 + (cc2 * 3 + bc)
                    V[v_idx, col] += cb * cc
            
            for (a, b, ga) in recon[s]:
                for (za, zb, gz) in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += ga * gz
            
            col += 1
        
        # Schoolbook for 2x2 x 2x1 (col 2 of the 2x3 block)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    rr, rc = r // 2, r % 2
                    u_idx = (rr * 2 + u) * 4 + (rc * 2 + v)
                    U[u_idx, col] += cr
                
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    cr, cc2 = c // 2, c % 2
                    v_idx = (cr * 2 + v) * 6 + (cc2 * 3 + 2)
                    V[v_idx, col] += cc
                
                for (a, b, ga) in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += ga
                
                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, R, trials=3):
    for _ in range(trials):
        A = np.random.randint(-3, 3, (n, m)).astype(float)
        B = np.random.randint(-3, 3, (m, p)).astype(float)
        C_exp = A @ B
        C_rec = np.zeros((n, p))
        for i in range(n):
            for r in range(p):
                for k in range(m):
                    C_rec[i, r] += np.dot(U[i*m+k, :], V[k*p+r, :] * W[r*n+i, :])
        if not np.allclose(C_rec, C_exp, atol=1e-9):
            return False
    return True

def _schoolbook():
    n, m, p = 4, 4, 6
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    idx = 0
    for i in range(n):
        for k in range(m):
            for r in range(p):
                U[i * m + k, idx] = 1
                V[k * p + r, idx] = 1
                W[r * n + i, idx] = 1
                idx += 1
    return U, V, W

def construct():
    n, m, p = 4, 4, 6
    
    try:
        U, V, W = _build_77()
        R = U.shape[1]
        
        if time.time() - START_TIME < TIME_LIMIT - 10:
            if _verify(U, V, W, n, m, p, R, trials=3):
                def to_list(mat):
                    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat.tolist()]
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    except Exception:
        pass
    
    U, V, W = _schoolbook()
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat.tolist()]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

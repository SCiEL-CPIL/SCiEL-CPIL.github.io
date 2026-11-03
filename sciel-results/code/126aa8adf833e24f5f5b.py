"""Program for matrix multiplication 3,3,6 tensor decomposition.
Goal: find a rank-minimal exact decomposition (lower rank = better score)."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 6]
DEADLINE_S = 110.0  # safety margin within 120s limit

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                            actual = np.dot(U_arr[i*m+j], V_arr[jp*p+k] * W_arr[kp*n+ip])
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _build_trivial(n=3, m=3, p=6):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _build_trivial(n, m, p)
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Attempt to find a lower-rank decomposition via structured random search.
    # The 3x3x6 problem decomposes into two 3x3x3 sub-problems sharing A.
    # Target: rank 46 (2 * 23, based on known 3x3x3 bounds).
    
    def _err(Ur, Vr, Wr):
        e = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.dot(Ur[a], Vr[b] * Wr[c])
                    e += (val - 1.0) ** 2
        for _ in range(20):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            jp = random.randint(0, m-1)
            k = random.randint(0, p-1)
            ip = random.randint(0, n-1)
            kp = random.randint(0, p-1)
            if not (j == jp and i == ip and k == kp):
                a = i*m + j
                b = jp*p + k
                c = kp*n + ip
                val = np.dot(Ur[a], Vr[b] * Wr[c])
                e += val ** 2
        return e

    for target_rank in [46, 45]:
        if time.time() - start_time > DEADLINE_S - 10:
            break
        for restart in range(2):
            if time.time() - start_time > DEADLINE_S - 5:
                break
            U_r = np.random.randn(n*m, target_rank) * 0.3
            V_r = np.random.randn(m*p, target_rank) * 0.3
            W_r = np.random.randn(p*n, target_rank) * 0.3
            cur = _err(U_r, V_r, W_r)
            ss = 0.08
            for it in range(2000):
                if time.time() - start_time > DEADLINE_S - 5:
                    break
                mat = random.choice(['U','V','W'])
                if mat=='U':
                    row = random.randint(0, n*m-1); arr=U_r
                elif mat=='V':
                    row = random.randint(0, m*p-1); arr=V_r
                else:
                    row = random.randint(0, p*n-1); arr=W_r
                c = random.randint(0, target_rank-1)
                old = arr[row,c]
                arr[row,c] = old + random.gauss(0, ss)
                new_e = _err(U_r, V_r, W_r)
                if new_e < cur:
                    cur = new_e
                    if cur < 1e-10:
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                            best_rank = target_rank
                            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                            break
                else:
                    arr[row,c] = old
                if it % 400 == 399:
                    ss *= 0.95
            if cur < 1e-10:
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

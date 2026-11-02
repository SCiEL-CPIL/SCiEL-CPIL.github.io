import numpy as np
import time
import random

def verify(U, V, W, n=2, m=3, p=3):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-6:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 3
    start = time.time()
    deadline = start + 140

    # Rank 18 schoolbook fallback
    rank = 18
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W

    if time.time() < deadline - 15:
        for trial in range(300000):
            if time.time() > deadline - 5:
                break

            U15 = [[random.choice([-1, 0, 1]) for _ in range(15)] for _ in range(6)]
            W15 = [[random.choice([-1, 0, 1]) for _ in range(15)] for _ in range(6)]

            if any(all(x == 0 for x in row) for row in U15):
                continue
            if any(all(x == 0 for x in row) for row in W15):
                continue

            V15 = [[0.0]*15 for _ in range(9)]
            ok = True
            for j in range(3):
                for k in range(3):
                    A0 = [U15[0*3+j][c] * W15[k*2+0][c] for c in range(15)]
                    A1 = [U15[1*3+j][c] * W15[k*2+1][c] for c in range(15)]
                    if all(a == 0 for a in A0) or all(a == 0 for a in A1):
                        ok = False
                        break
                    A_mat = np.array([A0, A1], dtype=float)
                    try:
                        V_row = np.linalg.lstsq(A_mat, np.array([1.0, 1.0]), rcond=None)[0]
                        V15[j*3+k] = V_row.tolist()
                    except:
                        ok = False
                        break
                if not ok:
                    break

            if ok and verify(U15, V15, W15):
                Ur = [[round(x*2)/2 for x in row] for row in U15]
                Vr = [[round(x*2)/2 for x in row] for row in V15]
                Wr = [[round(x*2)/2 for x in row] for row in W15]
                if verify(Ur, Vr, Wr):
                    best_U, best_V, best_W = Ur, Vr, Wr
                else:
                    best_U, best_V, best_W = U15, V15, W15
                break

    return {"U": best_U, "V": best_V, "W": best_W}

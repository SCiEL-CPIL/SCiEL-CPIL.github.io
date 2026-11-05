import time
import numpy as np

N, M, P = 2, 4, 8
_START = time.time()
_DEADLINE = 148.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

def construct():
    # Rank-56 decomposition for 2x4x8 matrix multiplication.
    # Structure: 4 independent 2x4x2 blocks, each decomposed via Strassen into 14 terms.
    # Total rank = 4 * 14 = 56.
    rank = 56
    U = [[0] * rank for _ in range(N * M)]   # shape (8, 56)
    V = [[0] * rank for _ in range(M * P)]   # shape (32, 56)
    W = [[0] * rank for _ in range(P * N)]   # shape (16, 56)

    def _add(col, uc, vc, wc):
        for (i, j), v in uc.items():
            if v:
                U[i * M + j][col] += v
        for (j, k), v in vc.items():
            if v:
                V[j * P + k][col] += v
        for (k, i), v in wc.items():
            if v:
                W[k * N + i][col] += v

    col = 0
    for k0 in (0, 2, 4, 6):
        k1 = k0 + 1
        # Left half: A[:,0:2] * B[:,k0:k0+2]
        _add(col, {(0,0):1,(1,1):1}, {(0,k0):1,(1,k1):1}, {(k0,0):1,(k1,1):1}); col+=1
        _add(col, {(1,0):1,(1,1):1}, {(0,k0):1}, {(k0,1):1,(k1,1):-1}); col+=1
        _add(col, {(0,0):1}, {(0,k1):1,(1,k1):-1}, {(k1,0):1,(k1,1):1}); col+=1
        _add(col, {(1,1):1}, {(1,k0):1,(0,k0):-1}, {(k0,0):1,(k0,1):1}); col+=1
        _add(col, {(0,0):1,(0,1):1}, {(1,k1):1}, {(k0,0):-1,(k1,0):1}); col+=1
        _add(col, {(1,0):1,(0,0):-1}, {(0,k0):1,(0,k1):1}, {(k1,1):1}); col+=1
        _add(col, {(0,1):1,(1,1):-1}, {(1,k0):1,(1,k1):1}, {(k0,0):1}); col+=1
        # Right half: A[:,2:4] * B[:,k0:k0+2]
        _add(col, {(0,2):1,(1,3):1}, {(2,k0):1,(3,k1):1}, {(k0,0):1,(k1,1):1}); col+=1
        _add(col, {(1,2):1,(1,3):1}, {(2,k0):1}, {(k0,1):1,(k1,1):-1}); col+=1
        _add(col, {(0,2):1}, {(2,k1):1,(3,k1):-1}, {(k1,0):1,(k1,1):1}); col+=1
        _add(col, {(1,3):1}, {(3,k0):1,(2,k0):-1}, {(k0,0):1,(k0,1):1}); col+=1
        _add(col, {(0,2):1,(0,3):1}, {(3,k1):1}, {(k0,0):-1,(k1,0):1}); col+=1
        _add(col, {(1,2):1,(0,2):-1}, {(2,k0):1,(2,k1):1}, {(k1,1):1}); col+=1
        _add(col, {(0,3):1,(1,3):-1}, {(3,k0):1,(3,k1):1}, {(k0,0):1}); col+=1

    # Validate with numpy if time permits
    if _time_left() > 2.0:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        target = np.zeros((N, M, P))
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    target[i, j, k] = 1.0
        if not np.allclose(recon, target, atol=1e-9):
            # Fallback: schoolbook
            rank = N * M * P
            U = [[0] * rank for _ in range(N * M)]
            V = [[0] * rank for _ in range(M * P)]
            W = [[0] * rank for _ in range(P * N)]
            c = 0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        U[i*M+j][c] = 1
                        V[j*P+k][c] = 1
                        W[k*N+i][c] = 1
                        c += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

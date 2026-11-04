import time
import numpy as np

START = time.time()
DEADLINE = START + 135.0

def _build_strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _build_rank98():
    U2, V2, W2 = _build_strassen_2x2x2()
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    U[:, 0:49] = U4
    for j in range(4):
        for kl in range(4):
            V[j*8 + kl, 0:49] = V4[j*4 + kl, :]
    for kl in range(4):
        for i in range(4):
            W[kl*4 + i, 0:49] = W4[kl*4 + i, :]
            
    U[:, 49:98] = U4
    for j in range(4):
        for kl in range(4):
            kg = 4 + kl
            V[j*8 + kg, 49:98] = V4[j*4 + kl, :]
    for kl in range(4):
        kg = 4 + kl
        for i in range(4):
            W[kg*4 + i, 49:98] = W4[kl*4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W):
    U_np = np.array(U, dtype=np.float64).reshape(4, 4, -1)
    V_np = np.array(V, dtype=np.float64).reshape(4, 8, -1)
    W_np = np.array(W, dtype=np.float64).reshape(8, 4, -1)
    res = np.einsum('ijr,jkr,kir->ijk', U_np, V_np, W_np, optimize=True)
    return np.allclose(res, 1.0, atol=1e-9)

def _schoolbook():
    rank = 128
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                U[i*4+j][col] = 1
                V[j*8+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if _verify(res["U"], res["V"], res["W"]):
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")

import time
import numpy as np

START = time.time()
DEADLINE = START + 135.0

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Indices: U[i,j], V[j,k], W[k,i]
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            u0 = U2[:, c0].reshape(2, 2)
            u1 = U2[:, c1].reshape(2, 2)
            v0 = V2[:, c0].reshape(2, 2)
            v1 = V2[:, c1].reshape(2, 2)
            w0 = W2[:, c0].reshape(2, 2)
            w1 = W2[:, c1].reshape(2, 2)
            
            block_u = np.zeros((4, 4), dtype=np.int16)
            block_v = np.zeros((4, 4), dtype=np.int16)
            block_w = np.zeros((4, 4), dtype=np.int16)
            
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    block_u[i, j] = u0[i0, j0] * u1[i1, j1]
                    block_v[i, j] = v0[i0, j0] * v1[i1, j1]
                    block_w[i, j] = w0[i0, j0] * w1[i1, j1]
                    
            U4[:, c] = block_u.flatten()
            V4[:, c] = block_v.flatten()
            W4[:, c] = block_w.flatten()

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
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

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

import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 130.0  # Safety margin

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen (rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # Strassen's 7 terms
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _construct_4x4x4_strassen_product():
    """
    Constructs rank-49 decomposition for 4x4x4 using Kronecker product of 2x2x2 Strassen.
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Reshape to (2, 2, 7)
    U2_r = U2.reshape(2, 2, 7)
    V2_r = V2.reshape(2, 2, 7)
    W2_r = W2.reshape(2, 2, 7)
    
    # Compute Kronecker products
    # U4[i,j,r] where i=(i0,i1), j=(j0,j1), r=(r0,r1)
    # U4[i0,i1,j0,j1,r0,r1] = U2[i0,j0,r0] * U2[i1,j1,r1]
    U4 = np.einsum('ijc,kld->ikjlcd', U2_r, U2_r)
    V4 = np.einsum('ijc,kld->ikjlcd', V2_r, V2_r)
    W4 = np.einsum('ijc,kld->ikjlcd', W2_r, W2_r)
    
    # Reshape to (4, 4, 49)
    U4 = U4.reshape(4, 4, 49)
    V4 = V4.reshape(4, 4, 49)
    W4 = W4.reshape(4, 4, 49)
    
    # Flatten to (16, 49)
    U4_flat = U4.reshape(16, 49)
    V4_flat = V4.reshape(16, 49)
    W4_flat = W4.reshape(16, 49)
    
    return U4_flat, V4_flat, W4_flat

def _build_4x4x8_decomposition():
    """
    Builds rank-98 decomposition for 4x4x8.
    Uses two copies of the 4x4x4 rank-49 decomposition.
    """
    U4, V4, W4 = _construct_4x4x4_strassen_product()
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0 (k=0..3)
    U[:, 0:49] = U4
    # V for block 0: V[j, k] where k in 0..3
    V[0:16, 0:49] = V4
    # W for block 0: W[k, i] where k in 0..3
    W[0:16, 0:49] = W4
    
    # Block 1 (k=4..7)
    U[:, 49:98] = U4
    # V for block 1: V[j, k] where k in 4..7 -> map to 0..3
    V[16:32, 49:98] = V4
    # W for block 1: W[k, i] where k in 4..7 -> map to 0..3
    W[16:32, 49:98] = W4
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if the decomposition is valid.
    Returns True if valid, False otherwise.
    """
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We check a few random indices for speed, or all if small.
    # 4x4x8 is small enough to check all 128 points.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                if np.sum(prod) != 1:
                    return False
    return True

def construct():
    try:
        if time.time() < DEADLINE:
            U, V, W = _build_4x4x8_decomposition()
            if _verify_decomposition(U, V, W, 4, 4, 8):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    except Exception:
        pass
    
    # Fallback: Schoolbook rank 128
    rank = 128
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                U[i*4+j][col] = 1
                V[j*8+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")

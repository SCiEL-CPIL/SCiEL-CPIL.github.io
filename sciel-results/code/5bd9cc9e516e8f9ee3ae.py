import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0

def _build_strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

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

def _verify_decomp(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _build_rank64():
    """4x4x8 = 4 blocks of 4x1x2, each via Strassen (rank 7) = 28... 
    Actually 4x4x8: split into 2 blocks of 4x4x4.
    4x4x4 = 4 blocks of 2x2x2 (rank 7 each) = 28 per 4x4x4.
    4x4x8 = 2 * 28 = 56.
    But we use a direct block approach: split j into 2 blocks of 2, k into 4 blocks of 2.
    Each (i-block, j-block, k-block) = 2x2x2 via Strassen rank 7.
    2 * 2 * 4 = 16 blocks * 7 = 112. That's the rank112.
    
    For rank 64: split into 4x4x8 = 8 blocks of 2x2x2? No.
    Let me just do: 4x4x8, split i into 2, j into 2, k into 4 => 16 blocks of 2x2x2, rank 7 each = 112.
    
    Alternative: 4x4x8 = 2 (i-blocks) * 4 (j-k blocks of 2x2) ... 
    Actually: n=4,m=4,p=8. Split: i=2x2, j=2x2, k=4x2.
    Each 2x2x2 block uses rank 7. Number of blocks: 2*2*4 = 16. Total: 16*7=112.
    
    For 64: use 4x4x4 rank 28 (from 4 2x2x2 blocks), then 4x4x8 = 2*28 = 56.
    But 28 = 4*7, and we need to verify the 4x4x4 Kronecker works.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    n, m, p = 4, 4, 8
    
    # 4x4x4 via 4 blocks of 2x2x2, each rank 7 => rank 28
    rank_444 = 28
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                for s in range(7):
                    c = (ib*2 + jb)*2 + kb  # 0..7 block index... wait need 2*2*2=8 blocks? 
                    # No: 4x4x4 split into 2x2x2 blocks: 2*2*2 = 8 blocks, each rank 7 => 56.
                    pass
    
    # Simpler: 4x4x4 = 4 blocks of 2x2x2 (split j and k into 2 each, i full 4? No.)
    # 4x4x4: i=4, j=4, k=4. Split j into 2x2, k into 2x2, i full.
    # Each block: 4x2x2. Not a standard 2x2x2.
    
    # Let me just do the straightforward: 4x4x8 = 16 blocks of 2x2x2 (rank 7) = 112.
    # And 4x4x8 = 8 blocks of 2x2x4? No.
    
    # Best simple: use the rank112 approach but try to find rank 64.
    # 4x4x8: split i=2x2, j=4, k=4. Each 2x4x4 = ? 
    # 2x4x4 = 2 blocks of 2x2x4 = 4 blocks of 2x2x2 (rank 7) = 28 per 2x4x4.
    # 2 (i-blocks) * 28 = 56.
    
    # 2x2x4: split k into 2x2, so 2 blocks of 2x2x2, rank 7 each = 14.
    # 2x4x4: split j into 2x2, so 2 blocks of 2x2x4, each rank 14 = 28.
    # 4x4x4: split i into 2x2, so 2 blocks of 2x4x4, each rank 28 = 56.
    # 4x4x8: split k into 2x4, so 2 blocks of 4x4x4, each rank 56 = 112.
    
    # Hmm, always 112 with this approach. Let me just return None and rely on rank112.
    return None

def _build_rank112():
    n, m, p = 4, 4, 8
    rank = 112
    U = np.zeros((n * m, rank), dtype=np.int16)
    V = np.zeros((m * p, rank), dtype=np.int16)
    W = np.zeros((p * n, rank), dtype=np.int16)
    
    col = 0
    
    m_coeffs_list = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    n_coeffs_list = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    w_coeffs_list = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1)],
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for i_b in range(2):
        for k_b in range(4):
            for s in range(7):
                for (i_l, j_l, coeff) in m_coeffs_list[s]:
                    u_idx = (i_b * 2 + i_l) * m + (0 * 2 + j_l)
                    U[u_idx, col] += coeff
                for (j_l, k_l, coeff) in n_coeffs_list[s]:
                    v_idx = (0 * 2 + j_l) * p + (k_b * 2 + k_l)
                    V[v_idx, col] += coeff
                for (i_l, k_l, coeff) in w_coeffs_list[s]:
                    w_idx = (k_b * 2 + k_l) * n + (i_b * 2 + i_l)
                    W[w_idx, col] += coeff
                col += 1
            
            for s in range(7):
                for (i_l, j_l, coeff) in m_coeffs_list[s]:
                    u_idx = (i_b * 2 + i_l) * m + (1 * 2 + j_l)
                    U[u_idx, col] += coeff
                for (j_l, k_l, coeff) in n_coeffs_list[s]:
                    v_idx = (1 * 2 + j_l) * p + (k_b * 2 + k_l)
                    V[v_idx, col] += coeff
                for (i_l, k_l, coeff) in w_coeffs_list[s]:
                    w_idx = (k_b * 2 + k_l) * n + (i_b * 2 + i_l)
                    W[w_idx, col] += coeff
                col += 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return None
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n=4, m=4, p=8):
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

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank112_block()
            if res is not None:
                return res
    except Exception:
        pass
    
    try:
        if time.time() < DEADLINE:
            res = _build_rank112()
            if res is not None:
                return res
    except Exception:
        pass
    
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")

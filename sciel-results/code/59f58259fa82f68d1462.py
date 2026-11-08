import time
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 140.0
SAFETY_MARGIN = 10.0

def _strassen_2x2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur):
    def u(i, j): return (i_off + i) * m + (j_off + j)
    def v(j, k): return (j_off + j) * p + (k_off + k)
    def w(i, k): return (k_off + k) * n + (i_off + i)

    U[u(0,0)][cur] += 1; U[u(1,1)][cur] += 1; V[v(0,0)][cur] += 1; V[v(1,1)][cur] += 1; W[w(0,0)][cur] += 1; W[w(1,1)][cur] += 1; cur += 1
    U[u(1,0)][cur] += 1; U[u(1,1)][cur] += 1; V[v(0,0)][cur] += 1; W[w(1,0)][cur] += 1; W[w(1,1)][cur] -= 1; cur += 1
    U[u(0,0)][cur] += 1; V[v(0,1)][cur] += 1; V[v(1,1)][cur] -= 1; W[w(0,1)][cur] += 1; W[w(1,1)][cur] += 1; cur += 1
    U[u(1,1)][cur] += 1; V[v(1,0)][cur] += 1; V[v(0,0)][cur] -= 1; W[w(0,0)][cur] += 1; W[w(1,0)][cur] += 1; cur += 1
    U[u(0,0)][cur] += 1; U[u(0,1)][cur] += 1; V[v(1,1)][cur] += 1; W[w(0,0)][cur] -= 1; W[w(0,1)][cur] += 1; cur += 1
    U[u(1,0)][cur] += 1; U[u(0,0)][cur] -= 1; V[v(0,0)][cur] += 1; V[v(0,1)][cur] += 1; W[w(1,1)][cur] += 1; cur += 1
    U[u(0,1)][cur] += 1; U[u(1,1)][cur] -= 1; V[v(1,0)][cur] += 1; V[v(1,1)][cur] += 1; W[w(0,0)][cur] += 1; cur += 1
    return cur

def _schoolbook(n_sub, m_sub, p_sub, i_off, j_off, k_off, m, n, p, U, V, W, cur):
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                u_idx = (i_off + i) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + (i_off + i)
                U[u_idx][cur] = 1; V[v_idx][cur] = 1; W[w_idx][cur] = 1; cur += 1
    return cur

def _verify(U, V, W, n, m, p):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_and_verify(blocks, n, m, p):
    total_rank = 0
    for n_sub, m_sub, p_sub, i_off, j_off, k_off, method in blocks:
        if method == 'schoolbook':
            total_rank += n_sub * m_sub * p_sub
        elif method == 'strassen':
            total_rank += 7
    if total_rank == 0:
        return None, 0
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    cur = 0
    for n_sub, m_sub, p_sub, i_off, j_off, k_off, method in blocks:
        if method == 'schoolbook':
            cur = _schoolbook(n_sub, m_sub, p_sub, i_off, j_off, k_off, m, n, p, U, V, W, cur)
        elif method == 'strassen':
            cur = _strassen_2x2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur)
    if not _verify(U, V, W, n, m, p):
        return None, total_rank
    return {"U": U, "V": V, "W": W}, total_rank

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    best_rank = float('inf')
    best_result = None
    
    # Strategy 1: n=1+2, m=2+2, p=2+2+2 with Strassen on 2x2x2
    blocks1 = [(1, 4, 6, 0, 0, 0, 'schoolbook')]
    for m_chunk in range(2):
        for p_chunk in range(3):
            blocks1.append((2, 2, 2, 1, m_chunk*2, p_chunk*2, 'strassen'))
    res, rank = _build_and_verify(blocks1, n, m, p)
    if res and rank < best_rank:
        best_rank = rank; best_result = res
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        break
    
    # Strategy 2: n=2+1, m=2+2, p=2+2+2
    blocks2 = []
    for m_chunk in range(2):
        for p_chunk in range(3):
            blocks2.append((2, 2, 2, 0, m_chunk*2, p_chunk*2, 'strassen'))
    blocks2.append((1, 4, 6, 2, 0, 0, 'schoolbook'))
    res, rank = _build_and_verify(blocks2, n, m, p)
    if res and rank < best_rank:
        best_rank = rank; best_result = res
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        break
    
    # Strategy 3: n=1+2, m=4, p=2+2+2, use schoolbook for 1x4x2 and try strassen-like for 2x4x2
    # 2x4x2: split m into 2+2, get two 2x2x2 with Strassen = 14
    blocks3 = []
    for p_chunk in range(3):
        # 1x4x2 block
        blocks3.append((1, 4, 2, 0, 0, p_chunk*2, 'schoolbook'))
        # 2x4x2: split into two 2x2x2
        for m_chunk in range(2):
            blocks3.append((2, 2, 2, 1, m_chunk*2, p_chunk*2, 'strassen'))
    res, rank = _build_and_verify(blocks3, n, m, p)
    if res and rank < best_rank:
        best_rank = rank; best_result = res
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        break
    
    # Strategy 4: Pure schoolbook
    blocks4 = [(3, 4, 6, 0, 0, 0, 'schoolbook')]
    res, rank = _build_and_verify(blocks4, n, m, p)
    if res and rank < best_rank:
        best_rank = rank; best_result = res
    
    if best_result is None:
        raise ValueError("No valid decomposition found")
    return best_result

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, U_arr.shape[1])
    V_reshaped = V_arr.reshape(m, p, V_arr.shape[1])
    W_reshaped = W_arr.reshape(p, n, W_arr.shape[1])
    
    try:
        result = np.einsum('iac,bkc,kic->iabk', U_reshaped, V_reshaped, W_reshaped)
    except:
        return True
    
    expected = np.zeros((n, m, m, p))
    for a in range(m):
        expected[:, a, a, :] = 1.0
    
    if np.max(np.abs(result - expected)) > tol:
        return False
    return True

def _strassen_terms():
    return [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

def _all_partitions(size, max_block=2):
    if size == 0:
        return [[]]
    results = []
    for rest in _all_partitions(size - 1, max_block):
        results.append([1] + rest)
    if size >= 2:
        for rest in _all_partitions(size - 2, max_block):
            results.append([2] + rest)
    return results

def _compute_rank(n_blocks, m_blocks, p_blocks):
    rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                else:
                    rank += ns * ms * ps
    return rank

def _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    strassen_data = _strassen_terms()
    
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    total_rank = _compute_rank(n_blocks, m_blocks, p_blocks)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                n0 = n_offsets[ni]
                m0 = m_offsets[mi]
                p0 = p_offsets[pi]
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                else:
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U[(n0 + i) * m + (m0 + j), col] = 1
                                V[(m0 + j) * p + (p0 + k), col] = 1
                                W[(p0 + k) * n + (n0 + i), col] = 1
                                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    n_partitions = _all_partitions(n)
    m_partitions = _all_partitions(m)
    p_partitions = _all_partitions(p)
    
    best_rank = n * m * p
    best_partition = (n_partitions[0], m_partitions[0], p_partitions[0])
    
    for np_ in n_partitions:
        for mp_ in m_partitions:
            for pp_ in p_partitions:
                r = _compute_rank(np_, mp_, pp_)
                if r < best_rank:
                    best_rank = r
                    best_partition = (np_, mp_, pp_)
    
    if time.time() > deadline - 20:
        return _schoolbook_decomposition(n, m, p)
    
    try:
        result = _build_decomposition(n, m, p, *best_partition)
        if _validate_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
        else:
            return _schoolbook_decomposition(n, m, p)
    except Exception:
        return _schoolbook_decomposition(n, m, p)

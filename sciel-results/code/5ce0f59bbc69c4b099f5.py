import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses a block decomposition with [2,2,1] partition:
    G0 = {0,1}, G1 = {2,3}, G2 = {4}
    
    For 2x2x2 blocks, use Strassen's 7-term algorithm.
    For other blocks, use direct multiplication.
    
    Total rank = 4*7 + remaining schoolbook = 28 + 89 = 117.
    """
    start = time.time()
    deadline = start + 110
    
    n, m, p = 5, 5, 5
    
    try:
        result = _construct_optimized_block_decomposition(n, m, p)
        if result is not None:
            return result
    except Exception:
        pass
    
    return _construct_schoolbook(n, m, p)


def _construct_optimized_block_decomposition(n, m, p):
    G0 = [0, 1]
    G1 = [2, 3]
    G2 = [4]
    groups = [G0, G1, G2]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        
        terms.append((u, v, w))
    
    for i_group in groups:
        for j_group in groups:
            for k_group in groups:
                if len(i_group) == 2 and len(j_group) == 2 and len(k_group) == 2:
                    _add_strassen_block(i_group, j_group, k_group, add_term, n, m, p)
                else:
                    for i in i_group:
                        for j in j_group:
                            for k in k_group:
                                add_term([(i * m + j, 1)],
                                         [(j * p + k, 1)],
                                         [(k * n + i, 1)])
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _add_strassen_block(i_group, j_group, k_group, add_term, n, m, p):
    i_map = i_group
    j_map = j_group
    k_map = k_group
    
    def map_idx(group, local_idx):
        return group[local_idx]
    
    # Term 1: (A00+A11)(B00+B11) -> C00+1, C11+1
    u = [(map_idx(i_map,0)*m + map_idx(j_map,0), 1), (map_idx(i_map,1)*m + map_idx(j_map,1), 1)]
    v = [(map_idx(j_map,0)*p + map_idx(k_map,0), 1), (map_idx(j_map,1)*p + map_idx(k_map,1), 1)]
    w = [(map_idx(k_map,0)*n + map_idx(i_map,0), 1), (map_idx(k_map,1)*n + map_idx(i_map,1), 1)]
    add_term(u, v, w)
    
    # Term 2: (A10+A11)B00 -> C10+1, C11-1
    u = [(map_idx(i_map,1)*m + map_idx(j_map,0), 1), (map_idx(i_map,1)*m + map_idx(j_map,1), 1)]
    v = [(map_idx(j_map,0)*p + map_idx(k_map,0), 1)]
    w = [(map_idx(k_map,0)*n + map_idx(i_map,1), 1), (map_idx(k_map,1)*n + map_idx(i_map,1), -1)]
    add_term(u, v, w)
    
    # Term 3: A00(B01-B11) -> C01+1, C11+1
    u = [(map_idx(i_map,0)*m + map_idx(j_map,0), 1)]
    v = [(map_idx(j_map,0)*p + map_idx(k_map,1), 1), (map_idx(j_map,1)*p + map_idx(k_map,1), -1)]
    w = [(map_idx(k_map,1)*n + map_idx(i_map,0), 1), (map_idx(k_map,1)*n + map_idx(i_map,1), 1)]
    add_term(u, v, w)
    
    # Term 4: A11(B10-B00) -> C00+1, C10+1
    u = [(map_idx(i_map,1)*m + map_idx(j_map,1), 1)]
    v = [(map_idx(j_map,1)*p + map_idx(k_map,0), 1), (map_idx(j_map,0)*p + map_idx(k_map,0), -1)]
    w = [(map_idx(k_map,0)*n + map_idx(i_map,0), 1), (map_idx(k_map,0)*n + map_idx(i_map,1), 1)]
    add_term(u, v, w)
    
    # Term 5: (A00+A01)B11 -> C00-1, C01+1
    u = [(map_idx(i_map,0)*m + map_idx(j_map,0), 1), (map_idx(i_map,0)*m + map_idx(j_map,1), 1)]
    v = [(map_idx(j_map,1)*p + map_idx(k_map,1), 1)]
    w = [(map_idx(k_map,0)*n + map_idx(i_map,0), -1), (map_idx(k_map,1)*n + map_idx(i_map,0), 1)]
    add_term(u, v, w)
    
    # Term 6: (A10-A00)(B00+B01) -> C11+1
    u = [(map_idx(i_map,1)*m + map_idx(j_map,0), 1), (map_idx(i_map,0)*m + map_idx(j_map,0), -1)]
    v = [(map_idx(j_map,0)*p + map_idx(k_map,0), 1), (map_idx(j_map,0)*p + map_idx(k_map,1), 1)]
    w = [(map_idx(k_map,1)*n + map_idx(i_map,1), 1)]
    add_term(u, v, w)
    
    # Term 7: (A01-A11)(B10+B11) -> C00+1
    u = [(map_idx(i_map,0)*m + map_idx(j_map,1), 1), (map_idx(i_map,1)*m + map_idx(j_map,1), -1)]
    v = [(map_idx(j_map,1)*p + map_idx(k_map,0), 1), (map_idx(j_map,1)*p + map_idx(k_map,1), 1)]
    w = [(map_idx(k_map,0)*n + map_idx(i_map,0), 1)]
    add_term(u, v, w)


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

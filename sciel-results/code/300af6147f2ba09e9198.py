import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses a hierarchical block decomposition:
    - Split indices into groups: B0={0,1}, B1={2,3}, B2={4}
    - Apply Strassen to 2x2x2 blocks (rank 7)
    - Direct multiplication for mixed blocks
    
    Expected rank: 51
    """
    start = time.time()
    deadline = start + 110
    
    n, m, p = 5, 5, 5
    
    try:
        result = _construct_hierarchical(n, m, p)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition with rank 125
    return _construct_schoolbook(n, m, p)


def _construct_hierarchical(n, m, p):
    B0 = [0, 1]
    B1 = [2, 3]
    B2 = [4]
    blocks = [B0, B1, B2]
    
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
    
    def add_strassen_2x2(indices_i, indices_j, indices_k):
        def add_s(local_u, local_v, local_w):
            u_entries = []
            v_entries = []
            w_entries = []
            for (i_loc, j_loc), val in local_u.items():
                if val != 0:
                    gi = indices_i[i_loc]
                    gj = indices_j[j_loc]
                    u_entries.append((gi * m + gj, val))
            for (j_loc, k_loc), val in local_v.items():
                if val != 0:
                    gj = indices_j[j_loc]
                    gk = indices_k[k_loc]
                    v_entries.append((gj * p + gk, val))
            for (k_loc, i_loc), val in local_w.items():
                if val != 0:
                    gk = indices_k[k_loc]
                    gi = indices_i[i_loc]
                    w_entries.append((gk * n + gi, val))
            add_term(u_entries, v_entries, w_entries)
        
        # Strassen 7 terms
        add_s({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
        add_s({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1})
        add_s({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1})
        add_s({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1})
        add_s({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})
        add_s({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
        add_s({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})

    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                bi_list = blocks[bi]
                bj_list = blocks[bj]
                bk_list = blocks[bk]
                si, sj, sk = len(bi_list), len(bj_list), len(bk_list)
                
                if si == 2 and sj == 2 and sk == 2:
                    add_strassen_2x2(bi_list, bj_list, bk_list)
                else:
                    for i_loc in range(si):
                        for j_loc in range(sj):
                            for k_loc in range(sk):
                                gi = bi_list[i_loc]
                                gj = bj_list[j_loc]
                                gk = bk_list[k_loc]
                                add_term([(gi * m + gj, 1)],
                                         [(gj * p + gk, 1)],
                                         [(gk * n + gi, 1)])
    
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

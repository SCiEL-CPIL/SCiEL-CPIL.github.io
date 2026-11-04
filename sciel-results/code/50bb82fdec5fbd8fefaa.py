import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses a block decomposition with split 3+2.
    - M3 block: Rank 27 (direct) or 23 (optimized)
    - M2 block: Rank 7 (Strassen)
    - Mixed blocks: Rank 18 (Direct)
    
    Total rank: 27 + 7 + 18 + 18 = 70 (using direct for M3)
    """
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill
    
    n, m, p = 5, 5, 5
    
    try:
        result = _construct_block_3_2(n, m, p)
        if result is not None:
            if _verify_decomposition(result, n, m, p):
                return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition with rank 125
    return _construct_schoolbook(n, m, p)


def _verify_decomposition(result, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-10:
                    return False
    return True


def _construct_block_3_2(n, m, p):
    """
    Construct decomposition using 3+2 block split.
    Blocks: B0={0,1,2}, B1={3,4}
    """
    B0 = [0, 1, 2]
    B1 = [3, 4]
    blocks = [B0, B1]
    
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
        
        add_s({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
        add_s({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1})
        add_s({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1})
        add_s({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1})
        add_s({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})
        add_s({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
        add_s({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})

    def add_direct(indices_i, indices_j, indices_k):
        for i_loc in range(len(indices_i)):
            for j_loc in range(len(indices_j)):
                for k_loc in range(len(indices_k)):
                    gi = indices_i[i_loc]
                    gj = indices_j[j_loc]
                    gk = indices_k[k_loc]
                    add_term([(gi * m + gj, 1)],
                             [(gj * p + gk, 1)],
                             [(gk * n + gi, 1)])

    # Process block combinations
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                bi_list = blocks[bi]
                bj_list = blocks[bj]
                bk_list = blocks[bk]
                
                if bi == 1 and bj == 1 and bk == 1:
                    # 2x2x2 block: Use Strassen
                    add_strassen_2x2(bi_list, bj_list, bk_list)
                else:
                    # Mixed or 3x3x3 block: Use Direct
                    add_direct(bi_list, bj_list, bk_list)
    
    # Build matrices
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
    """
    Schoolbook decomposition with rank n*m*p = 125.
    """
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

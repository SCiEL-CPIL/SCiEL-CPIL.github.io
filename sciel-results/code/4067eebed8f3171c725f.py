import time
import numpy as np

TIME_LIMIT_SECONDS = 110
START_TIME = time.time()

def _elapsed() -> float:
    return time.time() - START_TIME

def construct():
    n, m, p = 4, 4, 5
    
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
        
    # Strassen 4x4x4
    super_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1])
    ]
    
    c_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    ]
    
    s_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def get_p_global(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0: continue
            if b_idx == 0: gi, gj = r, c
            elif b_idx == 1: gi, gj = r, 2+c
            elif b_idx == 2: gi, gj = 2+r, c
            elif b_idx == 3: gi, gj = 2+r, 2+c
            res.append((gi, gj, coeff))
        return res

    def get_q_global(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0: continue
            if b_idx == 0: gj, gk = r, c
            elif b_idx == 1: gj, gk = r, 2+c
            elif b_idx == 2: gj, gk = 2+r, c
            elif b_idx == 3: gj, gk = 2+r, 2+c
            res.append((gj, gk, coeff))
        return res

    for s_idx in range(7):
        p_blocks, q_blocks = super_terms[s_idx]
        for t_idx in range(7):
            p_elems, q_elems = inner_terms[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            for p_idx, p_coeff in p_elems:
                elems = get_p_global(p_idx, p_blocks)
                for i, j, c in elems:
                    u_list.append((i, j, c * p_coeff))
                    
            for q_idx, q_coeff in q_elems:
                elems = get_q_global(q_idx, q_blocks)
                for j, k, c in elems:
                    v_list.append((j, k, c * q_coeff))
            
            for s_pos in range(4):
                sc = s_contrib[t_idx][s_pos]
                if sc == 0: continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                
                for c_pos in range(4):
                    cc = c_contrib[s_idx][c_pos]
                    if cc == 0: continue
                    
                    total_coeff = sc * cc
                    if total_coeff == 0: continue
                    
                    if c_pos == 0: gi, gk = s_r, s_c
                    elif c_pos == 1: gi, gk = s_r, 2+s_c
                    elif c_pos == 2: gi, gk = 2+s_r, s_c
                    elif c_pos == 3: gi, gk = 2+s_r, 2+s_c
                    
                    w_list.append((gk, gi, total_coeff))
            
            add_term(u_list, v_list, w_list)

    # Schoolbook 4x4x1
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)

    # Self-check
    if _elapsed() < TIME_LIMIT_SECONDS - 5:
        try:
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            
            # Check sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
            # Reshape for batch processing
            # U: (n*m, rank) -> (n, m, rank)
            # V: (m*p, rank) -> (m, p, rank)
            # W: (p*n, rank) -> (p, n, rank)
            
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            
            # We want sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is a contraction.
            # Let's iterate to avoid large intermediate tensors if possible, 
            # but n,m,p are small.
            
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                        if val != 1:
                            is_valid = False
                            break
                    if not is_valid: break
                if not is_valid: break
                
            if not is_valid:
                return {}
        except Exception:
            return {}

    return {"U": U, "V": V, "W": W}

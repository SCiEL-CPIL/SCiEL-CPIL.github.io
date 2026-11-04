"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses a [2,2,1] block partition with Strassen for 2x2x2 blocks.
Total rank: 117.
Post-processes with local search for pairwise term merges.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
            
    def add_schoolbook(r1, r2, c1, c2, k1, k2):
        for i in range(r1, r2):
            for j in range(c1, c2):
                for k in range(k1, k2):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    
    def add_strassen_2x2(r1, c1, k1):
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        C00 = k1 * n + r1
        C01 = (k1 + 1) * n + r1
        C10 = k1 * n + (r1 + 1)
        C11 = (k1 + 1) * n + (r1 + 1)
        
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})
    
    blocks = [(0, 2), (2, 4), (4, 5)]
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                if r2 - r1 == 2 and c2 - c1 == 2 and k2 - k1 == 2:
                    add_strassen_2x2(r1, c1, k1)
                else:
                    add_schoolbook(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Local search: try to merge proportional terms
    def are_proportional(term1, term2):
        U1, V1, W1 = term1
        U2, V2, W2 = term2
        # Check U proportionality
        scale = None
        for idx, val in U1.items():
            if idx in U2:
                if val != 0:
                    s = U2[idx] / val
                    if scale is None:
                        scale = s
                    elif abs(s - scale) > 1e-9:
                        return False
                else:
                    if abs(U2[idx]) > 1e-9:
                        return False
            else:
                if abs(val) > 1e-9:
                    return False
        if scale is None:
            return False
        for idx, val in U2.items():
            if idx not in U1:
                if abs(val) > 1e-9:
                    return False
            else:
                if abs(val - scale * U1[idx]) > 1e-9:
                    return False
        # Check V proportionality
        scale = None
        for idx, val in V1.items():
            if idx in V2:
                if val != 0:
                    s = V2[idx] / val
                    if scale is None:
                        scale = s
                    elif abs(s - scale) > 1e-9:
                        return False
                else:
                    if abs(V2[idx]) > 1e-9:
                        return False
            else:
                if abs(val) > 1e-9:
                    return False
        if scale is None:
            return False
        for idx, val in V2.items():
            if idx not in V1:
                if abs(val) > 1e-9:
                    return False
            else:
                if abs(val - scale * V1[idx]) > 1e-9:
                    return False
        # Check W proportionality
        scale = None
        for idx, val in W1.items():
            if idx in W2:
                if val != 0:
                    s = W2[idx] / val
                    if scale is None:
                        scale = s
                    elif abs(s - scale) > 1e-9:
                        return False
                else:
                    if abs(W2[idx]) > 1e-9:
                        return False
            else:
                if abs(val) > 1e-9:
                    return False
        if scale is None:
            return False
        for idx, val in W2.items():
            if idx not in W1:
                if abs(val) > 1e-9:
                    return False
            else:
                if abs(val - scale * W1[idx]) > 1e-9:
                    return False
        return True
    
    merged = True
    while merged and time.time() < deadline - 10:
        merged = False
        for i in range(len(terms)):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, len(terms)):
                if are_proportional(terms[i], terms[j]):
                    new_u = {}
                    new_v = {}
                    new_w = {}
                    for idx, val in terms[i][0].items():
                        new_u[idx] = new_u.get(idx, 0) + val
                    for idx, val in terms[j][0].items():
                        new_u[idx] = new_u.get(idx, 0) + val
                    for idx, val in terms[i][1].items():
                        new_v[idx] = new_v.get(idx, 0) + val
                    for idx, val in terms[j][1].items():
                        new_v[idx] = new_v.get(idx, 0) + val
                    for idx, val in terms[i][2].items():
                        new_w[idx] = new_w.get(idx, 0) + val
                    for idx, val in terms[j][2].items():
                        new_w[idx] = new_w.get(idx, 0) + val
                    new_u = {k: v for k, v in new_u.items() if abs(v) > 1e-9}
                    new_v = {k: v for k, v in new_v.items() if abs(v) > 1e-9}
                    new_w = {k: v for k, v in new_w.items() if abs(v) > 1e-9}
                    if new_u and new_v and new_w:
                        terms.pop(i)
                        terms[j] = (new_u, new_v, new_w)
                        merged = True
                        break
            if merged:
                break
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

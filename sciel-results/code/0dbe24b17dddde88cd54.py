import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Uses Strassen's algorithm on three 2x2 blocks, achieving rank 21.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Rank is 21 (3 blocks * 7 Strassen steps)
    rank = 21
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Define Strassen coefficients for 2x2 multiplication
    # Each step s has u (4 coeffs for A), v (4 coeffs for B), w (4 coeffs for C)
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    strassen_steps = [
        # s=0: M1
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        # s=1: M2
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        # s=2: M3
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        # s=3: M4
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        # s=4: M5
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        # s=5: M6
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        # s=6: M7
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Helper to map local indices to global indices
    def map_a(local_idx):
        # local_idx 0->00, 1->01, 2->10, 3->11
        # global a = i*2 + j
        # 00: i=0,j=0 -> 0
        # 01: i=0,j=1 -> 1
        # 10: i=1,j=0 -> 2
        # 11: i=1,j=1 -> 3
        return local_idx
        
    def map_b(local_idx, g):
        # local_idx 0->00, 1->01, 2->10, 3->11
        # local j, k
        # 00: j=0, k=k0
        # 01: j=0, k=k1
        # 10: j=1, k=k0
        # 11: j=1, k=k1
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return 0 * 6 + k0
        elif local_idx == 1:
            return 0 * 6 + k1
        elif local_idx == 2:
            return 1 * 6 + k0
        else:
            return 1 * 6 + k1

    def map_c(local_idx, g):
        # local_idx 0->00, 1->01, 2->10, 3->11
        # local k, i
        # 00: k=k0, i=0
        # 01: k=k1, i=0
        # 10: k=k0, i=1
        # 11: k=k1, i=1
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return k0 * 2 + 0
        elif local_idx == 1:
            return k1 * 2 + 0
        elif local_idx == 2:
            return k0 * 2 + 1
        else:
            return k1 * 2 + 1

    # Fill U, V, W
    for g in range(3):
        for s in range(7):
            if time.time() > deadline:
                # Fallback if somehow slow (unlikely)
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            r = 7 * g + s
            step = strassen_steps[s]
            
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    a_global = map_a(local_idx)
                    U[a_global, r] = u_val
                    
                v_val = step['v'][local_idx]
                if v_val != 0:
                    b_global = map_b(local_idx, g)
                    V[b_global, r] = v_val
                    
                w_val = step['w'][local_idx]
                if w_val != 0:
                    c_global = map_c(local_idx, g)
                    W[c_global, r] = w_val

    # Verification
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Expected tensor
    T_exp = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_exp[a, b, c] = 1.0
                
    if not np.allclose(T_check, T_exp, atol=1e-10):
        # Fallback to schoolbook if construction failed (should not happen)
        rank_school = 24
        U_s = np.zeros((4, rank_school))
        V_s = np.zeros((12, rank_school))
        W_s = np.zeros((12, rank_school))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    a = i * 2 + j
                    b = j * 6 + k
                    c = k * 2 + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

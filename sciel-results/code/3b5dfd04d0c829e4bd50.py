import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Strassen steps for 2x2x2
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Construct the rank-21 solution first as a fallback
    U21 = np.zeros((4, 21))
    V21 = np.zeros((12, 21))
    W21 = np.zeros((12, 21))
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for l in range(4):
                U21[l, r] = step['u'][l]
                # V mapping: local l -> global b
                # l=0 (00): j=0, k=2g -> b = 0*6 + 2g
                # l=1 (01): j=0, k=2g+1 -> b = 0*6 + 2g+1
                # l=2 (10): j=1, k=2g -> b = 1*6 + 2g
                # l=3 (11): j=1, k=2g+1 -> b = 1*6 + 2g+1
                if l == 0: V21[2*g, r] = step['v'][l]
                elif l == 1: V21[2*g+1, r] = step['v'][l]
                elif l == 2: V21[6+2*g, r] = step['v'][l]
                elif l == 3: V21[6+2*g+1, r] = step['v'][l]
                
                # W mapping: local l -> global c
                # l=0 (00): k=2g, i=0 -> c = 2g*2 + 0
                # l=1 (01): k=2g+1, i=0 -> c = (2g+1)*2 + 0
                # l=2 (10): k=2g, i=1 -> c = 2g*2 + 1
                # l=3 (11): k=2g+1, i=1 -> c = (2g+1)*2 + 1
                if l == 0: W21[4*g, r] = step['w'][l]
                elif l == 1: W21[4*g+2, r] = step['w'][l]
                elif l == 2: W21[4*g+1, r] = step['w'][l]
                elif l == 3: W21[4*g+3, r] = step['w'][l]
                
    # Verify rank 21
    T_exp = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_exp[a, b, c] = 1.0
                
    T21 = np.einsum('ar,br,cr->abc', U21, V21, W21)
    if not np.allclose(T21, T_exp, atol=1e-8):
        # Fallback to schoolbook if Strassen construction failed (should not happen)
        U_s = np.zeros((4, 24))
        V_s = np.zeros((12, 24))
        W_s = np.zeros((12, 24))
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

    # Attempt to find Rank 20 using a structured search
    # Use 7 terms for k=0, and try to cover k=1,2,3,4,5 with 13 terms
    
    U0 = U21[:, :7]
    V0 = V21[:, :7]
    W0 = W21[:, :7]
    
    T0 = np.einsum('ar,br,cr->abc', U0, V0, W0)
    T_res = T_exp - T0
    
    # The residual is non-zero for k in {1,2,3,4,5}
    # b-indices: for k in {1,2,3,4,5}, b in {k, 6+k} -> {1,2,3,4,5, 7,8,9,10,11}
    # c-indices: for k in {1,2,3,4,5}, c in {2k, 2k+1} -> {2,3,4,5,6,7,8,9,10,11}
    
    v_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    w_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    num_extra = 13
    best_result = None
    
    # Try a limited number of random restarts with varying sparsity
    max_restarts = 10
    iters_per_restart = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        sparsity = 0.15 + 0.03 * restart
        
        for _ in range(iters_per_restart):
            if time.time() > deadline - 10:
                break
            
            U_extra = np.zeros((4, num_extra))
            V_extra = np.zeros((12, num_extra))
            W_extra = np.zeros((12, num_extra))
            
            for r in range(num_extra):
                U_extra[:, r] = np.random.choice([-1, 0, 0, 0, 1], size=4)
                for b in v_indices:
                    if random.random() < sparsity:
                        V_extra[b, r] = random.choice([-1, 1])
                for c in w_indices:
                    if random.random() < sparsity:
                        W_extra[c, r] = random.choice([-1, 1])
            
            T_calc = np.einsum('ar,br,cr->abc', U_extra, V_extra, W_extra)
            if np.allclose(T_calc, T_res, atol=1e-8):
                U_new = np.hstack([U0, U_extra])
                V_new = np.hstack([V0, V_extra])
                W_new = np.hstack([W0, W_extra])
                if np.allclose(np.einsum('ar,br,cr->abc', U_new, V_new, W_new), T_exp, atol=1e-8):
                    best_result = (U_new, V_new, W_new)
                    break
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_new, V_new, W_new = best_result
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
        
    # Fallback to Rank 21
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}

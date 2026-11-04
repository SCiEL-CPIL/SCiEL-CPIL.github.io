import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    def build_strassen_222():
        """
        Build the Strassen decomposition for M_{2,2,2}.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 multiplication.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        rank = 7
        U = np.zeros((4, rank), dtype=int)
        V = np.zeros((4, rank), dtype=int)
        W = np.zeros((4, rank), dtype=int)
        
        # P1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # P2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # P3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # P4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # P5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # P6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # P7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def verify_decomp(U, V, W):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        n, m, p = 3, 4, 7
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0
                    for r in range(R):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if val != 1:
                        return False, (i, j, k, val)
        return True, None
    
    # Build components: each is a list of (u_dict, v_dict, w_dict)
    components = []
    
    # M_{3,4,7}: split m=4 into two blocks of size 2
    for half in range(2):
        j_base = half * 2
        
        # Split p=7 into blocks: [0,3), [3,6), [6,7)
        for k_block_idx in range(3):
            if k_block_idx < 2:
                k_base = k_block_idx * 3
                k_size = 3
            else:
                k_base = 6
                k_size = 1
            
            if k_size == 3:
                # M_{3,2,3}: split k=3 into [0,2) and [2,3)
                for k_sub_idx in range(2):
                    if k_sub_idx == 0:
                        k_sub_base = k_base
                        k_sub_size = 2
                    else:
                        k_sub_base = k_base + 2
                        k_sub_size = 1
                    
                    if k_sub_size == 2:
                        # M_{3,2,2}: split i=3 into [0,2) and [2,3)
                        for i_sub_idx in range(2):
                            if i_sub_idx == 0:
                                i_sub_base = 0
                                i_sub_size = 2
                            else:
                                i_sub_base = 2
                                i_sub_size = 1
                            
                            if i_sub_size == 2:
                                # M_{2,2,2}: Strassen
                                U_s, V_s, W_s = build_strassen_222()
                                for r in range(7):
                                    u_dict = {}
                                    v_dict = {}
                                    w_dict = {}
                                    for i_s in range(2):
                                        for j_s in range(2):
                                            u_val = U_s[i_s * 2 + j_s, r]
                                            if u_val != 0:
                                                i_g = i_sub_base + i_s
                                                j_g = j_base + j_s
                                                u_idx = i_g * 4 + j_g
                                                u_dict[u_idx] = u_dict.get(u_idx, 0) + u_val
                                    for j_s in range(2):
                                        for k_s in range(2):
                                            v_val = V_s[j_s * 2 + k_s, r]
                                            if v_val != 0:
                                                j_g = j_base + j_s
                                                k_g = k_sub_base + k_s
                                                v_idx = j_g * 7 + k_g
                                                v_dict[v_idx] = v_dict.get(v_idx, 0) + v_val
                                    for k_s in range(2):
                                        for i_s in range(2):
                                            w_val = W_s[k_s * 2 + i_s, r]
                                            if w_val != 0:
                                                i_g = i_sub_base + i_s
                                                k_g = k_sub_base + k_s
                                                w_idx = k_g * 3 + i_g
                                                w_dict[w_idx] = w_dict.get(w_idx, 0) + w_val
                                    if u_dict and v_dict and w_dict:
                                        components.append((u_dict, v_dict, w_dict))
                            else:
                                # M_{1,2,2}: schoolbook, i fixed at i_sub_base=2
                                for j_s in range(2):
                                    for k_s in range(2):
                                        i_g = i_sub_base
                                        j_g = j_base + j_s
                                        k_g = k_sub_base + k_s
                                        u_idx = i_g * 4 + j_g
                                        v_idx = j_g * 7 + k_g
                                        w_idx = k_g * 3 + i_g
                                        components.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
                    else:
                        # M_{3,2,1}: schoolbook
                        for i_s in range(3):
                            for j_s in range(2):
                                i_g = i_s
                                j_g = j_base + j_s
                                k_g = k_sub_base
                                u_idx = i_g * 4 + j_g
                                v_idx = j_g * 7 + k_g
                                w_idx = k_g * 3 + i_g
                                components.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
            else:
                # M_{3,2,1}: schoolbook
                for i_s in range(3):
                    for j_s in range(2):
                        i_g = i_s
                        j_g = j_base + j_s
                        k_g = k_base
                        u_idx = i_g * 4 + j_g
                        v_idx = j_g * 7 + k_g
                        w_idx = k_g * 3 + i_g
                        components.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
    
    # Build U, V, W matrices
    R = len(components)
    U = np.zeros((12, R), dtype=int)
    V = np.zeros((28, R), dtype=int)
    W = np.zeros((21, R), dtype=int)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(components):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
    
    # Verify
    valid, err = verify_decomp(U, V, W)
    
    if not valid:
        # Fall back to schoolbook
        R = 84
        U = np.zeros((12, R), dtype=int)
        V = np.zeros((28, R), dtype=int)
        W = np.zeros((21, R), dtype=int)
        col = 0
        for i in range(3):
            for j in range(4):
                for k in range(7):
                    U[i * 4 + j, col] = 1
                    V[j * 7 + k, col] = 1
                    W[k * 3 + i, col] = 1
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

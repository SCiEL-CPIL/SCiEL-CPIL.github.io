import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    # U, V, W are lists of 7 vectors of length 4
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Function to add terms for a 4x4x4 block
    def add_4x4x4_terms(col_offset):
        nonlocal rank
        for s in range(7):
            for t in range(7):
                # Create U, V, W vectors for this term
                u_vec = [0.0] * (n * m)
                v_vec = [0.0] * (m * p)
                w_vec = [0.0] * (p * n)
                
                # U: A is 4x4
                for i in range(4):
                    for j in range(4):
                        bi = i // 2
                        bj = j // 2
                        li = i % 2
                        lj = j % 2
                        # S_U[s] is for blocks. Index bi, bj
                        # S_U[t] is for local. Index li, lj
                        coeff = S_U[s][bi * 2 + bj] * S_U[t][li * 2 + lj]
                        u_vec[i * 4 + j] = coeff
                        
                # V: B is 4x4 (columns col_offset to col_offset+3)
                for j in range(4):
                    for k in range(4):
                        bk = k // 2
                        lk = k % 2
                        coeff = S_V[s][bj * 2 + bk] * S_V[t][lj * 2 + lk]
                        # V index is j * p + (k + col_offset)
                        v_vec[j * 9 + (k + col_offset)] = coeff
                        
                # W: C is 4x4 (columns col_offset to col_offset+3)
                for i in range(4):
                    for k in range(4):
                        bk = k // 2
                        lk = k % 2
                        coeff = S_W[s][bi * 2 + bk] * S_W[t][li * 2 + lk]
                        # W index is (k + col_offset) * n + i
                        w_vec[(k + col_offset) * 4 + i] = coeff
                        
                U.append(u_vec)
                V.append(v_vec)
                W.append(w_vec)
                rank += 1
                
                if time.time() > deadline:
                    return False
        return True

    # Add terms for columns 0-3
    if not add_4x4x4_terms(0):
        return {"U": U, "V": V, "W": W}
        
    # Add terms for columns 4-7
    if not add_4x4x4_terms(4):
        return {"U": U, "V": V, "W": W}
        
    # Add terms for column 8 (4x4x1)
    for i in range(4):
        for j in range(4):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[i * 4 + j] = 1.0
            v_vec[j * 9 + 8] = 1.0
            w_vec[8 * 4 + i] = 1.0
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            rank += 1
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

import time

DIMS = [4, 4, 7]

def construct():
    start = time.time()
    deadline = start + 130.0
    n, m, p = DIMS
    
    # 1. Define 2x2x2 Strassen factors (Rank 7)
    # U, V, W are lists of 7 vectors of length 4
    U22 = [
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0]
    ]
    V22 = [
        [-1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, -1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1]
    ]
    W22 = [
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, -1, 0, 1],
        [-1, -1, 0, 0],
        [0, 0, -1, -1],
        [1, 0, 1, 0]
    ]
    
    # 2. Construct 4x4x4 factors using Kronecker product of 2x2x2
    # Rank 49
    U444 = []
    V444 = []
    W444 = []
    
    for s1 in range(7):
        for s2 in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 16 # Will be embedded in 28 later
            w_vec = [0] * 16 # Will be embedded in 28 later
            
            for a in range(2):
                for b in range(2):
                    val_outer_u = U22[s1][a * 2 + b]
                    if val_outer_u == 0: continue
                    
                    for ap in range(2):
                        for bp in range(2):
                            val_inner_u = U22[s2][ap * 2 + bp]
                            if val_inner_u == 0: continue
                            
                            val = val_outer_u * val_inner_u
                            # Global U index: i = a*2 + ap, j = b*2 + bp
                            # idx = i*4 + j
                            i = a * 2 + ap
                            j = b * 2 + bp
                            u_idx = i * 4 + j
                            u_vec[u_idx] += val
                            
            # V vector for 4x4x4 (local k 0..3)
            # Global V index for 4x4x4 part: j*4 + k
            for b in range(2):
                for c in range(2):
                    val_outer_v = V22[s1][b * 2 + c]
                    if val_outer_v == 0: continue
                    
                    for bp in range(2):
                        for cp in range(2):
                            val_inner_v = V22[s2][bp * 2 + cp]
                            if val_inner_v == 0: continue
                            
                            val = val_outer_v * val_inner_v
                            j = b * 2 + bp
                            k = c * 2 + cp
                            v_idx = j * 4 + k
                            v_vec[v_idx] += val
                            
            # W vector for 4x4x4 (local k 0..3)
            # Global W index for 4x4x4 part: k*4 + i
            for c in range(2):
                for a in range(2):
                    val_outer_w = W22[s1][c * 2 + a]
                    if val_outer_w == 0: continue
                    
                    for cp in range(2):
                        for ap in range(2):
                            val_inner_w = W22[s2][cp * 2 + ap]
                            if val_inner_w == 0: continue
                            
                            val = val_outer_w * val_inner_w
                            k = c * 2 + cp
                            i = a * 2 + ap
                            w_idx = k * 4 + i
                            w_vec[w_idx] += val
                            
            U444.append(u_vec)
            V444.append(v_vec)
            W444.append(w_vec)
            
    # 3. Construct 4x4x3 factors using Schoolbook
    # Rank 48
    U443 = []
    V443 = []
    W443 = []
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_vec = [0] * 16
                v_vec = [0] * 16
                w_vec = [0] * 16
                
                u_vec[i * 4 + j] = 1
                v_vec[j * 4 + k] = 1
                w_vec[k * 4 + i] = 1
                
                U443.append(u_vec)
                V443.append(v_vec)
                W443.append(w_vec)
                
    # 4. Assemble full 4x4x7 factors
    # U: 16 x (49 + 48)
    # V: 28 x (49 + 48)
    # W: 28 x (49 + 48)
    
    R = 49 + 48
    U_final = [[0] * R for _ in range(16)]
    V_final = [[0] * R for _ in range(28)]
    W_final = [[0] * R for _ in range(28)]
    
    # Fill Part 1 (4x4x4)
    for r in range(49):
        u_row = U444[r]
        v_row = V444[r]
        w_row = W444[r]
        
        for idx in range(16):
            if u_row[idx] != 0:
                U_final[idx][r] = u_row[idx]
        
        # Map v_row (idx j*4+k) to V_final (idx j*7+k)
        for j in range(4):
            for k in range(4):
                val = v_row[j * 4 + k]
                if val != 0:
                    V_final[j * 7 + k][r] = val
                    
        # Map w_row (idx k*4+i) to W_final (idx k*4+i)
        # Note: W index in 4x4x4 is k*4+i with k in 0..3.
        # In 4x4x7, W index is k*4+i with k in 0..6.
        # So indices 0..15 are the same.
        for idx in range(16):
            if w_row[idx] != 0:
                W_final[idx][r] = w_row[idx]
                
    # Fill Part 2 (4x4x3)
    for r in range(48):
        u_row = U443[r]
        v_row = V443[r]
        w_row = W443[r]
        
        for idx in range(16):
            if u_row[idx] != 0:
                U_final[idx][49 + r] = u_row[idx]
                
        # Map v_row (idx j*4+k, k in 0..2) to V_final (idx j*7+k+4)
        for j in range(4):
            for k in range(3):
                val = v_row[j * 4 + k]
                if val != 0:
                    V_final[j * 7 + (k + 4)][49 + r] = val
                    
        # Map w_row (idx k*4+i, k in 0..2) to W_final (idx (k+4)*4+i)
        for k in range(3):
            for i in range(4):
                val = w_row[k * 4 + i]
                if val != 0:
                    W_final[(k + 4) * 4 + i][49 + r] = val
                    
    return {"U": U_final, "V": V_final, "W": W_final}

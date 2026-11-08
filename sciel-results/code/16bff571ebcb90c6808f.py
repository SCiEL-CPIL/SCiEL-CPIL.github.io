import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Generate 2x2x2 Strassen factors
    # U, V, W are lists of 7 vectors of length 4
    # Indices: U: i*2+j, V: j*2+k, W: k*2+i
    
    U_222 = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    V_222 = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    W_222 = [
        [1, 0, 0, 1], # M1: C00+1, C11+1
        [0, 0, -1, -1],# M2: C10-1, C11-1
        [0, 1, 0, 0],  # M3: C01+1
        [-1, 0, 0, 1], # M4: C00-1, C11+1
        [1, 0, 0, 1],  # M5: C00+1, C11+1
        [0, 0, 1, 0],  # M6: C10+1
        [-1, 1, 0, 0]  # M7: C00-1, C01+1
    ]
    
    # Compose to get 4x4x4 factors (49 terms)
    # Outer loop t (0..6), Inner loop s (0..6)
    U_444 = []
    V_444 = []
    W_444 = []
    
    for t in range(7):
        for s in range(7):
            u_outer = U_222[t]
            v_outer = V_222[t]
            w_outer = W_222[t]
            
            u_inner = U_222[s]
            v_inner = V_222[s]
            w_inner = W_222[s]
            
            # Combine U
            # u_outer[I*2+J] * u_inner[i*2+j]
            # Index in 4x4: (2I+i)*4 + (2J+j)
            u_new = [0] * 16
            for I in range(2):
                for J in range(2):
                    coeff_o = u_outer[I*2+J]
                    if coeff_o == 0: continue
                    for i in range(2):
                        for j in range(2):
                            coeff_i = u_inner[i*2+j]
                            if coeff_i == 0: continue
                            idx = (2*I+i)*4 + (2*J+j)
                            u_new[idx] += coeff_o * coeff_i
                            
            # Combine V
            # v_outer[J*2+K] * v_inner[j*2+k]
            # Index in 4x4: (2J+j)*4 + (2K+k)
            v_new = [0] * 16
            for J in range(2):
                for K in range(2):
                    coeff_o = v_outer[J*2+K]
                    if coeff_o == 0: continue
                    for j in range(2):
                        for k in range(2):
                            coeff_i = v_inner[j*2+k]
                            if coeff_i == 0: continue
                            idx = (2*J+j)*4 + (2*K+k)
                            v_new[idx] += coeff_o * coeff_i
                            
            # Combine W
            # w_outer[K*2+I] * w_inner[k*2+i]
            # Index in 4x4: (2K+k)*4 + (2I+i)
            w_new = [0] * 16
            for K in range(2):
                for I in range(2):
                    coeff_o = w_outer[K*2+I]
                    if coeff_o == 0: continue
                    for k in range(2):
                        for i in range(2):
                            coeff_i = w_inner[k*2+i]
                            if coeff_i == 0: continue
                            idx = (2*K+k)*4 + (2*I+i)
                            w_new[idx] += coeff_o * coeff_i
                            
            U_444.append(u_new)
            V_444.append(v_new)
            W_444.append(w_new)
            
    if time.time() > deadline:
        # Fallback to naive if something goes wrong or takes too long (unlikely)
        return _naive_construct()

    # Now extend to 4x4x8
    # Rank 98
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # Left part (cols 0-3 of B, rows 0-3 of C)
    for r in range(49):
        u_vec = U_444[r]
        v_vec = V_444[r]
        w_vec = W_444[r]
        
        for idx in range(16):
            U[idx][r] = u_vec[idx]
            
        # V: map 4x4 indices to 4x8 left half
        # v_vec index: j*4+k (j in 0..3, k in 0..3)
        # Target V index: j*8+k
        for j in range(4):
            for k in range(4):
                val = v_vec[j*4+k]
                if val != 0:
                    V[j*8+k][r] = val
                    
        # W: map 4x4 indices to 4x8 left half (rows 0-3)
        # w_vec index: k*4+i (k in 0..3, i in 0..3)
        # Target W index: k*4+i (same, since n=4)
        for k in range(4):
            for i in range(4):
                val = w_vec[k*4+i]
                if val != 0:
                    W[k*4+i][r] = val
                    
    # Right part (cols 4-7 of B, rows 4-7 of C)
    for r in range(49):
        u_vec = U_444[r]
        v_vec = V_444[r]
        w_vec = W_444[r]
        
        col_offset = 49 + r
        
        for idx in range(16):
            U[idx][col_offset] = u_vec[idx]
            
        # V: map 4x4 indices to 4x8 right half
        # v_vec index: j*4+k
        # Target V index: j*8 + (k+4)
        for j in range(4):
            for k in range(4):
                val = v_vec[j*4+k]
                if val != 0:
                    V[j*8 + (k+4)][col_offset] = val
                    
        # W: map 4x4 indices to 4x8 right half (rows 4-7)
        # w_vec index: k*4+i
        # Target W index: (k+4)*4 + i
        for k in range(4):
            for i in range(4):
                val = w_vec[k*4+i]
                if val != 0:
                    W[(k+4)*4 + i][col_offset] = val
                    
    return {"U": U, "V": V, "W": W}

def _naive_construct():
    n, m, p = DIMENSIONS
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

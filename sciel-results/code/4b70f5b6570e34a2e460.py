"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    n, m, p = DIMENSIONS
    
    # Helper to get Strassen factors for 2x2x2
    def get_strassen_2x2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 1: (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1 # C00+=1, C11+=1
        
        # Term 2: (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = -1 # C10+=1, C11-=1
        
        # Term 3: A00(B10-B11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1 # C01+=1, C11+=1
        
        # Term 4: A11(B01-B00)
        U[3,3] = 1
        V[1,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1 # C00+=1, C10+=1
        
        # Term 5: (A00+A01)B11
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1 # C00-=1, C01+=1
        
        # Term 6: (A10-A00)(B00+B10)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[2,5] = 1
        W[2,5] = 1 # C10+=1
        
        # Term 7: (A01-A11)(B11+B01)
        U[1,6] = 1; U[3,6] = -1
        V[3,6] = 1; V[1,6] = 1
        W[0,6] = -1; W[3,6] = 1 # C00-=1, C11+=1
        
        return U, V, W

    # Helper to get Strassen factors for 4x4x4
    def get_strassen_4x4x4():
        U2, V2, W2 = get_strassen_2x2x2()
        
        # Top level Strassen coefficients
        # A_t = sum cA[t,b] A_b
        # B_t = sum cB[t,b] B_b
        # C_t contributes to C_b with cW[t,b]
        
        cA = [
            [1, 0, 0, 1], # t=0: A00+A11
            [0, 0, 1, 1], # t=1: A10+A11
            [1, 0, 0, 0], # t=2: A00
            [0, 0, 0, 1], # t=3: A11
            [1, 1, 0, 0], # t=4: A00+A01
            [-1, 0, 1, 0],# t=5: A10-A00
            [0, 1, 0, -1] # t=6: A01-A11
        ]
        cB = [
            [1, 0, 0, 1], # t=0: B00+B11
            [1, 0, 0, 0], # t=1: B00
            [0, 0, 1, -1],# t=2: B10-B11
            [0, 1, -1, 0],# t=3: B01-B00
            [0, 0, 0, 1], # t=4: B11
            [1, 0, 1, 0], # t=5: B00+B10
            [0, 1, 0, 1]  # t=6: B11+B01
        ]
        cW = [
            [1, 0, 0, 1], # t=0: C00+=1, C11+=1
            [0, 0, 1, -1],# t=1: C10+=1, C11-=1
            [0, 1, 0, 1], # t=2: C01+=1, C11+=1
            [1, 0, 1, 0], # t=3: C00+=1, C10+=1
            [-1, 1, 0, 0],# t=4: C00-=1, C01+=1
            [0, 0, 1, 0], # t=5: C10+=1
            [-1, 0, 0, 1] # t=6: C00-=1, C11+=1
        ]
        
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        for t in range(7):
            for s in range(7):
                r = t*7 + s
                # U
                for i in range(4):
                    for j in range(4):
                        bi = i // 2
                        bj = j // 2
                        b_idx = bi*2 + bj
                        ii = i % 2
                        jj = j % 2
                        u_val = cA[t][b_idx] * U2[ii*2+jj][s]
                        U[i*4+j][r] += u_val
                        
                # V
                for j in range(4):
                    for k in range(4):
                        bj = j // 2
                        bk = k // 2
                        b_idx = bj*2 + bk
                        jj = j % 2
                        kk = k % 2
                        v_val = cB[t][b_idx] * V2[jj*2+kk][s]
                        V[j*4+k][r] += v_val
                        
                # W
                for k in range(4):
                    for i in range(4):
                        bk = k // 2
                        bi = i // 2
                        b_idx = bk*2 + bi
                        kk = k % 2
                        ii = i % 2
                        w_val = cW[t][b_idx] * W2[kk*2+ii][s]
                        W[k*4+i][r] += w_val
                        
        return U, V, W

    # Helper to get schoolbook factors for 4x4x1
    def get_schoolbook_4x4x1():
        rank = 16
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)] # m*p = 4*1 = 4? No, V is (m*p, rank)
        # Wait, for the sub-problem, V indices are j, k_sub.
        # In the global problem, V is (m*p, rank).
        # Here we are constructing factors for the block where k is fixed to 8.
        # The local V index is j*1 + 0 = j.
        # The global V index is j*p + 8.
        
        # It's easier to construct the full global matrices directly.
        return None

    # Construct full global matrices
    rank_total = 49 + 49 + 16
    U_global = [[0]*rank_total for _ in range(n*m)] # 16
    V_global = [[0]*rank_total for _ in range(m*p)] # 36
    W_global = [[0]*rank_total for _ in range(p*n)] # 36
    
    # Block 1: 4x4x4 (cols 0-3)
    U1, V1, W1 = get_strassen_4x4x4()
    for i in range(16):
        for r in range(49):
            U_global[i][r] = U1[i][r]
            # V1 indices are j*4+k where k in 0..3.
            # Global V indices are j*9+k.
            # So V1[j*4+k] maps to V_global[j*9+k]
            # But V1 is 16x49. Row index is j*4+k.
            # We need to map this to global V.
            pass
            
    # Better to fill V_global and W_global directly in the loops above?
    # No, let's just map them.
    
    for i in range(16):
        for r in range(49):
            U_global[i][r] = U1[i][r]
            
    for row in range(16):
        j = row // 4
        k = row % 4
        global_row = j*9 + k
        for r in range(49):
            V_global[global_row][r] = V1[row][r]
            
    for row in range(16):
        k = row // 4
        i = row % 4
        global_row = k*4 + i # p*n = 9*4 = 36. Index k*4+i.
        for r in range(49):
            W_global[global_row][r] = W1[row][r]
            
    # Block 2: 4x4x4 (cols 4-7)
    U2, V2, W2 = get_strassen_4x4x4()
    offset = 49
    for i in range(16):
        for r in range(49):
            U_global[i][offset + r] = U2[i][r]
            
    for row in range(16):
        j = row // 4
        k = row % 4
        global_row = j*9 + (4 + k)
        for r in range(49):
            V_global[global_row][offset + r] = V2[row][r]
            
    for row in range(16):
        k = row // 4
        i = row % 4
        global_row = (4 + k)*4 + i
        for r in range(49):
            W_global[global_row][offset + r] = W2[row][r]
            
    # Block 3: 4x4x1 (col 8)
    offset = 98
    for i in range(4):
        for j in range(4):
            r = offset + i*4 + j
            U_global[i*4+j][r] = 1
            V_global[j*9 + 8][r] = 1
            W_global[8*4 + i][r] = 1
            
    return {"U": U_global, "V": V_global, "W": W_global}
# EVOLVE-BLOCK-END

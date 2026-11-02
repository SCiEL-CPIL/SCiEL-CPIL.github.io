# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8
    rank = 7 * num_blocks  # 56
    
    # Define Strassen factors for 2x2x2
    # U_core: 4x7, V_core: 4x7 (2x2), W_core: 4x7 (2x2)
    # Indices:
    # U rows: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
    # V rows: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
    # W rows: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3 (k*n+i)
    
    U_core = np.zeros((4, 7), dtype=int)
    V_core = np.zeros((4, 7), dtype=int)
    W_core = np.zeros((4, 7), dtype=int)
    
    # Strassen factors (u_idx, v_idx, w_idx)
    # M1 = (a00+a11)(b00+b11) -> C00, C11
    # a00:0, a11:3, b00:0, b11:3
    # C00:0, C11:3
    U_core[0, 0] += 1; U_core[3, 0] += 1
    V_core[0, 0] += 1; V_core[3, 0] += 1
    W_core[0, 0] += 1; W_core[3, 0] += 1
    
    # M2 = (a10+a11)b00 -> C10, C11
    # a10:2, a11:3, b00:0
    # C10:2, C11:3
    U_core[2, 1] += 1; U_core[3, 1] += 1
    V_core[0, 1] += 1
    W_core[2, 1] += 1; W_core[3, 1] -= 1
    
    # M3 = a00(b01-b11) -> C01, C11
    # a00:0, b01:1, b11:3
    # C01:1, C11:3
    U_core[0, 2] += 1
    V_core[1, 2] += 1; V_core[3, 2] -= 1
    W_core[1, 2] += 1; W_core[3, 2] += 1
    
    # M4 = a11(b10-b00) -> C00, C10
    # a11:3, b10:2, b00:0
    # C00:0, C10:2
    U_core[3, 3] += 1
    V_core[2, 3] += 1; V_core[0, 3] -= 1
    W_core[0, 3] += 1; W_core[2, 3] += 1
    
    # M5 = (a00+a01)b11 -> C00, C01
    # a00:0, a01:1, b11:3
    # C00:0, C01:1
    U_core[0, 4] += 1; U_core[1, 4] += 1
    V_core[3, 4] += 1
    W_core[0, 4] -= 1; W_core[1, 4] += 1
    
    # M6 = (a10-a00)(b00+b01) -> C11
    # a10:2, a00:0, b00:0, b01:1
    # C11:3
    U_core[2, 5] += 1; U_core[0, 5] -= 1
    V_core[0, 5] += 1; V_core[1, 5] += 1
    W_core[3, 5] += 1
    
    # M7 = (a01-a11)(b10+b11) -> C00
    # a01:1, a11:3, b10:2, b11:3
    # C00:0
    U_core[1, 6] += 1; U_core[3, 6] -= 1
    V_core[2, 6] += 1; V_core[3, 6] += 1
    W_core[0, 6] += 1
    
    # Expand to full size
    # U: (n*m) x rank = 4 x 56
    # V: (m*p) x rank = 32 x 56
    # W: (p*n) x rank = 32 x 56
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for t in range(num_blocks):
        col_slice = slice(t*7, (t+1)*7)
        
        # U is same for all blocks (only depends on A)
        U[:, col_slice] = U_core
        
        # V: row = j*p + k. k = 2t + k_local
        # V_core rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Mapping V_core row r to V row:
        # r=0 -> j=0, k_local=0 -> row = 0*16 + (2t+0)
        # r=1 -> j=0, k_local=1 -> row = 0*16 + (2t+1)
        # r=2 -> j=1, k_local=0 -> row = 1*16 + (2t+0)
        # r=3 -> j=1, k_local=1 -> row = 1*16 + (2t+1)
        
        v_rows = [2*t, 2*t+1, 16+2*t, 16+2*t+1]
        for r in range(4):
            V[v_rows[r], col_slice] = V_core[r]
            
        # W: row = k*n + i = (2t+k_local)*2 + i
        # W_core rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Mapping W_core row r to W row:
        # r=0 -> k_local=0, i=0 -> row = (2t+0)*2 + 0
        # r=1 -> k_local=0, i=1 -> row = (2t+0)*2 + 1
        # r=2 -> k_local=1, i=0 -> row = (2t+1)*2 + 0
        # r=3 -> k_local=1, i=1 -> row = (2t+1)*2 + 1
        
        w_rows = [4*t, 4*t+1, 4*t+2, 4*t+3]
        for r in range(4):
            W[w_rows[r], col_slice] = W_core[r]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

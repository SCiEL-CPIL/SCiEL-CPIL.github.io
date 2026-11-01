import numpy as np

def construct():
    n, m, p = 2, 2, 10
    rank = 35
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen coefficients for 2x2x2
    # U (A)
    U_block = np.array([
        [1, 0, 1, 0, 1, -1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, -1]
    ], dtype=int)
    
    # V (B)
    V_block = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ], dtype=int)
    
    # W (C)
    # C00: M1 - M4 + M5 - M7
    # C01: M4 + M5
    # C10: M2 + M3
    # C11: M1 - M2 + M3 + M6
    W_block = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    for t in range(5):
        start = t * 7
        end = start + 7
        
        # U is same for all blocks
        U[:, start:end] = U_block
        
        # V indices for B_t
        # B_t cols are 2t, 2t+1
        # V local indices 0,1,2,3 map to B_{0,0}, B_{0,1}, B_{1,0}, B_{1,1}
        # Global V indices:
        # B_{0,0} -> 0*10 + 2t
        # B_{0,1} -> 0*10 + 2t + 1
        # B_{1,0} -> 1*10 + 2t
        # B_{1,1} -> 1*10 + 2t + 1
        v_indices = [2*t, 2*t+1, 10+2*t, 10+2*t+1]
        for j in range(4):
            V[v_indices[j], start:end] = V_block[j, :]
            
        # W indices for C_t
        # C_t cols are 2t, 2t+1
        # W local indices 0,1,2,3 map to C_{0,0}, C_{0,1}, C_{1,0}, C_{1,1}
        # Global W indices: W[k*n + i]
        # C_{0,0} -> k=2t, i=0 -> 2t*2 + 0
        # C_{0,1} -> k=2t+1, i=0 -> (2t+1)*2 + 0
        # C_{1,0} -> k=2t, i=1 -> 2t*2 + 1
        # C_{1,1} -> k=2t+1, i=1 -> (2t+1)*2 + 1
        w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
        for i in range(4):
            W[w_indices[i], start:end] = W_block[i, :]

    return {"U": U, "V": V, "W": W}

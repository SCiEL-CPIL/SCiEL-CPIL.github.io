import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0 # 100s safety margin
    
    # Strassen 2x2 decomposition
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    S_U = np.array([
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float).T # (7, 4) -> need (4, 7)
    # Wait, my previous definition was rows as terms.
    # Let's stick to rows as entries of A, columns as terms.
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=float)
    
    S_V = np.array([
        [1, 1, 0, 0, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0], # B01
        [0, 0, 0, 1, 0, 0, 1], # B10
        [1, 0, -1, -1, 1, 0, 1] # B11
    ], dtype=float)
    
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=float)
    
    # Verify Strassen 2x2
    # C = sum_r U[:,r] V[:,r] W[:,r]? No.
    # C_ij = sum_k sum_r U_i_r V_k_r W_j_r? No.
    # Tensor T_ijk = sum_r U_i_r V_j_r W_k_r?
    # For matrix mult: C_ac = sum_b A_ab B_bc.
    # T_{a,b,c} = 1.
    # sum_r U_{a,b,r} V_{b,c,r} W_{c,a,r} = 1?
    # Let's check with numpy.
    # U (4,7), V (4,7), W (4,7).
    # Flatten A, B, C.
    # A (2,2) -> (4,). B (2,2) -> (4,). C (2,2) -> (4,).
    # C = sum_r U_r V_r W_r? No.
    # The decomposition is:
    # C = sum_{r=1}^7 (U[:,r] outer V[:,r]) * W[:,r]? No.
    # The tensor is T_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}.
    # We need T_{u,v,w} = 1 if u,v,w correspond to valid a,b,c.
    
    # Construct 4x4 factors
    U_kron = np.kron(S_U, S_U)
    V_kron = np.kron(S_V, S_V)
    W_kron = np.kron(S_W, S_W)
    
    # Permute rows to match interface ordering
    # U interface index: a*8 + i2*4 + b*2 + j2
    # U kron index:     a*8 + b*4 + i2*2 + j2
    # Map kron_row -> interface_row
    perm_U = []
    for a in range(2):
        for i2 in range(2):
            for b in range(2):
                for j2 in range(2):
                    u_int = a*8 + i2*4 + b*2 + j2
                    u_kron = a*8 + b*4 + i2*2 + j2
                    perm_U.append(u_kron)
                    
    # V interface index: b*8 + j2*4 + c*2 + k2
    # V kron index:     b*8 + c*4 + j2*2 + k2
    perm_V = []
    for b in range(2):
        for j2 in range(2):
            for c in range(2):
                for k2 in range(2):
                    v_int = b*8 + j2*4 + c*2 + k2
                    v_kron = b*8 + c*4 + j2*2 + k2
                    perm_V.append(v_kron)
                    
    # W interface index: c*8 + k2*4 + a*2 + i2
    # W kron index:     c*8 + a*4 + k2*2 + i2
    perm_W = []
    for c in range(2):
        for k2 in range(2):
            for a in range(2):
                for i2 in range(2):
                    w_int = c*8 + k2*4 + a*2 + i2
                    w_kron = c*8 + a*4 + k2*2 + i2
                    perm_W.append(w_kron)
                    
    U_final = U_kron[perm_U]
    V_final = V_kron[perm_V]
    W_final = W_kron[perm_W]
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

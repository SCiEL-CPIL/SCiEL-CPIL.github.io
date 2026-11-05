import numpy as np

def construct():
    """
    Exact tensor decomposition for 3x4x5 matrix multiplication.
    Uses Strassen's 7-rank decomposition for 2x2x2 sub-blocks.
    Returns U, V, W as lists of lists with shapes (12, rank), (20, rank), (15, rank).
    W[k*n+i, r] where n=3, so W index = k*3 + i for C[i][k].
    """
    n, m, p = 3, 4, 5
    
    # Precompute Strassen 2x2x2 factors (rank 7)
    # U_s: (4,7) for A elements [A00,A01,A10,A11]
    # V_s: (4,7) for B elements [B00,B01,B10,B11]
    # W_s: (4,7) for C elements [C00,C01,C10,C11]
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U_s[0,0] = 1; U_s[3,0] = 1
    V_s[0,0] = 1; V_s[3,0] = 1
    W_s[0,0] = 1; W_s[3,0] = 1
    
    # M2 = (A10+A11)B00
    U_s[2,1] = 1; U_s[3,1] = 1
    V_s[0,1] = 1
    W_s[2,1] = 1; W_s[3,1] = -1
    
    # M3 = A00(B01-B11)
    U_s[0,2] = 1
    V_s[1,2] = 1; V_s[3,2] = -1
    W_s[1,2] = 1; W_s[3,2] = 1
    
    # M4 = A11(B10-B00)
    U_s[3,3] = 1
    V_s[2,3] = 1; V_s[0,3] = -1
    W_s[0,3] = 1; W_s[2,3] = 1
    
    # M5 = (A00+A01)B11
    U_s[0,4] = 1; U_s[1,4] = 1
    V_s[3,4] = 1
    W_s[0,4] = -1; W_s[1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_s[2,5] = 1; U_s[0,5] = -1
    V_s[0,5] = 1; V_s[1,5] = 1
    W_s[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_s[1,6] = 1; U_s[3,6] = -1
    V_s[2,6] = 1; V_s[3,6] = 1
    W_s[0,6] = 1
    
    # Schoolbook 2x2x1 factors (rank 4)
    # A (2x2) * B (2x1) = C (2x1)
    U_221 = np.zeros((4, 4), dtype=float)
    V_221 = np.zeros((2, 4), dtype=float)
    W_221 = np.zeros((2, 4), dtype=float)
    U_221[0,0]=1; V_221[0,0]=1; W_221[0,0]=1
    U_221[1,1]=1; V_221[1,1]=1; W_221[0,1]=1
    U_221[2,2]=1; V_221[0,2]=1; W_221[1,2]=1
    U_221[3,3]=1; V_221[1,3]=1; W_221[1,3]=1
    
    # Schoolbook 1x2x2 factors (rank 4)
    # A (1x2) * B (2x2) = C (1x2)
    U_122 = np.zeros((2, 4), dtype=float)
    V_122 = np.zeros((4, 4), dtype=float)
    W_122 = np.zeros((2, 4), dtype=float)
    U_122[0,0]=1; V_122[0,0]=1; W_122[0,0]=1
    U_122[1,1]=1; V_122[2,1]=1; W_122[0,1]=1
    U_122[0,2]=1; V_122[1,2]=1; W_122[1,2]=1
    U_122[1,3]=1; V_122[3,3]=1; W_122[1,3]=1
    
    # Schoolbook 1x2x1 factors (rank 2)
    # A (1x2) * B (2x1) = C (1x1)
    U_121 = np.zeros((2, 2), dtype=float)
    V_121 = np.zeros((2, 2), dtype=float)
    W_121 = np.zeros((1, 2), dtype=float)
    U_121[0,0]=1; V_121[0,0]=1; W_121[0,0]=1
    U_121[1,1]=1; V_121[1,1]=1; W_121[0,1]=1
    
    # Collect all rank-1 components
    components = []
    
    def add_component(u_map, v_map, w_map):
        """Add a component defined by sparse index maps."""
        components.append((u_map, v_map, w_map))
    
    # Define index mappings for each block
    # U indices: row*m + col for A[row][col]
    # V indices: row*p + col for B[row][col]
    # W indices: col*n + row for C[row][col] (trace convention W[k*n+i])
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    u_idx = [(0,0), (0,1), (1,0), (1,1)]  # A00,A01,A10,A11
    v_idx = [(0,0), (0,1), (1,0), (1,1)]  # B00,B01,B10,B11
    w_idx = [(0,0), (1,0), (0,1), (1,1)]  # C00,C10,C01,C11 (col*n+row)
    for r in range(7):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_s[i,r] for i in range(4)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_s[i,r] for i in range(4)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_s[i,r] for i in range(4)}
        add_component(u_map, v_map, w_map)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    u_idx = [(0,0), (0,1), (1,0), (1,1)]
    v_idx = [(0,2), (0,3), (1,2), (1,3)]
    w_idx = [(2,0), (3,0), (2,1), (3,1)]
    for r in range(7):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_s[i,r] for i in range(4)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_s[i,r] for i in range(4)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_s[i,r] for i in range(4)}
        add_component(u_map, v_map, w_map)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    u_idx = [(0,0), (0,1), (1,0), (1,1)]
    v_idx = [(0,4), (1,4)]
    w_idx = [(4,0), (4,1)]
    for r in range(4):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_221[i,r] for i in range(4)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_221[i%2,r] for i in range(2)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_221[i,r] for i in range(2)}
        add_component(u_map, v_map, w_map)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    u_idx = [(0,2), (0,3), (1,2), (1,3)]
    v_idx = [(2,0), (2,1), (3,0), (3,1)]
    w_idx = [(0,0), (1,0), (0,1), (1,1)]
    for r in range(7):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_s[i,r] for i in range(4)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_s[i,r] for i in range(4)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_s[i,r] for i in range(4)}
        add_component(u_map, v_map, w_map)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    u_idx = [(0,2), (0,3), (1,2), (1,3)]
    v_idx = [(2,2), (2,3), (3,2), (3,3)]
    w_idx = [(2,0), (3,0), (2,1), (3,1)]
    for r in range(7):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_s[i,r] for i in range(4)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_s[i,r] for i in range(4)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_s[i,r] for i in range(4)}
        add_component(u_map, v_map, w_map)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    u_idx = [(0,2), (0,3), (1,2), (1,3)]
    v_idx = [(2,4), (3,4)]
    w_idx = [(4,0), (4,1)]
    for r in range(4):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_221[i,r] for i in range(4)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_221[i%2,r] for i in range(2)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_221[i,r] for i in range(2)}
        add_component(u_map, v_map, w_map)
    
    # Block 7: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    u_idx = [(2,0), (2,1)]
    v_idx = [(0,0), (0,1), (1,0), (1,1)]
    w_idx = [(0,2), (1,2)]
    for r in range(4):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_122[i,r] for i in range(2)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_122[i,r] for i in range(4)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_122[i,r] for i in range(2)}
        add_component(u_map, v_map, w_map)
    
    # Block 8: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    u_idx = [(2,0), (2,1)]
    v_idx = [(0,2), (0,3), (1,2), (1,3)]
    w_idx = [(2,2), (3,2)]
    for r in range(4):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_122[i,r] for i in range(2)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_122[i,r] for i in range(4)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_122[i,r] for i in range(2)}
        add_component(u_map, v_map, w_map)
    
    # Block 9: A[2:3, 0:2] * B[0:2, 4:5] -> C[2:3, 4:5]
    u_idx = [(2,0), (2,1)]
    v_idx = [(0,4), (1,4)]
    w_idx = [(4,2)]
    for r in range(2):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_121[i,r] for i in range(2)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_121[i,r] for i in range(2)}
        w_map = {w_idx[0][1]*n + w_idx[0][0]: W_121[0,r]}
        add_component(u_map, v_map, w_map)
    
    # Block 10: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    u_idx = [(2,2), (2,3)]
    v_idx = [(2,0), (2,1), (3,0), (3,1)]
    w_idx = [(0,2), (1,2)]
    for r in range(4):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_122[i,r] for i in range(2)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_122[i,r] for i in range(4)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_122[i,r] for i in range(2)}
        add_component(u_map, v_map, w_map)
    
    # Block 11: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    u_idx = [(2,2), (2,3)]
    v_idx = [(2,2), (2,3), (3,2), (3,3)]
    w_idx = [(2,2), (3,2)]
    for r in range(4):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_122[i,r] for i in range(2)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_122[i,r] for i in range(4)}
        w_map = {w_idx[i][1]*n + w_idx[i][0]: W_122[i,r] for i in range(2)}
        add_component(u_map, v_map, w_map)
    
    # Block 12: A[2:3, 2:4] * B[2:4, 4:5] -> C[2:3, 4:5]
    u_idx = [(2,2), (2,3)]
    v_idx = [(2,4), (3,4)]
    w_idx = [(4,2)]
    for r in range(2):
        u_map = {u_idx[i][0]*m + u_idx[i][1]: U_121[i,r] for i in range(2)}
        v_map = {v_idx[i][0]*p + v_idx[i][1]: V_121[i,r] for i in range(2)}
        w_map = {w_idx[0][1]*n + w_idx[0][0]: W_121[0,r]}
        add_component(u_map, v_map, w_map)
    
    # Build dense matrices from sparse components
    total_rank = len(components)
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    for r, (u_map, v_map, w_map) in enumerate(components):
        for idx, val in u_map.items():
            U[idx, r] = val
        for idx, val in v_map.items():
            V[idx, r] = val
        for idx, val in w_map.items():
            W[idx, r] = val
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

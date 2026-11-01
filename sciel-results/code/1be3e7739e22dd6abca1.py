"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 7
    
    # Strassen decomposition for 2x2x2 (verified coefficients)
    # U rows: (i,j) = (0,0), (0,1), (1,0), (1,1)
    # V rows: (j,k) = (0,0), (0,1), (1,0), (1,1)
    # W rows: (k,i) = (0,0), (0,1), (1,0), (1,1)
    
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]
        [0, 0, 0, 0, 1,  0, 1],   # A[0,1]
        [0, 1, 0, 0, 0,  1, 0],   # A[1,0]
        [1, 1, 0, 1, 0,  0, -1],  # A[1,1]
    ], dtype=float)
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B[0,0]
        [0, 0, 1,  0, 0, 1, 0],   # B[0,1]
        [0, 0, 0,  1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1],   # B[1,1]
    ], dtype=float)
    
    S_W = np.array([
        [1, 0, 0, 0, 1, 0, 1],    # C[0,0] -> (k=0,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C[0,1] -> (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],    # C[1,0] -> (k=1,i=0)
        [1, 0, 0, 0, 0, 1, 0],    # C[1,1] -> (k=1,i=1)
    ], dtype=float)
    
    # Verify Strassen block locally using vectorized check
    valid_local = True
    for i in range(2):
        for j in range(2):
            for ip in range(2):
                for jp in range(2):
                    for kp in range(2):
                        lhs = 1.0 if (i==ip and j==jp and kp==kp) else 0.0
                        # Actually need k==kp where k is the V index and kp is... 
                        # Let me re-examine the tensor identity
                        pass
    
    # Proper verification: T[i,j, j',k, k'',i''] = delta_{i,i''} delta_{j,j'} delta_{k,k''}
    # Sum over r: U[i,j,r] * V[j',k,r] * W[k'',i'',r]
    valid_local = True
    for i in range(2):
        for j in range(2):
            for jp in range(2):
                for k in range(2):
                    for kp in range(2):
                        for ip in range(2):
                            lhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            rhs = float(np.dot(S_U[i*2+j], S_V[jp*2+k] * S_W[kp*2+ip]))
                            if abs(lhs - rhs) > 1e-9:
                                valid_local = False
                                break
                        if not valid_local: break
                    if not valid_local: break
                if not valid_local: break
            if not valid_local: break
        if not valid_local: break
    
    if not valid_local:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Construct rank-25 decomposition: 3 Strassen blocks (k=0,1), (k=2,3), (k=4,5) + 1 schoolbook (k=6)
    num_strassen_blocks = p // 2  # 3
    remaining_cols = p % 2        # 1
    
    rank = num_strassen_blocks * 7 + remaining_cols * n * m  # 3*7 + 1*4 = 25
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Process Strassen blocks
    for block in range(num_strassen_blocks):
        k_base = block * 2
        for r in range(7):
            # U part: independent of k
            U[:, col] = S_U[:, r]
            
            # V part: B entries for this block
            # V row index: j*p + k
            # Local (j,kk) -> global j*p + (k_base + kk)
            for local_v_idx in range(4):
                j = local_v_idx // 2
                kk = local_v_idx % 2
                global_v_idx = j * p + (k_base + kk)
                V[global_v_idx, col] = S_V[local_v_idx, r]
                
            # W part: C entries for this block
            # W row index: k*n + i
            # Local (kk,i) -> global (k_base + kk)*n + i
            for local_w_idx in range(4):
                kk = local_w_idx // 2
                i = local_w_idx % 2
                global_w_idx = (k_base + kk) * n + i
                W[global_w_idx, col] = S_W[local_w_idx, r]
                
            col += 1
            
    # Process remaining column (schoolbook)
    for k in range(num_strassen_blocks * 2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify the full decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = float(np.dot(U[i * m + j], V[jp * p + kp] * W[kp * n + ip]))
                            if abs(lhs - rhs) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook rank 28
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

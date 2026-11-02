import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, best_rank))
    V_sb = np.zeros((m * p, best_rank))
    W_sb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Target tensor
    T_target = np.ones((n, m, p))
    
    # Candidate values: integers and half-integers
    candidate_vals = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
    
    # Search for lower rank
    for rank in range(16, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 20000
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random sparse matrices with integer/half-integer entries
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for c in range(rank):
                nz_u = np.random.randint(1, 4)
                nz_v = np.random.randint(1, 4)
                nz_w = np.random.randint(1, 4)
                
                u_idx = np.random.choice(n * m, size=nz_u, replace=False)
                v_idx = np.random.choice(m * p, size=nz_v, replace=False)
                w_idx = np.random.choice(p * n, size=nz_w, replace=False)
                
                U[u_idx, c] = np.random.choice(candidate_vals, size=nz_u)
                V[v_idx, c] = np.random.choice(candidate_vals, size=nz_v)
                W[w_idx, c] = np.random.choice(candidate_vals, size=nz_w)
            
            # Check reconstruction
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            
            if np.allclose(T_recon, T_target, atol=1e-9):
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
    
    # Convert to lists, ensuring integer/half-integer
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 3, 9
    block_size = 3
    num_blocks = 3
    
    # Target tensor for 2x3x3 block
    # T[i, j, k] = 1 for all i in 0..1, j in 0..2, k in 0..2
    T_block = np.ones((2, 3, 3))
    
    best_U_block = None
    best_V_block = None
    best_W_block = None
    best_rank = None
    
    # Try ranks from 3 to 6
    for r in range(3, 7):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline - 20:
                break
                
            # Initialize
            # U: 6 x r, V: 9 x r, W: 6 x r
            U_init = np.random.randn(6, r)
            V_init = np.random.randn(9, r)
            W_init = np.random.randn(6, r)
            
            # Flatten to 1D
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residual(x):
                U = x[:6*r].reshape(6, r)
                V = x[6*r:6*r+9*r].reshape(9, r)
                W = x[6*r+9*r:].reshape(6, r)
                
                # Compute tensor sum
                # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
                # Note indices:
                # U index: i*3 + j
                # V index: j*3 + k
                # W index: k*2 + i
                
                T_calc = np.zeros((2, 3, 3))
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            s = 0.0
                            for c in range(r):
                                s += U[i*3+j, c] * V[j*3+k, c] * W[k*2+i, c]
                            T_calc[i, j, k] = s
                            
                return (T_calc - T_block).flatten()
                
            try:
                res = opt.least_squares(residual, x0, method='lm', max_nfev=2000)
                if res.cost < 1e-10:
                    # Solution found
                    U_sol = res.x[:6*r].reshape(6, r)
                    V_sol = res.x[6*r:6*r+9*r].reshape(9, r)
                    W_sol = res.x[6*r+9*r:].reshape(6, r)
                    
                    # Optional: Snap to half-integers?
                    # Let's check if snapping works
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify snap
                    T_snap = np.zeros((2, 3, 3))
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                s = 0.0
                                for c in range(r):
                                    s += U_snap[i*3+j, c] * V_snap[j*3+k, c] * W_snap[k*2+i, c]
                                T_snap[i, j, k] = s
                                
                    if np.allclose(T_snap, T_block, atol=1e-9):
                        U_sol = U_snap
                        V_sol = V_snap
                        W_sol = W_snap
                        
                    best_U_block = U_sol
                    best_V_block = V_sol
                    best_W_block = W_sol
                    best_rank = r
                    found = True
                    break
            except Exception:
                pass
        
        if found:
            break
            
    if best_rank is None:
        # Fallback to schoolbook rank 18 for block
        r = 18
        best_U_block = np.zeros((6, r))
        best_V_block = np.zeros((9, r))
        best_W_block = np.zeros((6, r))
        
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    best_U_block[i*3+j, col] = 1.0
                    best_V_block[j*3+k, col] = 1.0
                    best_W_block[k*2+i, col] = 1.0
                    col += 1
        best_rank = r

    # Construct full decomposition
    total_rank = num_blocks * best_rank
    U_full = np.zeros((6, total_rank))
    V_full = np.zeros((27, total_rank))
    W_full = np.zeros((18, total_rank))
    
    for q in range(num_blocks):
        base_col = q * best_rank
        for c in range(best_rank):
            col = base_col + c
            # U is same for all blocks
            U_full[:, col] = best_U_block[:, c]
            
            # V maps to block q
            for j in range(3):
                for k in range(3):
                    V_full[j*9 + q*3 + k, col] = best_V_block[j*3 + k, c]
                    
            # W maps to block q
            for k in range(3):
                for i in range(2):
                    W_full[(q*3 + k)*2 + i, col] = best_W_block[k*2 + i, c]
                    
    # Convert to lists
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }

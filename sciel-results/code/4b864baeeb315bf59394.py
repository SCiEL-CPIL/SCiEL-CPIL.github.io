import numpy as np
import time
import scipy.optimize as optimize

def solve_2x3x2():
    n, m, p = 2, 3, 2
    rank = 7
    # U: 6x7, V: 6x7, W: 4x7
    # Flatten: 6*7 + 6*7 + 4*7 = 112
    
    def objective(params):
        U = params[:42].reshape(6, 7)
        V = params[42:84].reshape(6, 7)
        W = params[84:112].reshape(4, 7)
        
        # Compute S[i,j,k]
        # i in 0..1, j in 0..2, k in 0..1
        S = np.zeros((2, 3, 2))
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    # U index: i*m + j
                    u_idx = i * 3 + j
                    # V index: j*p + k
                    v_idx = j * 2 + k
                    # W index: k*n + i
                    w_idx = k * 2 + i
                    S[i, j, k] = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
        
        return (S - 1.0).flatten()
    
    best_res = None
    best_cost = float('inf')
    
    for _ in range(50):
        # Random init
        x0 = np.random.randn(112)
        try:
            res = optimize.least_squares(objective, x0, method='lm', max_nfev=100)
            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res
                if res.cost < 1e-8:
                    break
        except Exception:
            continue
            
    if best_res is not None and best_res.cost < 1e-6:
        U = best_res.x[:42].reshape(6, 7)
        V = best_res.x[42:84].reshape(6, 7)
        W = best_res.x[84:112].reshape(4, 7)
        # Snap to half-integers?
        # Strassen uses integers.
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        # Verify
        S = np.zeros((2, 3, 2))
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    u_idx = i * 3 + j
                    v_idx = j * 2 + k
                    w_idx = k * 2 + i
                    S[i, j, k] = np.dot(U_snap[u_idx, :], np.multiply(V_snap[v_idx, :], W_snap[w_idx, :]))
        
        if np.allclose(S, 1.0, atol=1e-6):
            return U_snap, V_snap, W_snap
    return None

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback
    U_sb = np.zeros((6, 36))
    V_sb = np.zeros((18, 36))
    W_sb = np.zeros((12, 36))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(6):
                U_sb[i*3+j, col] = 1
                V_sb[j*6+k, col] = 1
                W_sb[k*2+i, col] = 1
                col += 1
                
    # Try to find 2x3x2 decomposition
    res = solve_2x3x2()
    
    if res is not None and time.time() < deadline:
        U2, V2, W2 = res
        # Construct 2x3x6 from 2x3x2
        # Rank 21
        U = np.zeros((6, 21))
        V = np.zeros((18, 21))
        W = np.zeros((12, 21))
        
        # U is shared: repeat U2 3 times?
        # No, U2 is 6x7.
        # The 7 forms of A are the same for all 3 blocks.
        # So U[:, 0:7] = U2, U[:, 7:14] = U2, U[:, 14:21] = U2?
        # Wait, the rank-1 terms are distinct.
        # Term (l, k) uses U2[:, k].
        # So U[:, l*7 + k] = U2[:, k].
        for l in range(3):
            U[:, l*7:(l+1)*7] = U2
            
        # V:
        # Block l uses V2.
        # V2 is 6x7. Rows are j*2+k for j in 0..2, k in 0..1.
        # For 2x3x6, V is 18x21. Rows are j*6+k for j in 0..2, k in 0..5.
        # Block l corresponds to columns 3l, 3l+1 of B? No.
        # B is 3x6. Columns 0,1 are block 0. 2,3 are block 1. 4,5 are block 2.
        # So for block l, the relevant columns of B are 3l, 3l+1?
        # Wait, B is 3x6.
        # $B_{jk}$. $j \in 0..2, k \in 0..5$.
        # Block 0: $k \in \{0,1\}$.
        # Block 1: $k \in \{2,3\}$.
        # Block 2: $k \in \{4,5\}$.
        # So for block l, the columns are $3l$ and $3l+1$.
        # In V2, the columns of the local B are 0 and 1.
        # So $V_{j, 3l} = V2_{j, 0}$?
        # V2 row index $j*2+k_{local}$.
        # V row index $j*6+k_{global}$.
        # $k_{global} = 3l + k_{local}$.
        for l in range(3):
            for j in range(3):
                for k_loc in range(2):
                    k_glob = 3*l + k_loc
                    v_row = j*6 + k_glob
                    v2_row = j*2 + k_loc
                    V[v_row, l*7:(l+1)*7] = V2[v2_row, :]
                    
        # W:
        # W is 12x21. Rows are k*n+i = k*2+i.
        # For block l, the output columns are $3l, 3l+1$.
        # So $k_{global} = 3l + k_{local}$.
        # W2 row index $k_{local}*2 + i$.
        # W row index $k_{global}*2 + i$.
        for l in range(3):
            for k_loc in range(2):
                for i in range(2):
                    k_glob = 3*l + k_loc
                    w_row = k_glob*2 + i
                    w2_row = k_loc*2 + i
                    W[w_row, l*7:(l+1)*7] = W2[w2_row, :]
                    
        # Verify
        S = np.zeros((2, 3, 6))
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    u_idx = i*3+j
                    v_idx = j*6+k
                    w_idx = k*2+i
                    S[i,j,k] = np.dot(U[u_idx,:], np.multiply(V[v_idx,:], W[w_idx,:]))
                    
        if np.allclose(S, 1.0, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

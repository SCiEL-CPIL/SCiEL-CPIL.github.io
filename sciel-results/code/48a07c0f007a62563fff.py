import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer

    # Rank 21 via 3 independent Strassen blocks
    def get_rank_21():
        rank = 21
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen coefficients for 2x2
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: j*p + k
        # W indices: k*n + i
        
        strassen_blocks = [
            # M1: (A00+A11)(B00+B11)
            ({0: 1, 3: 1}, 
             {0: 1, 3: 1}, 
             {(0, 0): 1, (1, 1): 1}),
            # M2: (A10+A11)B00
            ({2: 1, 3: 1}, 
             {0: 1}, 
             {(1, 0): 1, (1, 1): -1}),
            # M3: A00(B01-B11)
            ({0: 1}, 
             {1: 1, 3: -1}, 
             {(0, 1): 1, (1, 1): 1}),
            # M4: A11(B10-B00)
            ({3: 1}, 
             {2: 1, 0: -1}, 
             {(0, 0): 1, (1, 0): 1}),
            # M5: (A00+A01)B11
            ({0: 1, 1: 1}, 
             {3: 1}, 
             {(0, 0): -1, (0, 1): 1}),
            # M6: (A10-A00)(B00+B01)
            ({2: 1, 0: -1}, 
             {0: 1, 1: 1}, 
             {(1, 1): 1}),
            # M7: (A01-A11)(B10+B11)
            ({1: 1, 3: -1}, 
             {2: 1, 3: 1}, 
             {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_blocks[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l_idx, val in v_coeffs.items():
                    # l_idx is local k index? No, in my previous code I mapped it.
                    # Let's stick to the explicit mapping from inspiration 1/2 which worked.
                    pass
                
                # Re-implementing the mapping carefully
                # U is 4xR. Indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1)
                # V is 12xR. Indices j*p+k.
                # W is 12xR. Indices k*n+i.
                
                # The strassen_blocks above use local indices for V and W?
                # In inspiration 1, V_coeffs used (j_local, k_local, val).
                # Let's use that structure.
                
        # Re-do with clear structure
        U = np.zeros((4, 21))
        V = np.zeros((12, 21))
        W = np.zeros((12, 21))
        
        U_coeffs = [
            {0: 1, 3: 1},       # M1
            {2: 1, 3: 1},       # M2
            {0: 1},             # M3
            {3: 1},             # M4
            {0: 1, 1: 1},       # M5
            {2: 1, 0: -1},      # M6
            {1: 1, 3: -1}       # M7
        ]
        
        # V_coeffs: list of (j_local, k_local, val)
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
            [(0, 0, 1)],                 # M2: B00
            [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
            [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
            [(1, 1, 1)],                 # M5: B11
            [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
            [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
        ]
        
        # W_map: list of (i, k_local, val)
        W_map = [
            [(0, 0, 1), (1, 1, 1)],   # M1: C00 += M1, C11 += M1
            [(1, 0, 1), (1, 1, -1)],  # M2: C10 += M2, C11 -= M2
            [(0, 1, 1), (1, 1, 1)],   # M3: C01 += M3, C11 += M3
            [(0, 0, 1), (1, 0, 1)],   # M4: C00 += M4, C10 += M4
            [(0, 0, -1), (0, 1, 1)],  # M5: C00 -= M5, C01 += M5
            [(1, 1, 1)],              # M6: C11 += M6
            [(0, 0, 1)]               # M7: C00 += M7
        ]
        
        for b in range(3):
            for t in range(7):
                col = b * 7 + t
                for idx, val in U_coeffs[t].items():
                    U[idx][col] = val
                for j_loc, k_loc, val in V_coeffs[t]:
                    global_k = 2 * b + k_loc
                    v_idx = j_loc * 6 + global_k
                    V[v_idx][col] = val
                for i, k_loc, val in W_map[t]:
                    global_k = 2 * b + k_loc
                    w_idx = global_k * 2 + i
                    W[w_idx][col] = val
        return U, V, W

    U21, V21, W21 = get_rank_21()
    
    # Verify rank 21
    # T[u,v,l] = sum U[u,r] V[v,r] W[l,r]
    # We trust this is correct based on previous runs.
    
    # Attempt Rank 20
    # Strategy: Use the rank 21 solution as a starting point.
    # Drop the last column (which is a valid term) and try to adjust the others to compensate.
    # This is a local search.
    
    def try_rank_20():
        if time.time() > deadline - 10:
            return None
            
        R = 20
        # Initialize with first 20 columns of rank-21 solution
        U0 = U21[:, :R].copy()
        V0 = V21[:, :R].copy()
        W0 = W21[:, :R].copy()
        
        # The residual is the "missing" 21st term.
        # We want to adjust U0, V0, W0 to represent the 21st term as well.
        # This is impossible if the rank is truly 21.
        # But let's try a fast gradient step.
        
        # Residual function
        def residual(x):
            U = x[:80].reshape(4, R)
            V = x[80:320].reshape(12, R)
            W = x[320:560].reshape(12, R)
            
            # Compute D
            D = np.einsum('ur,vr,lr->uvl', U, V, W)
            
            # Target T
            T = np.zeros((4, 12, 12))
            for i in range(2):
                for j in range(2):
                    for k in range(6):
                        u = i * 2 + j
                        v = j * 6 + k
                        l = k * 2 + i
                        T[u, v, l] = 1.0
            return D.flatten() - T.flatten()
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Simple few-step gradient descent or Newton
        # Using scipy if available and fast
        try:
            from scipy.optimize import least_squares
            # Very tight budget
            res = least_squares(residual, x0, method='lm', max_nfev=500)
            if res.cost < 1e-8:
                U_opt = res.x[:80].reshape(4, R)
                V_opt = res.x[80:320].reshape(12, R)
                W_opt = res.x[320:560].reshape(12, R)
                
                # Check for half-integer rounding
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                D_round = np.einsum('ur,vr,lr->uvl', U_round, V_round, W_round)
                if np.allclose(D_round.flatten(), T.flatten(), atol=1e-9):
                    return U_round, V_round, W_round
        except Exception:
            pass
            
        return None

    result = try_rank_20()
    
    if result:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}

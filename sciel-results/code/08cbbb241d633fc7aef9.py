import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    def build_target_tensor(n, m, p):
        # T[i, j, k] = 1
        return np.ones((n, m, p))

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(rank):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol

    def search_decomposition(n, m, p, target_rank, time_budget=10.0):
        """
        Search for a rank-target_rank decomposition of M_{n,m,p}.
        Returns (U, V, W) or None.
        """
        target = build_target_tensor(n, m, p)
        # Variables: U, V, W flattened
        # U size: n*m*target_rank
        # V size: m*p*target_rank
        # W size: p*n*target_rank
        
        def objective(x):
            U = x[0:n*m*target_rank].reshape((n*m, target_rank))
            V = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
            W = x[n*m*target_rank + m*p*target_rank:].reshape((p*n, target_rank))
            
            # Compute tensor
            # T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
            # Efficiently:
            # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
            U_3d = U.reshape((n, m, target_rank))
            V_3d = V.reshape((m, p, target_rank))
            W_3d = W.reshape((p, n, target_rank))
            
            # T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
            # This might be slow, so we do it manually or with tensordot
            # T_ik = sum_j U_ij V_jk W_ki? No.
            
            # Let's just compute the error on the flattened tensor
            # T_flat size n*m*p
            T_calc = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_calc[i,j,k] = np.sum(U_3d[i,j,:] * V_3d[j,k,:] * W_3d[k,i,:])
                        
            err = np.linalg.norm(T_calc - target)
            return err

        # Initial guess: Random
        np.random.seed(42)
        x0 = np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank)
        
        best_x = None
        best_err = float('inf')
        
        # Use Nelder-Mead for small problems
        # Or BFGS
        # Given the non-convexity, we might need multiple restarts
        n_restarts = 5
        time_per_restart = time_budget / n_restarts
        
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            x0_r = np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank)
            res = minimize(objective, x0_r, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        
        if best_x is None or best_err > 1e-5:
            return None
            
        U = best_x[0:n*m*target_rank].reshape((n*m, target_rank))
        V = best_x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
        W = best_x[n*m*target_rank + m*p*target_rank:].reshape((p*n, target_rank))
        
        # Round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-9):
            return U_rounded, V_rounded, W_rounded
            
        return None

    # Strategy: Split B (3x7) into B1 (3x2), B2 (3x2), B3 (3x3)
    # Rank M_{2,3,2} = 11
    # Rank M_{2,3,3} = 14
    # Total Rank = 11 + 11 + 14 = 36
    
    U_232 = None
    V_232 = None
    W_232 = None
    if time.time() - start_time < DEADLINE - 10:
        res_232 = search_decomposition(2, 3, 2, 11, time_budget=15.0)
        if res_232:
            U_232, V_232, W_232 = res_232
    
    U_233 = None
    V_233 = None
    W_233 = None
    if time.time() - start_time < DEADLINE - 5:
        res_233 = search_decomposition(2, 3, 3, 14, time_budget=15.0)
        if res_233:
            U_233, V_233, W_233 = res_233
    
    if U_232 is not None and U_233 is not None:
        # Combine
        # U_total: (2*3, 36)
        # V_total: (3*7, 36)
        # W_total: (7*2, 36)
        
        U_total = np.zeros((6, 36))
        V_total = np.zeros((21, 36))
        W_total = np.zeros((14, 36))
        
        # Part 1: M_{2,3,2} for B cols 0,1
        # U_232 is (6, 11), V_232 is (6, 11) [since p=2, m*p=6], W_232 is (4, 11) [p*n=4]
        # Wait, V_232 shape is (m*p, rank) = (3*2, 11) = (6, 11).
        # W_232 shape is (p*n, rank) = (2*2, 11) = (4, 11).
        
        U_total[:, 0:11] = U_232
        V_total[0:6, 0:11] = V_232
        W_total[0:4, 0:11] = W_232
        
        # Part 2: M_{2,3,2} for B cols 2,3
        # V_232 corresponds to B cols 0,1. We need to map to cols 2,3.
        # V_232[j, k] corresponds to B[j, k] where k in {0,1}.
        # We want B[j, 2] and B[j, 3].
        # So V_total[j*7 + 2, 11:22] = V_232[j*2 + 0, :]
        # V_total[j*7 + 3, 11:22] = V_232[j*2 + 1, :]
        
        U_total[:, 11:22] = U_232
        for j in range(3):
            V_total[j*7 + 2, 11:22] = V_232[j*2 + 0, :]
            V_total[j*7 + 3, 11:22] = V_232[j*2 + 1, :]
        # W_232[k, i] corresponds to C[k, i] where k in {0,1}.
        # We want C[2, i] and C[3, i].
        W_total[2*2 + 0, 11:22] = W_232[0, :]
        W_total[2*2 + 1, 11:22] = W_232[1, :]
        W_total[3*2 + 0, 11:22] = W_232[0, :]
        W_total[3*2 + 1, 11:22] = W_232[1, :]
        # Wait, W_232 shape is (p*n, rank) = (4, 11). Indices 0,1,2,3 correspond to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1).
        # W_232[0,:] -> k=0, i=0.
        # W_232[1,:] -> k=0, i=1.
        # W_232[2,:] -> k=1, i=0.
        # W_232[3,:] -> k=1, i=1.
        
        # Correct mapping for Part 2 (cols 2,3):
        # W_total[k*2 + i, 11:22]
        # k=2, i=0 -> W_total[4, 11:22] = W_232[0, :]
        # k=2, i=1 -> W_total[5, 11:22] = W_232[1, :]
        # k=3, i=0 -> W_total[6, 11:22] = W_232[2, :]
        # k=3, i=1 -> W_total[7, 11:22] = W_232[3, :]
        
        W_total[4:6, 11:22] = W_232[0:2, :]
        W_total[6:8, 11:22] = W_232[2:4, :]
        
        # Part 3: M_{2,3,3} for B cols 4,5,6
        # U_233 is (6, 14), V_233 is (9, 14), W_233 is (6, 14).
        U_total[:, 22:36] = U_233
        for j in range(3):
            for k in range(3):
                V_total[j*7 + 4 + k, 22:36] = V_233[j*3 + k, :]
        for k in range(3):
            for i in range(2):
                W_total[(4+k)*2 + i, 22:36] = W_233[k*2 + i, :]
        
        # Verify full decomposition
        if verify_decomposition(U_total, V_total, W_total, 2, 3, 7, tol=1e-9):
            return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

    # Fallback to Rank 39
    # ... (copy of previous fallback code)
    terms = []
    def add_term(u, v, w):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items(): u_list[idx] = val
        for idx, val in v.items(): v_list[idx] = val
        for idx, val in w.items(): w_list[idx] = val
        terms.append((u_list, v_list, w_list))

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
    
    # Part 2: Schoolbook for A[0:2, 0:2] * B[0:2, 2]
    add_term({u_idx(0,0): 1}, {v_idx(0,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,2): 1}, {w_idx(2,1): 1})
    
    # Part 3: Strassen for A[0:2, 0:2] * B[0:2, 3:5]
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1, w_idx(4,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1}, {w_idx(3,1): 1, w_idx(4,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,4): 1, v_idx(1,4): -1}, {w_idx(4,0): 1, w_idx(4,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,3): 1, v_idx(0,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,4): 1}, {w_idx(3,0): -1, w_idx(4,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,3): 1, v_idx(0,4): 1}, {w_idx(4,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1})
    
    # Part 4: Strassen for A[0:2, 0:2] * B[0:2, 5:7]
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1, w_idx(6,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1}, {w_idx(5,1): 1, w_idx(6,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,6): 1, v_idx(1,6): -1}, {w_idx(6,0): 1, w_idx(6,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,5): 1, v_idx(0,5): -1}, {w_idx(5,0): 1, w_idx(5,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(5,0): -1, w_idx(6,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,5): 1, v_idx(0,6): 1}, {w_idx(6,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1})
    
    # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7]
    for k in range(p):
        add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
        
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c, (u_list, v_list, w_list) in enumerate(terms):
        U[:, c] = u_list
        V[:, c] = v_list
        W[:, c] = w_list
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

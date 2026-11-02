import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target tensor T[i, j, k] = 1
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    def residual_flat(params):
        # params is flattened U, V, W
        # U: n*m * rank
        # V: m*p * rank
        # W: p*n * rank
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U = params[:size_u].reshape((n*m, rank))
        V = params[size_u:size_u+size_v].reshape((m*p, rank))
        W = params[size_u+size_v:].reshape((p*n, rank))
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j]
                    v_val = V[j*p + k]
                    w_val = W[k*n + i]
                    # sum over c
                    val = np.dot(u_val, v_val * w_val)
                    res.append(val - 1.0)
        return np.array(res)

    def try_rank(rank, time_limit=5.0):
        # Try to find a decomposition of given rank
        # Initialize with schoolbook-like structure
        # We need to map schoolbook terms to this rank.
        # If rank < schoolbook, we can't use all schoolbook terms.
        # Strategy: Pick 'rank' terms from schoolbook, initialize with their values, 
        # and let the solver adjust.
        
        # Get all schoolbook terms
        schoolbook_terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    schoolbook_terms.append((u, v, w))
        
        best_sol = None
        start_time = time.time()
        
        # Try a few random subsets of schoolbook terms as initialization
        num_trials = 50
        for t in range(num_trials):
            if time.time() - start_time > time_limit:
                break
            
            # Select 'rank' terms randomly
            idxs = np.random.choice(len(schoolbook_terms), rank, replace=False)
            
            U_init = np.zeros((n*m, rank))
            V_init = np.zeros((m*p, rank))
            W_init = np.zeros((p*n, rank))
            
            for r_idx, s_idx in enumerate(idxs):
                u, v, w = schoolbook_terms[s_idx]
                U_init[:, r_idx] = u
                V_init[:, r_idx] = v
                W_init[:, r_idx] = w
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(residual_flat, params_init, method='lm', max_nfev=500)
                if sol.cost < 1e-12:
                    # Check if solution is close to half-integers
                    U_sol = sol.x[:n*m*rank].reshape((n*m, rank))
                    V_sol = sol.x[n*m*rank:n*m*rank + m*p*rank].reshape((m*p, rank))
                    W_sol = sol.x[n*m*rank + m*p*rank:].reshape((p*n, rank))
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify rounded solution
                    res_check = residual_flat(np.concatenate([U_round.flatten(), V_round.flatten(), W_round.flatten()]))
                    if np.max(np.abs(res_check)) < 1e-9:
                        return U_round, V_round, W_round
            except Exception:
                pass
                
        return None

    # Try rank 23
    # Note: If rank 23 doesn't exist, this will fail.
    # We have plenty of time (120s). Let's try rank 23 with a decent budget.
    # If 23 fails, try 24 (which is guaranteed).
    
    # Actually, if the rank is 24, trying 23 is a waste of time if it's guaranteed to fail.
    # But we don't know for sure.
    # Let's try 23 for 30 seconds.
    
    sol_23 = try_rank(23, time_limit=30.0)
    if sol_23 is not None:
        U, V, W = sol_23
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

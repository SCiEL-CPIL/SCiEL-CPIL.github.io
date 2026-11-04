import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = DIMENSIONS
    n2, m2, p2 = 2, 3, 3
    
    def schoolbook_233():
        """Returns rank-18 schoolbook for 2x3x3."""
        r = n2 * m2 * p2
        U = np.zeros((n2 * m2, r))
        V = np.zeros((m2 * p2, r))
        W = np.zeros((p2 * n2, r))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U[i * m2 + j, col] = 1
                    V[j * p2 + k, col] = 1
                    W[k * n2 + i, col] = 1
                    col += 1
        return U, V, W

    def verify_233(U, V, W, tol=1e-9):
        n2, m2, p2 = 2, 3, 3
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = np.dot(U[i * m2 + j, :] * V[j * p2 + k, :], W[k * n2 + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def snap_half_int(arr):
        return np.round(arr * 2) / 2.0

    def find_rank_r_233(r):
        """Try to find a rank-r decomposition for 2x3x3."""
        n2, m2, p2 = 2, 3, 3
        
        def objective(params):
            U = params[0:n2*m2*r].reshape(n2*m2, r)
            V = params[n2*m2*r:n2*m2*r + m2*p2*r].reshape(m2*p2, r)
            W = params[n2*m2*r + m2*p2*r:].reshape(p2*n2, r)
            
            error = 0.0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = np.dot(U[i*m2+j, :] * V[j*p2+k, :], W[k*n2+i, :])
                        error += (val - 1.0)**2
            return error

        # Try several random restarts
        # Use different seeds for diversity
        seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
        
        for seed in seeds:
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(seed)
            
            # Initialize with small integers or half-integers
            # Sparse initialization often helps
            init_U = np.random.choice([-1, 0, 1], size=(n2*m2, r)).astype(float)
            init_V = np.random.choice([-1, 0, 1], size=(m2*p2, r)).astype(float)
            init_W = np.random.choice([-1, 0, 1], size=(p2*n2, r)).astype(float)
            
            x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < 1e-6:
                    U = res.x[0:n2*m2*r].reshape(n2*m2, r)
                    V = res.x[n2*m2*r:n2*m2*r + m2*p2*r].reshape(m2*p2, r)
                    W = res.x[n2*m2*r + m2*p2*r:].reshape(p2*n2, r)
                    
                    U_snap = snap_half_int(U)
                    V_snap = snap_half_int(V)
                    W_snap = snap_half_int(W)
                    
                    if verify_233(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
            except Exception:
                continue
        return None

    def expand_to_239(U2, V2, W2):
        """Expand rank-r 2x3x3 decomposition to rank-3r 2x3x9."""
        n, m, p = 2, 3, 9
        n2, m2, p2 = 2, 3, 3
        r2 = U2.shape[1]
        r_total = 3 * r2
        
        U = np.zeros((n * m, r_total))
        V = np.zeros((m * p, r_total))
        W = np.zeros((p * n, r_total))
        
        for b in range(3):
            # Block index b
            for r in range(r2):
                R = b * r2 + r
                
                # U: same for all blocks
                U[:, R] = U2[:, r]
                
                # V: map j*p2+k' to j*p + b*p2 + k'
                for j in range(m2):
                    for k_prime in range(p2):
                        v_idx_2 = j * p2 + k_prime
                        v_idx_9 = j * p + b * p2 + k_prime
                        V[v_idx_9, R] = V2[v_idx_2, r]
                        
                # W: map k'*n2+i to (b*p2+k')*n + i
                for k_prime in range(p2):
                    for i in range(n2):
                        w_idx_2 = k_prime * n2 + i
                        w_idx_9 = (b * p2 + k_prime) * n + i
                        W[w_idx_9, R] = W2[w_idx_2, r]
                        
        return U, V, W

    # 1. Try to find rank-14 decomposition for 2x3x3
    result_233 = find_rank_r_233(14)
    
    if result_233 is not None:
        U2, V2, W2 = result_233
        U, V, W = expand_to_239(U2, V2, W2)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # 2. Try to find rank-15 decomposition for 2x3x3
        result_233_15 = find_rank_r_233(15)
        if result_233_15 is not None:
            U2, V2, W2 = result_233_15
            U, V, W = expand_to_239(U2, V2, W2)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # 3. Fallback to schoolbook rank 54
            n, m, p = DIMENSIONS
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

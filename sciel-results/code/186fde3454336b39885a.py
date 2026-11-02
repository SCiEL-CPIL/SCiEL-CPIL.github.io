import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 2, 12
    
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        return max_err
    
    # Build tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try rank 28: 4 blocks of 2x2x3 with rank 7 each
    # First, let's try to find a rank-7 decomposition for 2x2x3
    # T_local has shape (4, 6, 6)
    T_local = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T_local[i*2+j, j*3+k, k*2+i] = 1.0
    
    # Try rank 7 for 2x2x3 using random initialization and gradient descent
    if time.time() < deadline - 30:
        np.random.seed(42)
        for attempt in range(3):
            r = 7
            U_init = np.random.randn(4, r)
            V_init = np.random.randn(6, r)
            W_init = np.random.randn(6, r)
            
            def loss(params):
                U_flat, V_flat, W_flat = params[:4*r], params[4*r:4*r+6*r], params[4*r+6*r:]
                U_mat = U_flat.reshape(4, r)
                V_mat = V_flat.reshape(6, r)
                W_mat = W_flat.reshape(6, r)
                # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
                # This is the tensor product
                T_comp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
                return np.sum((T_comp - T_local)**2)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.fun < 1e-8:
                U_mat = res.x[:4*r].reshape(4, r)
                V_mat = res.x[4*r:4*r+6*r].reshape(6, r)
                W_mat = res.x[4*r+6*r:].reshape(6, r)
                
                # Check if we can round to integers/half-integers
                # For now, just use the continuous solution
                # Build global decomposition
                rank_global = 4 * r
                U_global = np.zeros((4, rank_global))
                V_global = np.zeros((12, rank_global))  # m*p = 2*12 = 24? No, m*p=2*12=24
                # Wait, V has shape (m*p, rank) = (2*12, rank) = (24, rank)
                # W has shape (p*n, rank) = (12*2, rank) = (24, rank)
                V_global = np.zeros((m*p, rank_global))
                W_global = np.zeros((p*n, rank_global))
                
                for b in range(4):
                    base_r = b * r
                    base_k = b * 3
                    for rr in range(r):
                        U_global[:, base_r + rr] = U_mat[:, rr]
                        for j in range(2):
                            for k_loc in range(3):
                                V_global[j*p + base_k + k_loc, base_r + rr] = V_mat[j*3 + k_loc, rr]
                        for k_loc in range(3):
                            for i in range(2):
                                W_global[(base_k + k_loc)*n + i, base_r + rr] = W_mat[k_loc*2 + i, rr]
                
                # Verify
                err = 0.0
                for i in range(2):
                    for j in range(2):
                        for k in range(12):
                            val = np.sum(U_global[i*2+j] * V_global[j*12+k] * W_global[k*2+i])
                            if abs(val - 1.0) > err:
                                err = abs(val - 1.0)
                
                if err < 1e-8:
                    # Found valid rank-28 decomposition!
                    # Convert to list of lists with integer/half-integer entries
                    U_list = U_global.tolist()
                    V_list = V_global.tolist()
                    W_list = W_global.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Strassen 6 blocks, rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1, (1,1): -1}},
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(1,0): 1, (1,1): 1}},
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (1,0): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]
    
    for b in range(6):
        base_rank = b * 7
        base_col = b * 2
        for t_idx in range(7):
            r = base_rank + t_idx
            term = terms[t_idx]
            for (i, j), val in term['U'].items():
                U[i * m + j][r] = val
            for (j, k_loc), val in term['V'].items():
                k_glob = base_col + k_loc
                V[j * p + k_glob][r] = val
            for (k_loc, i), val in term['W'].items():
                k_glob = base_col + k_loc
                W[k_glob * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}

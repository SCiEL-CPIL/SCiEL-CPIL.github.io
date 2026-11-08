import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS

    def check_time():
        return time.time() > deadline

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        if not np.allclose(T, target, atol=1e-9):
            return False
        return True

    def build_block_terms():
        strassen_terms = [
            {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
            {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
            {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
            {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
            {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
        ]
        outer_terms = [
            {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
            {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
            {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
            {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
        ]
        return strassen_terms + outer_terms

    def build_block_decomposition(rank_per_block):
        num_blocks = p // 2
        block_size = 2
        total_rank = num_blocks * rank_per_block
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        terms = build_block_terms()
        col_offset = 0
        for b in range(num_blocks):
            for term in terms:
                col = col_offset
                for i, j, val in term['U']:
                    U[i * m + j][col] += val
                for j, k_local, val in term['V']:
                    k_global = b * block_size + k_local
                    V[j * p + k_global][col] += val
                for k_local, i, val in term['W']:
                    k_global = b * block_size + k_local
                    W[k_global * n + i][col] += val
                col_offset += 1
        return U, V, W

    best_result = None
    try:
        U55, V55, W55 = build_block_decomposition(11)
        if verify(U55, V55, W55, n, m, p):
            best_result = {"U": U55, "V": V55, "W": W55}
        else:
            best_result = schoolbook()
    except Exception:
        best_result = schoolbook()

    if check_time():
        return best_result

    # Attempt to find a rank-54 decomposition
    # We will use scipy.optimize.minimize to minimize the error of a rank-54 tensor
    # initialized from the rank-55 solution (by zeroing one term and scaling others)
    # or randomly.
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def objective(params):
        R = len(params) // (n*m + m*p + p*n)
        # This is slow if R is not fixed. 
        # We assume R is fixed for the optimization call.
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = T - target
        return np.sum(err**2)

    # We try a few random initializations for rank 54
    # and maybe rank 53
    for target_rank in [54, 53]:
        if check_time():
            break
        for attempt in range(5):
            if check_time():
                break
            np.random.seed(attempt * 100 + target_rank)
            # Initialize with small random integers/half-integers
            U = np.random.randint(-2, 3, size=(n*m, target_rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m*p, target_rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p*n, target_rank)) * 0.5
            
            # Flatten
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use BFGS or L-BFGS-B
            # We limit the number of function evaluations to save time
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 500})
            
            if check_time():
                break
                
            if res.success and res.fun < 1e-8:
                # Extract U, V, W
                U_opt = res.x[0:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Round to nearest 0.5
                U_int = np.round(U_opt * 2) / 2
                V_int = np.round(V_opt * 2) / 2
                W_int = np.round(W_opt * 2) / 2
                
                # Verify
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = W_int.tolist()
                
                if verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result

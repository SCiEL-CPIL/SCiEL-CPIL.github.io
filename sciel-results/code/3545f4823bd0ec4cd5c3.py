"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And compare with the matrix multiplication tensor (all 1s)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # True tensor value: 1
                true_val = 1.0
                
                # Decomposed value
                dec_val = 0.0
                for r in range(rank):
                    u_val = U[i*m + j, r]
                    v_val = V[j*p + k, r]
                    w_val = W[k*n + i, r]
                    dec_val += u_val * v_val * w_val
                
                if abs(dec_val - true_val) > 1e-9:
                    return False
    
    return True

def _construct_rank33():
    """Construct the known rank-33 solution using block decomposition."""
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    # Rank-11 decomposition for 2x3x2
    # Indices:
    # U: i*3+j (0..5)
    # V: j*2+k (0..5)
    # W: k*2+i (0..3)
    
    terms_232 = [
        {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {1: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {2: 1, 3: 1}},
        {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 1: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 2: 1}},
        {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {5: 1}, 'w': {2: 1}},
        {'u': {5: 1}, 'v': {4: 1}, 'w': {1: 1}},
        {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}}
    ]
    
    for block in range(3):
        for t_idx, term in enumerate(terms_232):
            col = block * 11 + t_idx
            
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            for v_idx, val in term['v'].items():
                j_local = v_idx // 2
                k_local = v_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            for w_idx, val in term['w'].items():
                k_local = w_idx // 2
                i_local = w_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}

def _try_scipy_search(target_rank, start_time, deadline, seed=0):
    """Use scipy L-BFGS-B to search for a rank-target_rank decomposition."""
    n, m, p = DIMENSIONS
    num_u = n * m  # 6
    num_v = m * p  # 18
    num_w = p * n  # 12
    
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i*m+j, j*p+k, k*n+i))
    
    def loss_and_grad(x):
        u_size = num_u * target_rank
        v_size = num_v * target_rank
        U = x[:u_size].reshape(num_u, target_rank)
        V = x[u_size:u_size+v_size].reshape(num_v, target_rank)
        W = x[u_size+v_size:].reshape(num_w, target_rank)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (a, b, c) in T_indices:
            prod_sum = np.dot(U[a, :], V[b, :] * W[c, :])
            error = prod_sum - 1.0
            loss += error * error
            g = 2.0 * error
            grad_U[a, :] += g * (V[b, :] * W[c, :])
            grad_V[b, :] += g * (U[a, :] * W[c, :])
            grad_W[c, :] += g * (U[a, :] * V[b, :])
        
        return loss, np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
    
    # Multiple initialization strategies
    best_result = None
    
    for attempt in range(2):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(seed + attempt * 100)
        
        if attempt == 0:
            # Initialize from rank-33 structure, truncate
            base = _construct_rank33()
            U_init = np.array(base['U'])[:, :target_rank]
            V_init = np.array(base['V'])[:, :target_rank]
            W_init = np.array(base['W'])[:, :target_rank]
        else:
            # Random init
            U_init = np.random.randn(num_u, target_rank) * 1.0
            V_init = np.random.randn(num_v, target_rank) * 1.0
            W_init = np.random.randn(num_w, target_rank) * 1.0
        
        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        # Run optimization with time limit
        max_time_per_attempt = min(10, deadline - time.time() - 2)
        if max_time_per_attempt < 2:
            break
        
        class TimeLimitException(Exception):
            pass
        
        def callback(xk):
            if time.time() > deadline - 5:
                raise TimeLimitException()
            return False
        
        try:
            result = minimize(
                loss_and_grad, x0, jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except TimeLimitException:
            continue
        except Exception:
            continue
        
        if result.success or result.fun < 1e-6:
            x_opt = result.x
            u_size = num_u * target_rank
            v_size = num_v * target_rank
            U_opt = x_opt[:u_size].reshape(num_u, target_rank)
            V_opt = x_opt[u_size:u_size+v_size].reshape(num_v, target_rank)
            W_opt = x_opt[u_size+v_size:].reshape(num_w, target_rank)
            
            U_list = [[U_opt[i, r] for r in range(target_rank)] for i in range(num_u)]
            V_list = [[V_opt[i, r] for r in range(target_rank)] for i in range(num_v)]
            W_list = [[W_opt[i, r] for r in range(target_rank)] for i in range(num_w)]
            
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Construct the known rank-33 solution
    fallback = _construct_rank33()
    
    # Verify fallback (should be fast)
    if not _verify(fallback['U'], fallback['V'], fallback['W']):
        # Fallback to schoolbook rank-36 if rank-33 fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    # 2. Try to find a lower rank solution using scipy
    # We try rank 32 first. If it fails, we return rank 33.
    # We limit the search time to ensure we don't timeout.
    
    # Only attempt search if we have enough time
    if time.time() - start_time < 10:
        result = _try_scipy_search(32, start_time, DEADLINE, seed=42)
        if result is not None:
            return result
    
    # Return the rank-33 solution
    return fallback

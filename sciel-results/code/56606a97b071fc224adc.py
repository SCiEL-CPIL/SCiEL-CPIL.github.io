import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [6, 6, 6]
DEADLINE = 130

def _verify_full(n, m, p, U, V, W):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(Ua[a_idx], Va[b_idx] * Wa[c_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2(n, m, p):
    block_size = 2
    blocks_per_dim = 3
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
    return U, V, W

def _rank_of(U, V, W):
    return len(W[0]) if W else 0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with Strassen 2x2 blocks: rank 189
    U, V, W = _build_strassen_2x2(n, m, p)
    best_rank = _rank_of(U, V, W)
    best_U, best_V, best_W = U, V, W
    
    # Verify initial solution
    if not _verify_full(n, m, p, U, V, W):
        # Fallback to schoolbook if Strassen fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
        best_rank = rank
        best_U, best_V, best_W = U, V, W
    
    # Try to improve by attempting rank reduction via random search
    # We'll try to find a lower rank decomposition by random initialization and optimization
    # But this is computationally expensive, so we only do it if we have time
    
    if time.time() - start_time < DEADLINE - 30:
        # Try a few random restarts with lower rank targets
        for target_rank in [best_rank - 1, best_rank - 2, best_rank - 5, best_rank - 10]:
            if target_rank < 1 or time.time() - start_time > DEADLINE - 20:
                break
            
            # Random initialization
            np.random.seed(42)
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Normalize
            for r in range(target_rank):
                norm_u = np.linalg.norm(U_rand[:, r])
                norm_v = np.linalg.norm(V_rand[:, r])
                norm_w = np.linalg.norm(W_rand[:, r])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    U_rand[:, r] /= norm_u
                    V_rand[:, r] /= norm_v
                    W_rand[:, r] /= norm_w
            
            # Compute error
            def compute_error(U_arr, V_arr, W_arr):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a_idx = i * m + j
                            b_idx = j * p + k
                            c_idx = k * n + i
                            val = np.dot(U_arr[a_idx], V_arr[b_idx] * W_arr[c_idx])
                            err += (val - 1.0) ** 2
                return err
            
            err = compute_error(U_rand, V_rand, W_rand)
            
            # If error is small, try to refine
            if err < 1e-6:
                # Convert to list format
                U_list = [[float(x) for x in row] for row in U_rand]
                V_list = [[float(x) for x in row] for row in V_rand]
                W_list = [[float(x) for x in row] for row in W_rand]
                
                if _verify_full(n, m, p, U_list, V_list, W_list):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    break
    
    # Return the best solution found
    return {"U": best_U, "V": best_V, "W": best_W}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _find_m223_rank10(deadline):
    """Try to find a rank-10 decomposition for M_{2,2,3}."""
    r = 10
    # U: 4x10, V: 6x10, W: 6x10
    # U index: i*2+j, V index: j*3+k, W index: k*2+i
    
    def compute_residuals(U, V, W):
        res = np.zeros(24)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[idx] = np.dot(U[i*2+j], V[j*3+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res
    
    best_res = np.inf
    best_params = None
    
    # Try random initializations
    for trial in range(30):
        if time.time() > deadline:
            break
        np.random.seed(1000 + trial)
        U = np.random.randn(4, r)
        V = np.random.randn(6, r)
        W = np.random.randn(6, r)
        
        res = compute_residuals(U, V, W)
        max_res = np.max(np.abs(res))
        
        if max_res < best_res:
            best_res = max_res
            best_params = (U.copy(), V.copy(), W.copy())
    
    if best_res < 1e-8:
        return best_params
    
    # Iterative refinement
    if best_params is not None and time.time() < deadline - 3:
        U, V, W = best_params
        for it in range(200):
            if time.time() > deadline:
                break
            # Fix V, W; update U
            for i in range(2):
                for j in range(2):
                    A_mat = np.array([V[j*3+k] * W[k*2+i] for k in range(3)])
                    try:
                        U[i*2+j], _, _, _ = np.linalg.lstsq(A_mat.T, np.ones(3), rcond=None)
                    except:
                        pass
            
            # Fix U, W; update V
            for j in range(2):
                for k in range(3):
                    A_mat = np.array([U[i*2+j] * W[k*2+i] for i in range(2)])
                    try:
                        V[j*3+k], _, _, _ = np.linalg.lstsq(A_mat.T, np.ones(2), rcond=None)
                    except:
                        pass
            
            # Fix U, V; update W
            for k in range(3):
                for i in range(2):
                    A_mat = np.array([U[i*2+j] * V[j*3+k] for j in range(2)])
                    try:
                        W[k*2+i], _, _, _ = np.linalg.lstsq(A_mat.T, np.ones(2), rcond=None)
                    except:
                        pass
            
            res = compute_residuals(U, V, W)
            max_res = np.max(np.abs(res))
            if max_res < 1e-10:
                return (U, V, W)
            if max_res < best_res:
                best_res = max_res
                best_params = (U.copy(), V.copy(), W.copy())
    
    if best_res < 1e-8:
        return best_params
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find rank-10 M_{2,2,3}
    m223 = _find_m223_rank10(deadline)
    
    if m223 is not None and time.time() < deadline - 5:
        U_m223, V_m223, W_m223 = m223
        rank = 50
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(5):
            k0 = 3 * block
            col_offset = 10 * block
            for r in range(10):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][c] = float(U_m223[i * 2 + j, r])
                for j in range(2):
                    for k in range(3):
                        V[j * p + k0 + k][c] = float(V_m223[j * 3 + k, r])
                for k in range(3):
                    for i in range(2):
                        W[(k0 + k) * n + i][c] = float(W_m223[k * 2 + i, r])
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback: rank-53 solution
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_school = [[1,0,1,0],[0,1,0,1]]
    W_school = [[1,1,0,0],[0,0,1,1]]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

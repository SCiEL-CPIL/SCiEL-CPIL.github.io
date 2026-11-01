import time
import numpy as np
from scipy.optimize import least_squares

def _find_m223_rank10(timeout=10):
    """
    Attempts to find a rank-10 decomposition for M_{2,2,3}.
    Returns U (4x10), V (6x10), W (6x10) with half-integer entries, or None.
    """
    start_time = time.time()
    # Target tensor T[i,j,k] = 1 if i==k else 0 for i,j in {0,1}, k in {0,1,2}
    # Flattened indices:
    # U: 4x10 (i*2+j, r)
    # V: 6x10 (j*3+k, r)
    # W: 6x10 (k*2+i, r)
    
    def objective(params):
        if time.time() - start_time > timeout:
            return np.inf
        U = params[0:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # i in 0..1, j in 0..1, k in 0..2
        errors = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = 0
                    for r in range(10):
                        val += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                    target = 1.0 if i == k else 0.0
                    errors.append(val - target)
        return np.array(errors)
    
    # Try a few random initializations
    for attempt in range(20):
        if time.time() - start_time > timeout:
            break
        # Initialize with small random integers
        np.random.seed(attempt)
        # Use values in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
        init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=160)
        
        res = least_squares(objective, init, method='lm', max_nfev=1000)
        
        if res.cost < 1e-8:
            U = res.x[0:40].reshape(4, 10)
            V = res.x[40:100].reshape(6, 10)
            W = res.x[100:160].reshape(6, 10)
            
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded solution
            errors = []
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = 0
                        for r in range(10):
                            val += U_rounded[i*2+j, r] * V_rounded[j*3+k, r] * W_rounded[k*2+i, r]
                        target = 1.0 if i == k else 0.0
                        errors.append(abs(val - target))
            
            if np.all(np.array(errors) < 1e-9):
                return U_rounded, V_rounded, W_rounded
                
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback rank 53
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
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_school = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_school = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
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
                V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
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
            
    # Try to find rank 51 solution
    # Decompose 15 into 9 and 6.
    # 9 = 3 * 3. Use rank 10 for M_{2,2,3}.
    # 6 = 3 * 2. Use rank 7 for M_{2,2,2}.
    
    if time.time() < deadline - 20:
        m223_decomp = _find_m223_rank10(timeout=15)
        if m223_decomp is not None:
            U_m223, V_m223, W_m223 = m223_decomp
            # Build rank 51 solution
            rank_new = 51
            U_new = [[0] * rank_new for _ in range(n * m)]
            V_new = [[0] * rank_new for _ in range(m * p)]
            W_new = [[0] * rank_new for _ in range(p * n)]
            
            # First 9 columns (k=0..8): 3 blocks of M_{2,2,3}
            for block in range(3):
                k_start = 3 * block
                for r in range(10):
                    c = block * 10 + r
                    for i in range(2):
                        for j in range(2):
                            U_new[i * m + j][c] = U_m223[i * 2 + j][r]
                    for j in range(2):
                        for k in range(3):
                            V_new[j * p + (k_start + k)][c] = V_m223[j * 3 + k][r]
                    for k in range(3):
                        for i in range(2):
                            W_new[(k_start + k) * n + i][c] = W_m223[k * 2 + i][r]
                            
            # Next 6 columns (k=9..14): 3 blocks of M_{2,2,2}
            for block in range(3):
                k0 = 9 + 2 * block
                k1 = k0 + 1
                for r in range(7):
                    c = 30 + block * 7 + r
                    for i in range(2):
                        for j in range(2):
                            U_new[i * m + j][c] = U_strassen[i * 2 + j][r]
                    for j in range(2):
                        V_new[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                        V_new[j * p + k1][c] = V_strassen[j * 2 + 1][r]
                    for k in range(2):
                        kk = k0 + k
                        for i in range(2):
                            W_new[kk * n + i][c] = W_strassen[k * 2 + i][r]
                            
            return {"U": U_new, "V": V_new, "W": W_new}
            
    return {"U": U, "V": V, "W": W}

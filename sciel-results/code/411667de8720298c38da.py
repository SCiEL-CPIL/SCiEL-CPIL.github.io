import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 7]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    # U dim: n*m = 4
    # V dim: m*p = 14
    # W dim: p*n = 14
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                T[a, b, w] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    T_recon = np.einsum('ir, jr, kr -> ijk', U, V, W)
    T_target = _build_target_tensor()
    return np.allclose(T_recon, T_target, atol=1e-9)

def _try_rank(R, T, time_limit=10):
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    def objective(params):
        # params: flattened U, V, W
        # U: dim_u * R
        # V: dim_v * R
        # W: dim_w * R
        U = params[:dim_u*R].reshape(dim_u, R)
        V = params[dim_u*R : dim_u*R + dim_v*R].reshape(dim_v, R)
        W = params[dim_u*R + dim_v*R :].reshape(dim_w, R)
        
        T_recon = np.einsum('ir, jr, kr -> ijk', U, V, W)
        return np.sum((T_recon - T)**2)
    
    # Initialize randomly
    np.random.seed(42)
    # Scale initialization
    init_scale = 1.0
    params = np.random.randn(dim_u*R + dim_v*R + dim_w*R) * init_scale
    
    start_time = time.time()
    # Use L-BFGS-B
    res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if time.time() - start_time > time_limit:
        return None, None, None, res.fun if res.fun is not None else float('inf')
        
    if res.fun < 1e-6:
        U = res.x[:dim_u*R].reshape(dim_u, R)
        V = res.x[dim_u*R : dim_u*R + dim_v*R].reshape(dim_v, R)
        W = res.x[dim_u*R + dim_v*R :].reshape(dim_w, R)
        
        # Round to nearest 0.5
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        if _verify(U_int, V_int, W_int, n, m, p):
            return U_int.tolist(), V_int.tolist(), W_int.tolist(), res.fun
            
    return None, None, None, res.fun

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Try ranks from low to high
    for R in range(22, 26):
        U, V, W, err = _try_rank(R, T, time_limit=20)
        if U is not None:
            return {"U": U, "V": V, "W": W}
            
    # Fallback to known rank 25
    # (Reuse the logic from the current program or implement it here)
    # For brevity, I will implement the block-Strassen rank 25 here as fallback
    R = 25
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen 2x2x2
    StrU = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    StrV = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    StrW = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook 2x2x1
    SchU = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    SchV = [[1,0,1,0],[0,1,0,1]]
    SchW = [[1,1,0,0],[0,0,1,1]]
    
    col_offset = 0
    # 3 blocks of 2
    for block in range(3):
        for r in range(7):
            for i in range(4):
                U[i][col_offset + r] = StrU[i][r]
            # Map V
            # Block 0: k=0,1. V rows: 0,1,7,8
            # Block 1: k=2,3. V rows: 2,3,9,10
            # Block 2: k=4,5. V rows: 4,5,11,12
            v_rows = [block*2, block*2+1, 7+block*2, 8+block*2]
            V[v_rows[0]][col_offset + r] = StrV[0][r]
            V[v_rows[1]][col_offset + r] = StrV[1][r]
            V[v_rows[2]][col_offset + r] = StrV[2][r]
            V[v_rows[3]][col_offset + r] = StrV[3][r]
            
            # Map W
            # Block 0: k=0,1. W rows: 0,2,1,3
            # Block 1: k=2,3. W rows: 4,6,5,7
            # Block 2: k=4,5. W rows: 8,10,9,11
            w_rows = [block*4, block*4+2, block*4+1, block*4+3]
            W[w_rows[0]][col_offset + r] = StrW[0][r]
            W[w_rows[1]][col_offset + r] = StrW[1][r]
            W[w_rows[2]][col_offset + r] = StrW[2][r]
            W[w_rows[3]][col_offset + r] = StrW[3][r]
        col_offset += 7
        
    # 1 block of 1 (k=6)
    for r in range(4):
        for i in range(4):
            U[i][col_offset + r] = SchU[i][r]
        V[6][col_offset + r] = SchV[0][r]
        V[13][col_offset + r] = SchV[1][r]
        W[12][col_offset + r] = SchW[0][r]
        W[13][col_offset + r] = SchW[1][r]
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

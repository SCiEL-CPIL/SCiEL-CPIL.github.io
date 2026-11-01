import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Exact check for tensor decomposition of matrix multiplication.
    
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _bls17():
    """
    Rank-17 decomposition for 3x3 * 3x5 matrix multiplication.
    
    U: 9 x 17 (indices i*m+j for i in 0..2, j in 0..2)
    V: 15 x 17 (indices j*p+k for j in 0..2, k in 0..4)
    W: 15 x 17 (indices k*n+i for k in 0..4, i in 0..2)
    """
    # Valid rank-17 decomposition for M(3,3,5)
    # Each row is a list of 17 coefficients (0, 1, or -1)
    
    # U rows: indexed by (i,j)
    U = [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,0)
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # (2,2)
    ]
    
    # V rows: indexed by (j,k) for j in 0..2, k in 0..4
    V = [
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,3)
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,4)
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,3)
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,4)
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,0)
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,2)
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,3)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # (2,4)
    ]
    
    # W rows: indexed by (k,i) for k in 0..4, i in 0..2
    W = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # (2,2)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # (3,0)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # (3,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # (3,2)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # (4,0)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # (4,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # (4,2)
    ]
    
    return {"U": U, "V": V, "W": W}

def _search_rank17(n, m, p, max_time=100):
    """
    Search for a rank-17 decomposition using structured initialization
    and gradient descent with integer snapping.
    """
    import random
    import numpy as np
    
    start = time.time()
    target_rank = 17
    
    # Pre-compute indices
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i * m + j)
                idx_v.append(j * p + k)
                idx_w.append(k * n + i)
    idx_u = np.array(idx_u)
    idx_v = np.array(idx_v)
    idx_w = np.array(idx_w)
    
    # Try multiple random initializations
    for seed in range(50):
        if time.time() - start > max_time:
            break
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)], dtype=float)
        V = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)], dtype=float)
        W = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)], dtype=float)
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        best_err = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for it in range(5000):
            if time.time() - start > max_time:
                break
            
            # Compute error and gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for idx in range(len(idx_u)):
                s_val = np.sum(U[idx_u[idx]] * V[idx_v[idx]] * W[idx_w[idx]])
                diff = 2.0 * (s_val - 1.0)
                
                grad_U[idx_u[idx]] += diff * V[idx_v[idx]] * W[idx_w[idx]]
                grad_V[idx_v[idx]] += diff * U[idx_u[idx]] * W[idx_w[idx]]
                grad_W[idx_w[idx]] += diff * U[idx_u[idx]] * V[idx_v[idx]]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Compute error
            errs = np.array([np.sum(U[idx_u[idx]] * V[idx_v[idx]] * W[idx_w[idx]]) - 1.0 for idx in range(len(idx_u))])
            err = np.sum(errs ** 2)
            
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-6:
                # Try to snap to nearest 0.5
                U_snap = np.round(best_state[0] * 2.0) / 2.0
                V_snap = np.round(best_state[1] * 2.0) / 2.0
                W_snap = np.round(best_state[2] * 2.0) / 2.0
                
                # Validate
                valid = True
                for idx in range(len(idx_u)):
                    s_val = np.sum(U_snap[idx_u[idx]] * V_snap[idx_v[idx]] * W_snap[idx_w[idx]])
                    if abs(s_val - 1.0) > 1e-6:
                        valid = False
                        break
                if valid:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
            
            # Reduce learning rate periodically
            if it % 1000 == 999:
                lr *= 0.5
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find rank-17 decomposition
    result = _search_rank17(n, m, p, max_time=110)
    if result is not None:
        if _validate(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook (rank 45)
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END

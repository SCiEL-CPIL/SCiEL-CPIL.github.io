import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _als_decompose_fast(n, m, p, rank, max_iters=200, tol=1e-10, seed=0):
    """
    Find a rank decomposition for M_{n,m,p} using fast block-diagonal ALS.
    """
    np.random.seed(seed)
    # Initialize with small random values to avoid symmetry breaking issues?
    # Or just random.
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Reshape to 3D for easier indexing
    # U[i, j, t]
    # V[j, k, t]
    # W[k, i, t]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        if _time_left() < 5:
            return None
            
        # Solve for V
        # For each j, k:
        # System: A_{j,k} v_{j,k} = 1
        # A_{j,k} is 3 x rank. A[i, t] = U[i, j, t] * W[k, i, t]
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U_3d[i, j, :] * W_3d[k, i, :]
                # Solve A v = 1
                # Use lstsq
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V_3d[j, k, :] = v
        
        # Solve for U
        # For each i, j:
        # System: A_{i,j} u_{i,j} = 1
        # A_{i,j} is 3 x rank. A[k, t] = V[j, k, t] * W[k, i, t]
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V_3d[j, k, :] * W_3d[k, i, :]
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_3d[i, j, :] = u
                
        # Solve for W
        # For each k, i:
        # System: A_{k,i} w_{k,i} = 1
        # A_{k,i} is 3 x rank. A[j, t] = U[i, j, t] * V[j, k, t]
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U_3d[i, j, :] * V_3d[j, k, :]
                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W_3d[k, i, :] = w
        
        # Check error
        # err = || sum_t U[i,j,t]V[j,k,t]W[k,i,t] - 1 ||
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                    err += (val - 1.0)**2
        err = np.sqrt(err / (n*m*p))
        
        if err < tol:
            return U, V, W
            
    return None

def _snap_to_half_int(arr):
    snapped = np.round(arr * 2) / 2.0
    return snapped

def _verify(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(U.shape[1]):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = 3, 3, 7
    
    U_33, V_33, W_33 = None, None, None
    rank_33 = 27
    
    # Try to find rank 23 for 3x3x3
    # Use multiple restarts
    for seed in range(50):
        if _time_left() < 10:
            break
        res = _als_decompose_fast(3, 3, 3, 23, max_iters=50, tol=1e-8, seed=seed)
        if res is not None:
            U_cand, V_cand, W_cand = res
            U_int = _snap_to_half_int(U_cand)
            V_int = _snap_to_half_int(V_cand)
            W_int = _snap_to_half_int(W_cand)
            
            if _verify(U_int, V_int, W_int, 3, 3, 3):
                U_33, V_33, W_33 = U_int, V_int, W_int
                rank_33 = 23
                break
    
    if U_33 is not None:
        U_31, V_31, W_31 = _schoolbook(3, 3, 1)
        rank_31 = 9
        
        total_rank = rank_33 + rank_33 + rank_31
        U = np.zeros((9, total_rank))
        V = np.zeros((21, total_rank))
        W = np.zeros((21, total_rank))
        
        # Part 1: k=0,1,2
        U[:, 0:rank_33] = U_33
        for j in range(3):
            for k in range(3):
                V[j*7+k, 0:rank_33] = V_33[j*3+k, :]
        for k in range(3):
            for i in range(3):
                W[k*3+i, 0:rank_33] = W_33[k*3+i, :]
        
        # Part 2: k=3,4,5
        U[:, rank_33:2*rank_33] = U_33
        for j in range(3):
            for k in range(3):
                V[j*7+(k+3), rank_33:2*rank_33] = V_33[j*3+k, :]
        for k in range(3):
            for i in range(3):
                W[(k+3)*3+i, rank_33:2*rank_33] = W_33[k*3+i, :]
        
        # Part 3: k=6
        U[:, 2*rank_33:2*rank_33+rank_31] = U_31
        for j in range(3):
            V[j*7+6, 2*rank_33:2*rank_33+rank_31] = V_31[j*1+0, :]
        for i in range(3):
            W[6*3+i, 2*rank_33:2*rank_33+rank_31] = W_31[0*3+i, :]
        
        if _verify(U, V, W, 3, 3, 7):
            def to_list(arr):
                return [[float(x) for x in row] for row in arr]
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        
    # Fallback to schoolbook
    U, V, W = _schoolbook(3, 3, 7)
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

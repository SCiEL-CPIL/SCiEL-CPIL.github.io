import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-10, seed=0):
    """
    Find a rank decomposition for M_{n,m,p} using ALS.
    """
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for it in range(max_iters):
        if _time_left() < 10:
            return None
            
        # Solve for V
        M_V = np.zeros((n*m*p, m*p*rank))
        rhs_V = np.ones(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = idx
                    for t in range(rank):
                        coeff = U[i*m+j, t] * W[k*n+i, t]
                        col = (j*p+k) * rank + t
                        M_V[row, col] = coeff
                    idx += 1
        V_flat, _, _, _ = np.linalg.lstsq(M_V, rhs_V, rcond=None)
        V = V_flat.reshape(m*p, rank)
        
        # Solve for U
        M_U = np.zeros((n*m*p, n*m*rank))
        rhs_U = np.ones(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = idx
                    for t in range(rank):
                        coeff = V[j*p+k, t] * W[k*n+i, t]
                        col = (i*m+j) * rank + t
                        M_U[row, col] = coeff
                    idx += 1
        U_flat, _, _, _ = np.linalg.lstsq(M_U, rhs_U, rcond=None)
        U = U_flat.reshape(n*m, rank)
        
        # Solve for W
        M_W = np.zeros((n*m*p, p*n*rank))
        rhs_W = np.ones(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = idx
                    for t in range(rank):
                        coeff = U[i*m+j, t] * V[j*p+k, t]
                        col = (k*n+i) * rank + t
                        M_W[row, col] = coeff
                    idx += 1
        W_flat, _, _, _ = np.linalg.lstsq(M_W, rhs_W, rcond=None)
        W = W_flat.reshape(p*n, rank)
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(rank):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    err += (val - 1.0)**2
        err = np.sqrt(err / (n*m*p))
        
        if err < tol:
            return U, V, W
            
    return None

def _snap_to_half_int(arr):
    """Snap array to nearest half-integer if close enough."""
    # Multiply by 2, round, divide by 2
    snapped = np.round(arr * 2) / 2.0
    return snapped

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition with integer/half-integer entries."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(U.shape[1]):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                # Check if val is close to 1.0
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
    
    # Strategy:
    # 1. Try to find rank-23 decomposition for 3x3x3 using ALS with snapping.
    # 2. If found, combine two 3x3x3 and one 3x3x1. Total rank 49.
    # 3. If not, use schoolbook rank 63.
    
    U_33, V_33, W_33 = None, None, None
    rank_33 = 27
    
    # Try to find rank 23 for 3x3x3
    # Use multiple restarts
    for seed in range(20):
        if _time_left() < 20:
            break
        res = _als_decompose(3, 3, 3, 23, max_iters=100, tol=1e-9, seed=seed)
        if res is not None:
            U_cand, V_cand, W_cand = res
            # Snap to half-integers
            U_int = _snap_to_half_int(U_cand)
            V_int = _snap_to_half_int(V_cand)
            W_int = _snap_to_half_int(W_cand)
            
            if _verify(U_int, V_int, W_int, 3, 3, 3):
                U_33, V_33, W_33 = U_int, V_int, W_int
                rank_33 = 23
                break
    
    if U_33 is not None:
        # Construct 3x3x7 decomposition
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
        
        # Verify full decomposition
        if _verify(U, V, W, 3, 3, 7):
            def to_list(arr):
                # Convert to list of lists of floats (0.5, 1.0, etc.)
                return [[float(x) for x in row] for row in arr]
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        
    # Fallback to schoolbook
    U, V, W = _schoolbook(3, 3, 7)
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

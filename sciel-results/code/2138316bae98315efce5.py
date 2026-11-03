# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _polish(U, V, W, n, m, p, max_iters=100, deadline=None):
    """Polish the decomposition using least_squares."""
    def flatten(x):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten(x):
        size_U = U.size
        size_V = V.size
        U_ = x[:size_U].reshape(U.shape)
        V_ = x[size_U:size_U+size_V].reshape(V.shape)
        W_ = x[size_U+size_V:].reshape(W.shape)
        return U_, V_, W_
        
    def residuals(x):
        U_, V_, W_ = unflatten(x)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_[i*m+j, :] * V_[j*p+k, :] * W_[k*n+i, :])
                    res.append(s - 1.0)
        return np.array(res)
        
    x0 = flatten(x)
    try:
        result = least_squares(residuals, x0, method='trf', max_nfev=max_iters, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        U_, V_, W_ = unflatten(result.x)
        err = np.linalg.norm(result.fun)
        if err < 1e-8:
            return U_, V_, W_, err
    except:
        pass
    return None, None, None, float('inf')

def _als_3x3x3(rank, max_iters=500, seed=0, deadline=None):
    """Find rank-r decomposition for 3x3x3 multiplication tensor."""
    n = m = p = 3
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-8:
                break
                
    # Polish
    if best_err < 1e-4:
        U_p, V_p, W_p, err_p = _polish(best_U, best_V, best_W, n, m, p, max_iters=50, deadline=deadline)
        if U_p is not None and err_p < 1e-8:
            return U_p, V_p, W_p, err_p
            
    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, n*m*p
    
    # Try to find rank-23 decomposition for 3x3x3
    # If found, construct rank-46 for 3x3x6
    # If not, try higher ranks
    
    target_ranks_3x3 = [23, 24, 25, 26, 27]
    
    for r3 in target_ranks_3x3:
        if time.time() > deadline - 10.0:
            break
        found = False
        for seed in range(30): # Increased seeds
            if time.time() > deadline - 10.0:
                break
            U3, V3, W3, err = _als_3x3x3(r3, max_iters=200, seed=seed, deadline=deadline)
            if U3 is not None and err < 1e-6:
                # Construct 3x3x6 factors
                total_rank = 2 * r3
                U = np.zeros((n*m, total_rank))
                V = np.zeros((m*p, total_rank))
                W = np.zeros((p*n, total_rank))
                
                # Copy slice 0 (k=0,1,2)
                U[:, 0:r3] = U3
                for j in range(m):
                    for k in range(3):
                        V[j*p+k, 0:r3] = V3[j*3+k, 0:r3]
                for k in range(3):
                    for i in range(n):
                        W[k*n+i, 0:r3] = W3[k*3+i, 0:r3]
                        
                # Copy slice 1 (k=3,4,5)
                U[:, r3:total_rank] = U3
                for j in range(m):
                    for k in range(3):
                        V[j*p+(3+k), r3:total_rank] = V3[j*3+k, 0:r3]
                for k in range(3):
                    for i in range(n):
                        W[(3+k)*n+i, r3:total_rank] = W3[k*3+i, 0:r3]
                        
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    best_U, best_V, best_W, best_rank = U, V, W, total_rank
                    found = True
                    break
        if found:
            break
            
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

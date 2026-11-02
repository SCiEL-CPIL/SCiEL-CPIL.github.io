import time
import numpy as np
import itertools
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 15]

_START = time.time()
_DEADLINE = 115.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    tmp = np.einsum('uk,vk->uvk', U, V)
    result = np.einsum('uvk,wk->uvw', tmp, W)
    if np.max(np.abs(result - T)) > tol:
        return False
    return True

def _strassen_222():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=-1; V[1,0]=1; V[3,0]=1; W[2,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=-1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[2,3]=1; W[0,3]=-1
    U[0,4]=1; U[3,4]=1; V[3,4]=1; V[1,4]=-1; W[0,4]=1; W[3,4]=-1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=1; W[1,5]=1
    U[3,6]=1; U[2,6]=-1; V[1,6]=1; V[3,6]=1; W[1,6]=1; W[3,6]=1
    return U, V, W

def _rank3_221():
    U = np.zeros((4, 3))
    V = np.zeros((2, 3))
    W = np.zeros((2, 3))
    U[:,0]=[1,0,0,0]; V[:,0]=[1,0]; W[:,0]=[1,0]
    U[:,1]=[0,0,1,0]; V[:,1]=[0,1]; W[:,1]=[0,1]
    U[:,2]=[0,1,0,1]; V[:,2]=[1,1]; W[:,2]=[1,1]
    return U, V, W

def _build_223_rank10():
    U_s, V_s, W_s = _strassen_222()
    U_r, V_r, W_r = _rank3_221()
    U = np.zeros((4, 10))
    V = np.zeros((6, 10))
    W = np.zeros((6, 10))
    U[:, :7] = U_s
    for j in range(2):
        for kl in range(2):
            V[j * 2 + kl, :7] = V_s[j * 2 + kl] # Wait, V_s is 4x7 for 2x2x2. V indices are j*2+k.
            # In _strassen_222, V is 4x7. Indices 0..3 correspond to j*2+k for j=0,1, k=0,1.
            # So V_s[j*2+kl] is correct.
    for kl in range(2):
        for i in range(2):
            W[kl * 2 + i, :7] = W_s[kl * 2 + i]
            
    U[:, 7:] = U_r
    for j in range(2):
        V[j * 3 + 2, 7:] = V_r[j] # V_r is 2x3. j is 0,1.
    for i in range(2):
        W[2 * 2 + i, 7:] = W_r[i] # W_r is 2x3. i is 0,1.
    return U, V, W

def _build_2215_blockwise():
    n, m, p = 2, 2, 15
    U_223, V_223, W_223 = _build_223_rank10()
    r223 = U_223.shape[1]
    U = np.zeros((n * m, 5 * r223))
    V = np.zeros((m * p, 5 * r223))
    W = np.zeros((p * n, 5 * r223))
    for block in range(5):
        k0 = block * 3
        col = block * r223
        U[:, col:col+r223] = U_223
        for j in range(m):
            for kl in range(3):
                V[j * p + (k0 + kl), col:col+r223] = V_223[j * 3 + kl]
        for kl in range(3):
            for i in range(n):
                W[(k0 + kl) * n + i, col:col+r223] = W_223[kl * 2 + i]
    return U, V, W

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

def _residual(params, n, m, p, R):
    U = params[:R*4].reshape(R, 4).T
    V = params[R*4:R*4+R*30].reshape(R, 30).T
    W = params[R*4+R*30:].reshape(R, 30).T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    tmp = np.einsum('uk,vk->uvk', U, V)
    result = np.einsum('uvk,wk->uvw', tmp, W)
    return (result - T).flatten()

def _optimize_rank(R, init_U, init_V, init_W, n, m, p):
    # Flatten
    U_flat = init_U.flatten()
    V_flat = init_V.flatten()
    W_flat = init_W.flatten()
    params = np.concatenate([U_flat, V_flat, W_flat])
    
    try:
        res = least_squares(_residual, params, args=(n, m, p, R), method='lm', max_nfev=1000)
        if res.success or res.cost < 1e-10:
            opt_params = res.x
            U_opt = opt_params[:R*4].reshape(R, 4).T
            V_opt = opt_params[R*4:R*4+R*30].reshape(R, 30).T
            W_opt = opt_params[R*4+R*30:].reshape(R, 30).T
            return U_opt, V_opt, W_opt, res.cost
    except Exception:
        pass
    return None, None, None, float('inf')

def construct():
    n, m, p = DIMENSIONS
    
    if _remaining() < 5:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Get baseline rank 50
    U50, V50, W50 = _build_2215_blockwise()
    best_U, best_V, best_W = U50, V50, W50
    best_rank = 50
    
    # Try to reduce rank using LM
    # We try R = 49, 48, 47, 46, 45
    for R in [49, 48, 47, 46, 45]:
        if _remaining() < 10:
            break
        # Initialize with first R columns of the 50-rank solution
        init_U = U50[:, :R]
        init_V = V50[:, :R]
        init_W = W50[:, :R]
        
        U_opt, V_opt, W_opt, cost = _optimize_rank(R, init_U, init_V, init_W, n, m, p)
        
        if U_opt is not None and cost < 1e-8:
            # Check if we can round to half-integers
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if _verify(U_round, V_round, W_round, n, m, p, tol=1e-9):
                best_U, best_V, best_W = U_round, V_round, W_round
                best_rank = R
                break # Found lower rank, stop
            else:
                # If rounding fails, maybe keep float? But requirement is integer/half-integer.
                # If we can't satisfy the type constraint, we can't use this solution.
                # We continue searching for a lower rank or stick to 50.
                pass
                
    # If no lower rank found, return 50
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

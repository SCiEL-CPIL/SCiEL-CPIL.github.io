import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110

def _verify(U, V, W, tol=1e-10):
    T = np.einsum('ar,br,cr->abc', U, V, W)
    valid_mask = np.zeros((9, 9, 9), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                valid_mask[a, b, c] = True
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    invalid = T[~valid_mask]
    if np.max(np.abs(invalid)) > tol:
        return False
    return True

def _get_rank27():
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    return U, V, W

def _als_decompose(rank, max_iters=200, tol=1e-8, init_U=None, init_V=None, init_W=None):
    if init_U is not None:
        U = init_U.copy()
        V = init_V.copy()
        W = init_W.copy()
    else:
        U = np.random.randn(9, rank)
        V = np.random.randn(9, rank)
        W = np.random.randn(9, rank)
        
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0
                
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
            
        B_mat = np.zeros((81, rank))
        for r in range(rank):
            B_mat[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        M_mat = M.reshape(9, 81)
        try:
            U_new, _, _, _ = np.linalg.lstsq(B_mat.T, M_mat.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        M_mat2 = M.transpose(1, 0, 2).reshape(9, 81)
        B_mat2 = np.zeros((81, rank))
        for r in range(rank):
            B_mat2[:, r] = np.outer(U[:, r], W[:, r]).flatten()
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_mat2.T, M_mat2.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        M_mat3 = M.transpose(2, 0, 1).reshape(9, 81)
        B_mat3 = np.zeros((81, rank))
        for r in range(rank):
            B_mat3[:, r] = np.outer(U[:, r], V[:, r]).flatten()
        try:
            W_new, _, _, _ = np.linalg.lstsq(B_mat3.T, M_mat3.T, rcond=None)
            W = W_new.T
        except:
            pass
            
        T = np.einsum('ar,br,cr->abc', U, V, W)
        if np.linalg.norm(T - M) < tol:
            break
            
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _verify(U_r, V_r, W_r):
        return U_r, V_r, W_r
    return None

def construct():
    U27, V27, W27 = _get_rank27()
    
    for attempt in range(10):
        if time.time() > _DEADLINE - 10:
            break
        cols = np.random.choice(27, 21, replace=False)
        U_init = U27[:, cols]
        V_init = V27[:, cols]
        W_init = W27[:, cols]
        res = _als_decompose(21, init_U=U_init, init_V=V_init, init_W=W_init)
        if res:
            return {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}
            
    for attempt in range(5):
        if time.time() > _DEADLINE - 10:
            break
        res = _als_decompose(21)
        if res:
            return {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}

    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}

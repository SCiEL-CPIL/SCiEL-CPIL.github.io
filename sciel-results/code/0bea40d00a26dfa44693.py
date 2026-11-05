"""Rewritten program: attempts rank-16, falls back to rank-17."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-5):
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=tol)

def _rank17_decomposition():
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = n * m, m * p, p * n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_U(i, j, r, val): U[i * m + j, r] = val
    def set_V(j, k, r, val): V[j * p + k, r] = val
    def set_W(k, i, r, val): W[k * n + i, r] = val

    # Strassen 2x2x2
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
    
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
    
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
    
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
    
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)
    
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)

    # Direct 2x2x1
    set_U(0, 0, 7, 1); set_V(0, 2, 7, 1); set_W(2, 0, 7, 1)
    set_U(0, 1, 8, 1); set_V(1, 2, 8, 1); set_W(2, 0, 8, 1)
    set_U(1, 0, 9, 1); set_V(0, 2, 9, 1); set_W(2, 1, 9, 1)
    set_U(1, 1, 10, 1); set_V(1, 2, 10, 1); set_W(2, 1, 10, 1)

    # Direct 2x1x3
    set_U(0, 2, 11, 1); set_V(2, 0, 11, 1); set_W(0, 0, 11, 1)
    set_U(0, 2, 12, 1); set_V(2, 1, 12, 1); set_W(1, 0, 12, 1)
    set_U(0, 2, 13, 1); set_V(2, 2, 13, 1); set_W(2, 0, 13, 1)
    set_U(1, 2, 14, 1); set_V(2, 0, 14, 1); set_W(0, 1, 14, 1)
    set_U(1, 2, 15, 1); set_V(2, 1, 15, 1); set_W(1, 1, 15, 1)
    set_U(1, 2, 16, 1); set_V(2, 2, 16, 1); set_W(2, 1, 16, 1)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # Try rank 16
    def try_rank16():
        if time.time() > deadline - 20:
            return None
            
        # Try a few random initializations
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            U = np.random.randn(n_a, 16)
            V = np.random.randn(n_b, 16)
            W = np.random.randn(n_c, 16)
            
            def residual(x):
                U_f = x[:n_a*16].reshape(n_a, 16)
                V_f = x[n_a*16:n_a*16+n_b*16].reshape(n_b, 16)
                W_f = x[n_a*16+n_b*16:].reshape(n_c, 16)
                T_calc = np.zeros((n_a, n_b, n_c))
                for r in range(16):
                    T_calc += np.einsum('a,b,c->abc', U_f[:, r], V_f[:, r], W_f[:, r])
                return (T_calc - T).flatten()
                
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            try:
                res = least_squares(residual, x0, method='lm', max_nfev=5000)
                if res.cost < 1e-8:
                    U_f = res.x[:n_a*16].reshape(n_a, 16)
                    V_f = res.x[n_a*16:n_a*16+n_b*16].reshape(n_b, 16)
                    W_f = res.x[n_a*16+n_b*16:].reshape(n_c, 16)
                    if _verify_decomposition(U_f, V_f, W_f, T):
                        return U_f, V_f, W_f
            except Exception:
                pass
        return None

    U16, V16, W16 = try_rank16()
    if U16 is not None:
        return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
        
    # Fallback to rank 17
    U17, V17, W17 = _rank17_decomposition()
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

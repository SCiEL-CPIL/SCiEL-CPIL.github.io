import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(result - T)) < tol

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _optimize_rank(r, n, m, p, max_iter=1000, seed=0, tol=1e-9):
    np.random.seed(seed)
    T = _build_tensor(n, m, p)
    
    # Better initialization: use random but with appropriate scale
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    def loss(params):
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r:n*m*r+m*p*r]
        W_flat = params[n*m*r+m*p*r:]
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((result - T)**2)
    
    def grad(params):
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r:n*m*r+m*p*r]
        W_flat = params[n*m*r+m*p*r:]
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        result = np.einsum('ar,br,cr->abc', U, V, W)
        err = result - T
        gU = 2*np.einsum('abc,br,cr->ar', err, V, W)
        gV = 2*np.einsum('abc,ar,cr->br', err, U, W)
        gW = 2*np.einsum('abc,ar,br->cr', err, U, V)
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
    initial = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    res = minimize(loss, initial, jac=grad, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
    
    final_params = res.x
    U_opt = final_params[:n*m*r].reshape(n*m, r)
    V_opt = final_params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
    W_opt = final_params[n*m*r+m*p*r:].reshape(p*n, r)
    
    # Try snapping to half-integers with a local search
    U_r = np.round(U_opt * 2) / 2
    V_r = np.round(V_opt * 2) / 2
    W_r = np.round(W_opt * 2) / 2
    
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # If direct snapping fails, try a local search around the snapped solution
    # Try small perturbations
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for delta in [0.5, 0.25, 0.125]:
        for i in range(n*m):
            for j in range(r):
                if abs(U_r[i,j]) < 10:  # Only try if not too large
                    orig = U_r[i,j]
                    for dv in [-delta, delta]:
                        U_r[i,j] = orig + dv
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol):
                            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                            return best_U.tolist(), best_V.tolist(), best_W.tolist()
                        U_r[i,j] = orig
        
        for i in range(m*p):
            for j in range(r):
                if abs(V_r[i,j]) < 10:
                    orig = V_r[i,j]
                    for dv in [-delta, delta]:
                        V_r[i,j] = orig + dv
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol):
                            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                            return best_U.tolist(), best_V.tolist(), W_r.tolist()
                        V_r[i,j] = orig
        
        for i in range(p*n):
            for j in range(r):
                if abs(W_r[i,j]) < 10:
                    orig = W_r[i,j]
                    for dv in [-delta, delta]:
                        W_r[i,j] = orig + dv
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol):
                            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                            return best_U.tolist(), best_V.tolist(), W_r.tolist()
                        W_r[i,j] = orig
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from a reasonable lower bound to schoolbook
    # The rank is at least max(n*m, m*p, p*n) is false, but at least max(n, m, p) * min(n, m, p) or something
    # Let's try from 15 to 35
    for r in range(15, 36):
        if time.time() > deadline - 30:
            break
        found = False
        for attempt in range(30):
            if time.time() > deadline - 30:
                break
            res = _optimize_rank(r, n, m, p, max_iter=500, seed=attempt)
            if res:
                best = {"U": res[0], "V": res[1], "W": res[2]}
                best_rank = r
                found = True
                break
        if found:
            # Found a solution at rank r, return immediately since we're going from low to high
            return best
    
    return best

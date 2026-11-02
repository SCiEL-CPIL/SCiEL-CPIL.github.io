"""Find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    u_idxs = np.array([i*m + j for i in range(n) for j in range(m) for k in range(p)])
    v_idxs = np.array([j*p + k for i in range(n) for j in range(m) for k in range(p)])
    w_idxs = np.array([k*n + i for i in range(n) for j in range(m) for k in range(p)])
    
    def compute_errors(U, V, W):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        terms = U[u_idxs] * V[v_idxs] * W[w_idxs]
        return np.sum(terms, axis=1) - 1.0
    
    def max_abs_error(U, V, W):
        return np.max(np.abs(compute_errors(U, V, W)))
    
    def verify(U, V, W, tol=1e-9):
        return max_abs_error(U, V, W) < tol
    
    def is_half_integer(U, V, W, tol=1e-9):
        for mat in [np.asarray(U, dtype=float), np.asarray(V, dtype=float), np.asarray(W, dtype=float)]:
            if not np.allclose(2*mat, np.round(2*mat), atol=tol):
                return False
        return True
    
    def round_half(U, V, W):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        return np.round(2*U)/2.0, np.round(2*V)/2.0, np.round(2*W)/2.0
    
    def try_optimize(rank, num_trials=40, max_iters=100):
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            lr = 0.03
            best_err = float('inf')
            U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                errs = compute_errors(U, V, W)
                cur_err = np.max(np.abs(errs))
                
                if cur_err < 1e-10:
                    break
                
                if cur_err < best_err:
                    best_err = cur_err
                    U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for c in range(len(u_idxs)):
                    u_i, v_i, w_i = int(u_idxs[c]), int(v_idxs[c]), int(w_idxs[c])
                    e = errs[c]
                    grad_U[u_i] += 2 * e * V[v_i] * W[w_i]
                    grad_V[v_i] += 2 * e * U[u_i] * W[w_i]
                    grad_W[w_i] += 2 * e * U[u_i] * V[v_i]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if it % 10 == 9:
                    new_err = max_abs_error(U, V, W)
                    if new_err > cur_err * 1.3:
                        lr *= 0.5
                    elif new_err < cur_err * 0.7:
                        lr = min(lr * 1.1, 0.05)
            
            err = max_abs_error(U_best, V_best, W_best)
            if err < 1e-8:
                U_r, V_r, W_r = round_half(U_best, V_best, W_best)
                if verify(U_r, V_r, W_r) and is_half_integer(U_r, V_r, W_r):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                if verify(U_best, V_best, W_best):
                    return U_best.tolist(), V_best.tolist(), W_best.tolist()
        
        return None
    
    for rank in range(18, 55):
        if time.time() > deadline - 10:
            break
        result = try_optimize(rank, num_trials=30)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Schoolbook fallback
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

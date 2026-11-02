import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    target_int = (target * 8).astype(np.int64)

    def compute_error_int(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        diff = T_approx - target_int
        return np.sum(diff**2)

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        U_int = np.round(U * 2.0)
        V_int = np.round(V * 2.0)
        W_int = np.round(W * 2.0)
        
        # Check if they are close to integers
        if not np.allclose(U * 2.0, U_int, atol=1e-6): return None
        if not np.allclose(V * 2.0, V_int, atol=1e-6): return None
        if not np.allclose(W * 2.0, W_int, atol=1e-6): return None
        
        # Check exact decomposition with integers
        U_f = U_int.astype(np.float64) / 2.0
        V_f = V_int.astype(np.float64) / 2.0
        W_f = W_int.astype(np.float64) / 2.0
        
        T_approx = np.einsum('ak,bk,ck->abc', U_f, V_f, W_f)
        if np.allclose(T_approx, target, atol=1e-9):
            return U_f, V_f, W_f
        return None

    def polish_integer(U_init, V_init, W_init, rank, max_steps=2000):
        # Convert to int representation (x2)
        U = np.round(U_init * 2.0).astype(np.int64)
        V = np.round(V_init * 2.0).astype(np.int64)
        W = np.round(W_init * 2.0).astype(np.int64)
        
        curr_err = compute_error_int(U, V, W)
        
        for _ in range(max_steps):
            if time.time() > deadline - 2:
                break
            mat_idx = np.random.choice(3)
            if mat_idx == 0:
                mat = U
                shape = (nm, rank)
            elif mat_idx == 1:
                mat = V
                shape = (mp, rank)
            else:
                mat = W
                shape = (pn, rank)
                
            idx1 = np.random.randint(shape[0])
            idx2 = np.random.randint(shape[1])
            delta = np.random.choice([-1, 1])
            
            old_val = mat[idx1, idx2]
            mat[idx1, idx2] = old_val + delta
            new_err = compute_error_int(U, V, W)
            
            if new_err < curr_err:
                curr_err = new_err
                if curr_err == 0:
                    return U.astype(np.float64)/2.0, V.astype(np.float64)/2.0, W.astype(np.float64)/2.0
            else:
                mat[idx1, idx2] = old_val
        return None

    def try_continuous_rank(rank, max_restarts=10):
        if time.time() > deadline - 10:
            return None
            
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U = x[:nm*rank].reshape(nm, rank)
                V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W = x[nm*rank+mp*rank:].reshape(pn, rank)
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                return np.sum(diff**2)
            
            # Optimize
            try:
                result = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
            except Exception:
                continue
            
            if result.fun < 1e-6:
                x_opt = result.x
                U_opt = x_opt[:nm*rank].reshape(nm, rank)
                V_opt = x_opt[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_opt = x_opt[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Snap and verify
                sol = verify_solution(U_opt, V_opt, W_opt)
                if sol is not None:
                    return sol
                
                # If not exact, try to polish
                sol = polish_integer(U_opt, V_opt, W_opt, rank, max_steps=500)
                if sol is not None:
                    return sol

        return None

    # Try ranks from low to high
    # Known rank for 2x3x4 is 18
    for rank in range(18, 25):
        if time.time() > deadline - 20:
            break
            
        res = try_continuous_rank(rank, max_restarts=5)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

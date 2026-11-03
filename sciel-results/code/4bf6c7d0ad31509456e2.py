import numpy as np
import time
from scipy.optimize import minimize

def _verify_fast(U, V, W, n, m, p):
    """
    Fast verification using NumPy.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    R = U.shape[1]
    # Reshape to 3D tensors
    # U[i, j, r], V[j, k, r], W[k, i, r]
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    
    # Compute the tensor product sum over r
    # T[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
    # This is a bit tricky to do in one shot without loops or einsum.
    # Let's use einsum.
    # U_3d[i, j, r] * W_3d[k, i, r] -> sum over i? No.
    # We want T[i, j, k] = sum_r (U[i, j, r] * W[k, i, r]) * V[j, k, r]
    # Let A[i, j, k, r] = U[i, j, r] * W[k, i, r]
    # Then T[i, j, k] = sum_r A[i, j, k, r] * V[j, k, r]
    
    # Construct A
    # U_3d: (n, m, R)
    # W_3d: (p, n, R)
    # We need to broadcast W to (n, m, p, R)?
    # U_3d[i, j, r] depends on i, j.
    # W_3d[k, i, r] depends on k, i.
    # So A[i, j, k, r] = U_3d[i, j, r] * W_3d[k, i, r]
    
    # U_3d[:, :, np.newaxis, :] -> (n, m, 1, R)
    # W_3d[np.newaxis, :, :, :] -> (1, n, p, R) ... wait, W_3d is (p, n, R)
    # W_3d.T is (n, p, R).
    # We want W[k, i, r] for fixed i, j, k, r.
    # W_3d is indexed by [k, i, r].
    # So we need to align indices.
    
    # Let's just use a loop over i, j, k but vectorize over r.
    # Or better, use einsum with explicit indices.
    # U_3d: i, j, r
    # V_3d: j, k, r
    # W_3d: k, i, r
    
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    
    # np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Target is 1 everywhere
    errors = T - 1.0
    return np.max(np.abs(errors))

def _local_search(U, V, W, n, m, p, max_iters=1000):
    """
    Try to improve the solution by flipping individual entries to nearest half-integers.
    Assumes U, V, W are already close to half-integers.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Round to nearest 0.5 first
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    current_err = _verify_fast(U, V, W, n, m, p)
    
    for _ in range(max_iters):
        improved = False
        # Try flipping each entry in U, V, W
        # This is O(n*m*R + m*p*R + p*n*R) evaluations per iteration, which might be slow.
        # We can sample a few entries.
        
        # For efficiency, let's just try a few random flips or the most promising ones.
        # Actually, let's just try flipping one entry at a time in a random order.
        
        # Pick a random matrix and entry
        mat = np.random.choice(3)
        if mat == 0:
            idx = (np.random.randint(n*m), np.random.randint(U.shape[1]))
            old_val = U[idx]
            # Try neighbors: old_val - 0.5, old_val + 0.5
            for delta in [-0.5, 0.5]:
                new_val = old_val + delta
                if new_val < -10 or new_val > 10: continue # Bounds
                U[idx] = new_val
                err = _verify_fast(U, V, W, n, m, p)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    U[idx] = old_val
        elif mat == 1:
            idx = (np.random.randint(m*p), np.random.randint(V.shape[1]))
            old_val = V[idx]
            for delta in [-0.5, 0.5]:
                new_val = old_val + delta
                if new_val < -10 or new_val > 10: continue
                V[idx] = new_val
                err = _verify_fast(U, V, W, n, m, p)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    V[idx] = old_val
        else:
            idx = (np.random.randint(p*n), np.random.randint(W.shape[1]))
            old_val = W[idx]
            for delta in [-0.5, 0.5]:
                new_val = old_val + delta
                if new_val < -10 or new_val > 10: continue
                W[idx] = new_val
                err = _verify_fast(U, V, W, n, m, p)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    W[idx] = old_val
                    
        if not improved:
            break
            
    if current_err < 1e-9:
        return U, V, W
    return None

def _try_rank(R, n, m, p, timeout=5.0):
    """
    Attempts to find a decomposition of rank R.
    """
    start_time = time.time()
    
    # Try a few random initializations
    for attempt in range(5):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(42 + attempt)
        
        # Initialize with small random values
        U = np.random.rand(n * m, R) * 0.5 - 0.25
        V = np.random.rand(m * p, R) * 0.5 - 0.25
        W = np.random.rand(p * n, R) * 0.5 - 0.25
        
        def flatten(params):
            u = params[:n*m*R].reshape(n*m, R)
            v = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            w = params[n*m*R + m*p*R:].reshape(p*n, R)
            return u, v, w

        def objective(params):
            if time.time() - start_time > timeout:
                return float('inf')
            u, v, w = flatten(params)
            # Use fast verification for objective?
            # _verify_fast returns max error, we want sum of squares or similar.
            # Let's compute the sum of squared errors.
            U_3d = u.reshape(n, m, R)
            V_3d = v.reshape(m, p, R)
            W_3d = w.reshape(p, n, R)
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            err = np.sum((T - 1.0) ** 2)
            return err

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        except Exception:
            continue
        
        if time.time() - start_time > timeout:
            break
            
        if res.fun < 1e-4: # Reasonably close
            u, v, w = flatten(res.x)
            # Local search
            result = _local_search(u, v, w, n, m, p, max_iters=200)
            if result is not None:
                return result

    return None

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Try to find rank < 48
    # Start from 47 down to 30
    for R in range(47, 29, -1):
        if time.time() - start_time > deadline:
            break
        result = _try_rank(R, n, m, p, timeout=5.0)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    rank = n * m * p
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][col] = 1
                V_list[j * p + k][col] = 1
                W_list[k * n + i][col] = 1
                col += 1
                
    return {"U": U_list, "V": V_list, "W": W_list}

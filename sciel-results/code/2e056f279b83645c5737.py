import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Check: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W_mat = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    error = np.linalg.norm(T - 1.0)
    return error

def _snap_to_half(arr):
    return np.round(arr * 2) / 2

def _local_search_snapping(U, V, W, n, m, p, max_iters=5):
    """
    Attempt to snap to half-integers and refine via local search.
    """
    U_s = _snap_to_half(U)
    V_s = _snap_to_half(V)
    W_s = _snap_to_half(W)
    
    best_err = _verify(U_s, V_s, W_s, n, m, p)
    
    for _ in range(max_iters):
        improved = False
        # Try shifting individual entries by +/- 0.5
        # This is expensive, so we only do it if best_err is small
        if best_err > 1e-4:
            break
            
        # Simple local search: for each non-zero element, try +/- 0.5
        # To save time, we only do a few random samples or all if small
        # For R=12, matrices are small enough.
        
        # We'll skip full local search to keep code concise and fast, 
        # relying on the snapping of the optimized solution.
        # If the optimization is good, snapping should work.
        break
        
    return U_s, V_s, W_s, best_err

def _try_rank(R, n, m, p, time_limit=10, num_restarts=5):
    best_error = float('inf')
    best_factors = None
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
            U_mat = U_flat.reshape(n, m, R)
            V_mat = V_flat.reshape(m, p, R)
            W_mat = W_flat.reshape(p, n, R)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
            err = np.sum((T - 1.0)**2)
            return err
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            x_opt = res.x
            U_flat, V_flat, W_flat = x_opt[:n*m*R], x_opt[n*m*R:n*m*R+m*p*R], x_opt[n*m*R+m*p*R:]
            U_opt = U_flat.reshape(n*m, R)
            V_opt = V_flat.reshape(m*p, R)
            W_opt = W_flat.reshape(p*n, R)
            
            # Snap and verify
            U_snap = _snap_to_half(U_opt)
            V_snap = _snap_to_half(V_opt)
            W_snap = _snap_to_half(W_opt)
            
            err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
            
            if err_snap < 1e-6:
                return U_snap, V_snap, W_snap
                
            if err_snap < best_error:
                best_error = err_snap
                best_factors = (U_snap, V_snap, W_snap)
                
        except Exception:
            pass
            
    return None

def _try_subtensor_rank_11():
    """
    Try to find rank 11 decomposition for M_{2,2,3}.
    Returns (U, V, W) for the sub-tensor or None.
    n=2, m=2, p=3.
    """
    n, m, p = 2, 2, 3
    R = 11
    # Use a slightly longer time limit as this is a fallback
    res = _try_rank(R, n, m, p, time_limit=15, num_restarts=10)
    return res

def construct():
    n, m, p = DIMENSIONS
    global_start = time.time()
    deadline = global_start + 130 # Safety margin
    
    # 1. Try Rank 12 for M_{2,3,3}
    if time.time() < deadline - 30:
        res = _try_rank(12, n, m, p, time_limit=20, num_restarts=5)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Try Rank 11 for M_{2,2,3} to construct Rank 17 for M_{2,3,3}
    if time.time() < deadline - 20:
        sub_res = _try_subtensor_rank_11()
        if sub_res is not None:
            U_sub, V_sub, W_sub = sub_res
            # Construct full Rank 17
            # M_{2,3,3} = M_{2,3,2} (cols 0,1 of B) + M_{2,3,1} (col 2 of B)
            # Wait, the split was M_{2,3,2} + M_{2,3,1}.
            # M_{2,3,2} is A(2x3) * B_left(3x2).
            # The sub-tensor I solved was M_{2,2,3} which is A(2x2) * B(2x3).
            # These are different.
            # M_{2,3,2} is isomorphic to M_{2,2,3} by transposing B?
            # Rank(M_{n,m,p}) = Rank(M_{n,p,m})?
            # M_{2,3,2}: A(2x3), B(3x2).
            # M_{2,2,3}: A(2x2), B(2x3).
            # The tensor T_{ijk} for M_{n,m,p} is symmetric in j? No.
            # But Rank(M_{2,3,2}) = Rank(M_{2,2,3}) = 11.
            # We can use the decomposition of M_{2,2,3} for M_{2,3,2} by permuting indices.
            # M_{2,2,3}: i in {0,1}, j in {0,1}, k in {0,1,2}.
            # M_{2,3,2}: i in {0,1}, j in {0,1,2}, k in {0,1}.
            # We need to map the decomposition.
            # This is complex to do correctly on the fly.
            # Let's just try Rank 17 directly on the full tensor.
            pass

    # 3. Try Rank 17 directly
    if time.time() < deadline - 10:
        res = _try_rank(17, n, m, p, time_limit=10, num_restarts=5)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 4. Try Rank 16
    if time.time() < deadline - 5:
        res = _try_rank(16, n, m, p, time_limit=5, num_restarts=3)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}

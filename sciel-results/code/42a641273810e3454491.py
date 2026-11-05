import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def _check_decomposition(U, V, W, n, m, p, rank, tol=1e-8):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Construct decomposed tensor
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
    return np.allclose(T_decomp, T, atol=tol)

def _solve_subproblem(n, m, p, target_rank, max_time=10.0):
    """
    Try to find a decomposition of rank target_rank for n x m x p multiplication.
    Uses continuous optimization followed by snapping to half-integers.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    T_flat = T.flatten()
    
    def compute_error_flat(params):
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank:]
        
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        diff = T_decomp.flatten() - T_flat
        return np.dot(diff, diff)

    def compute_error_and_params(params):
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank:]
        
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        diff = T_decomp.flatten() - T_flat
        err = np.dot(diff, diff)
        return err, U, V, W

    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    # Number of restarts based on time
    # For 3x3x3, rank 23, we need more restarts
    num_restarts = 100 if target_rank <= 24 else 50
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(compute_error_flat, params, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
        except:
            continue
            
        err, U_opt, V_opt, W_opt = compute_error_and_params(res.x)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
        if err < 1e-12:
            # Found a continuous solution, try to snap to half-integers
            # Strategy: Round to nearest half-integer, then local discrete search
            U_snap = np.round(U_opt * 2) / 2.0
            V_snap = np.round(V_opt * 2) / 2.0
            W_snap = np.round(W_opt * 2) / 2.0
            
            if _check_decomposition(U_snap, V_snap, W_snap, n, m, p, target_rank):
                return U_snap, V_snap, W_snap
            
            # If direct snap fails, try a local discrete search
            # We focus on the entries that are far from integers/half-integers
            # or just do a quick random local search
            U_d = U_snap.copy()
            V_d = V_snap.copy()
            W_d = W_snap.copy()
            
            def compute_error_discrete(U, V, W):
                T_decomp = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                return np.linalg.norm(T_decomp - T)
            
            current_err = compute_error_discrete(U_d, V_d, W_d)
            
            # Local search: try changing single entries to nearby half-integers
            values = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            
            for step in range(200):
                if time.time() > deadline:
                    break
                if current_err < 1e-8:
                    return U_d, V_d, W_d
                    
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    arr = U_d
                    rows, cols = n*m, target_rank
                elif mat_choice == 'V':
                    arr = V_d
                    rows, cols = m*p, target_rank
                else:
                    arr = W_d
                    rows, cols = p*n, target_rank
                
                row = random.randint(0, rows - 1)
                col = random.randint(0, cols - 1)
                
                old_val = arr[row, col]
                # Pick a new value close to the old one
                new_val = random.choice(values)
                
                if new_val == old_val:
                    continue
                
                arr[row, col] = new_val
                new_err = compute_error_discrete(U_d, V_d, W_d)
                
                if new_err < current_err:
                    current_err = new_err
                else:
                    arr[row, col] = old_val

            if current_err < 1e-8:
                return U_d, V_d, W_d

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    
    # Strategy:
    # 1. Try to solve 3x3x3 with rank 23 (Bini). If fail, try 24, 25, 26. Else 27.
    # 2. Solve 3x3x2 with schoolbook rank 18. (Searching for lower is risky/slow)
    
    res_333 = None
    rank_333 = 27
    # Prioritize finding rank 23
    for r in [23, 24, 25, 26]:
        if time.time() > deadline - 20:
            break
        res_333 = _solve_subproblem(3, 3, 3, r, max_time=10.0)
        if res_333:
            rank_333 = r
            break
            
    if not res_333:
        # Fall back to schoolbook for 3x3x3
        rank_333 = 27
        U_333 = np.zeros((9, rank_333))
        V_333 = np.zeros((9, rank_333))
        W_333 = np.zeros((9, rank_333))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_333[i*3+j, col] = 1
                    V_333[j*3+k, col] = 1
                    W_333[k*3+i, col] = 1
                    col += 1
        res_333 = (U_333, V_333, W_333)

    # 3x3x2 block: Use schoolbook rank 18. 
    # We could try to search for rank < 18, but it's unlikely to succeed quickly.
    # To be safe and fast, we use schoolbook.
    rank_332 = 18
    U_332 = np.zeros((9, rank_332))
    V_332 = np.zeros((6, rank_332))
    W_332 = np.zeros((6, rank_332))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U_332[i*3+j, col] = 1
                V_332[j*2+k, col] = 1
                W_332[k*3+i, col] = 1
                col += 1
    res_332 = (U_332, V_332, W_332)
    
    U_333, V_333, W_333 = res_333
    
    total_rank = rank_333 + rank_332
    
    # Construct combined matrices
    U_total = np.zeros((9, total_rank))
    V_total = np.zeros((15, total_rank))
    W_total = np.zeros((15, total_rank))
    
    # Part 1: 3x3x3 (k=0,1,2)
    U_total[:, :rank_333] = U_333
    for j in range(3):
        for k in range(3):
            V_total[j*5+k, :rank_333] = V_333[j*3+k, :rank_333]
    W_total[:9, :rank_333] = W_333
    
    # Part 2: 3x3x2 (k=3,4)
    U_total[:, rank_333:] = U_332
    for j in range(3):
        for kp in range(2):
            k = 3 + kp
            V_total[j*5+k, rank_333:] = V_332[j*2+kp, :]
    for kp in range(2):
        k = 3 + kp
        W_total[k*3:(k+1)*3, rank_333:] = W_332[kp*3:(kp+1)*3, :]
        
    # Convert to lists of lists
    U_best = [[float(x) for x in row] for row in U_total]
    V_best = [[float(x) for x in row] for row in V_total]
    W_best = [[float(x) for x in row] for row in W_total]
    
    # Final verification
    if not _check_decomposition(U_best, V_best, W_best, n, m, p, total_rank):
        # Fall back to full schoolbook
        schoolbook_rank = n * m * p
        U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
        V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
        W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j][column] = 1
                    V_fallback[j * p + k][column] = 1
                    W_fallback[k * n + i][column] = 1
                    column += 1
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
        
    return {"U": U_best, "V": V_best, "W": W_best}

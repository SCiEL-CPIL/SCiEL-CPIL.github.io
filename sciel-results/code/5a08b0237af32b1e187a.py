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
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
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
                
    # Flatten the tensor for easier objective calculation
    T_flat = T.flatten()
    
    def compute_error_flat(params):
        """Compute squared error given flattened parameters."""
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
        """Compute error and return U, V, W."""
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
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Symmetry breaking: fix one entry to avoid trivial zero solution scaling
        # Actually, random start is usually fine for non-zero targets
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B for optimization
        try:
            res = minimize(compute_error_flat, params, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15})
        except:
            continue
            
        err, U_opt, V_opt, W_opt = compute_error_and_params(res.x)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
        if err < 1e-10:
            # Found a continuous solution, try to snap to half-integers
            U_snap = np.round(U_opt * 2) / 2.0
            V_snap = np.round(V_opt * 2) / 2.0
            W_snap = np.round(W_opt * 2) / 2.0
            
            # Check snapped solution
            if _check_decomposition(U_snap, V_snap, W_snap, n, m, p, target_rank):
                return U_snap, V_snap, W_snap
            else:
                # If snapping fails, maybe we can do a local discrete search around the snap
                # For now, return the continuous one if it's extremely close? 
                # The prompt requires integer/half-integer entries.
                # So we MUST return half-integers.
                # If the continuous solution is valid but not half-integer, it's useless.
                # We should try to find a half-integer solution.
                pass

    # If we found a good continuous solution but couldn't snap it,
    # try a local discrete search around the best continuous solution
    if best_U is not None and best_err < 1.0:
        U_start = np.round(best_U * 2) / 2.0
        V_start = np.round(best_V * 2) / 2.0
        W_start = np.round(best_W * 2) / 2.0
        
        values = [-1, -0.5, 0, 0.5, 1]
        
        def compute_error_discrete(U, V, W):
            T_decomp = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            return np.linalg.norm(T_decomp - T)
        
        err = compute_error_discrete(U_start, V_start, W_start)
        best_U_d, best_V_d, best_W_d = U_start.copy(), V_start.copy(), W_start.copy()
        best_err_d = err
        
        # Simple local search
        for _ in range(1000):
            if time.time() > deadline:
                break
            if best_err_d < 1e-8:
                break
                
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                arr = best_U_d
                rows, cols = n*m, target_rank
            elif mat_choice == 'V':
                arr = best_V_d
                rows, cols = m*p, target_rank
            else:
                arr = best_W_d
                rows, cols = p*n, target_rank
                
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            
            old_val = arr[row, col]
            new_val = random.choice(values)
            
            if new_val == old_val:
                continue
                
            arr[row, col] = new_val
            new_err = compute_error_discrete(best_U_d, best_V_d, best_W_d)
            
            if new_err < best_err_d:
                best_err_d = new_err
            else:
                arr[row, col] = old_val
                
            if best_err_d < 1e-8:
                return best_U_d, best_V_d, best_W_d
                
        if best_err_d < 1e-8:
            return best_U_d, best_V_d, best_W_d

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    p1 = 3
    p2 = 2

    # Solve 3x3x3
    res_333 = None
    rank_333 = 27
    # Try ranks lower than schoolbook (27)
    for r in [23, 24, 25, 26]:
        if time.time() > deadline - 30:
            break
        res_333 = _solve_subproblem(3, 3, p1, r, max_time=15.0)
        if res_333:
            rank_333 = r
            break
            
    if not res_333:
        # Fall back to schoolbook
        U_333 = [[0]*rank_333 for _ in range(9)]
        V_333 = [[0]*rank_333 for _ in range(9)]
        W_333 = [[0]*rank_333 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(p1):
                    U_333[i*3+j][col] = 1
                    V_333[j*p1+k][col] = 1
                    W_333[k*3+i][col] = 1
                    col += 1
        res_333 = (np.array(U_333), np.array(V_333), np.array(W_333))
        rank_333 = 27

    # Solve 3x3x2
    res_332 = None
    rank_332 = 18
    # Try ranks lower than schoolbook (18)
    for r in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
        if time.time() > deadline - 15:
            break
        res_332 = _solve_subproblem(3, 3, p2, r, max_time=5.0)
        if res_332:
            rank_332 = r
            break
            
    if not res_332:
        # Fall back to schoolbook
        U_332 = [[0]*rank_332 for _ in range(9)]
        V_332 = [[0]*rank_332 for _ in range(6)]
        W_332 = [[0]*rank_332 for _ in range(6)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(p2):
                    U_332[i*3+j][col] = 1
                    V_332[j*p2+k][col] = 1
                    W_332[k*3+i][col] = 1
                    col += 1
        res_332 = (np.array(U_332), np.array(V_332), np.array(W_332))
        rank_332 = 18

    U_333, V_333, W_333 = res_333
    U_332, V_332, W_332 = res_332
    
    total_rank = rank_333 + rank_332
    
    # Construct combined matrices
    U_total = np.zeros((9, total_rank))
    V_total = np.zeros((15, total_rank))
    W_total = np.zeros((15, total_rank))
    
    # Part 1: 3x3x3 (k=0,1,2)
    U_total[:, :rank_333] = U_333
    for j in range(3):
        for k in range(p1):
            V_total[j*5+k, :rank_333] = V_333[j*3+k, :rank_333]
    W_total[:9, :rank_333] = W_333
    
    # Part 2: 3x3x2 (k=3,4)
    U_total[:, rank_333:] = U_332
    for j in range(3):
        for kp in range(p2):
            k = p1 + kp
            V_total[j*5+k, rank_333:] = V_332[j*2+kp, :]
    for kp in range(p2):
        k = p1 + kp
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

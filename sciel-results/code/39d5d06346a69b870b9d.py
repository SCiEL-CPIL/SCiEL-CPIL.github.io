import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target equation: sum_r U_ij^r V_jk^r W_ki^r = 1
    # Using integers: U = u/2, V = v/2, W = w/2
    # sum_r (u/2)(v/2)(w/2) = 1 => sum_r u v w = 8
    
    target = 8
    
    def verify_int(u, v, w):
        # u: (n*m, R), v: (m*p, R), w: (p*n, R)
        # Check if sum_r u[i*m+j, r] * v[j*p+k, r] * w[k*n+i, r] == 8 for all i,j,k
        R = u.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += u[i*m+j, r] * v[j*p+k, r] * w[k*n+i, r]
                    if s != target:
                        return False
        return True

    def compute_residuals(u, v, w):
        # Returns a 3D array of residuals E[i,j,k] = target - sum
        R = u.shape[1]
        E = np.zeros((n, m, p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += u[i*m+j, r] * v[j*p+k, r] * w[k*n+i, r]
                    E[i,j,k] = target - s
        return E

    def total_error(E):
        return np.sum(E * E)

    def local_search(R, time_budget):
        start_time = time.time()
        best_error = float('inf')
        best_u, best_v, best_w = None, None, None
        
        # We will try a few random initializations
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break
                
            # Initialize with small random integers
            # Values in {-3, -2, -1, 0, 1, 2, 3}
            u = np.random.randint(-3, 4, size=(n*m, R))
            v = np.random.randint(-3, 4, size=(m*p, R))
            w = np.random.randint(-3, 4, size=(p*n, R))
            
            E = compute_residuals(u, v, w)
            curr_error = total_error(E)
            
            if curr_error == 0:
                return u, v, w
                
            if curr_error < best_error:
                best_error = curr_error
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            # Local search loop
            iter_limit = 10000
            for it in range(iter_limit):
                if time.time() - start_time > time_budget:
                    break
                
                # Pick a random equation (i, j, k) with non-zero residual
                non_zero = np.nonzero(E)
                if len(non_zero[0]) == 0:
                    return u, v, w
                
                idx = random.randint(0, len(non_zero[0]) - 1)
                i, j, k = non_zero[0][idx], non_zero[1][idx], non_zero[2][idx]
                err = E[i, j, k]
                
                # Try to fix this error by modifying one variable
                # The terms involved are u[i*m+j, r], v[j*p+k, r], w[k*n+i, r] for all r
                # We pick a random r and one of the three variables to change
                r = random.randint(0, R - 1)
                var_choice = random.randint(0, 2) # 0: u, 1: v, 2: w
                
                # Calculate the "gain" factor for changing the variable
                # If we change u[i*m+j, r] by delta, the sum changes by delta * v[j*p+k, r] * w[k*n+i, r]
                # We want the new sum to be closer to target.
                # Current sum S = target - err.
                # New sum S' = S + delta * factor.
                # We want S' to be closer to target, i.e., |S' - target| < |S - target| => |err - delta*factor| < |err|
                
                factor = 1
                if var_choice == 0:
                    factor = v[j*p+k, r] * w[k*n+i, r]
                    if factor == 0:
                        continue
                    # Try delta in {-2, -1, 1, 2}
                    # Ideally delta = -err / factor
                    best_delta = None
                    best_local_err = abs(err)
                    for d in [-2, -1, 1, 2]:
                        new_err = err - d * factor
                        if abs(new_err) < best_local_err:
                            best_local_err = abs(new_err)
                            best_delta = d
                    if best_delta is None:
                        continue
                    
                    # Check global impact
                    old_u_val = u[i*m+j, r]
                    u[i*m+j, r] = old_u_val + best_delta
                    new_E = compute_residuals(u, v, w)
                    new_error = total_error(new_E)
                    
                    if new_error < curr_error:
                        curr_error = new_error
                        E = new_E
                        if curr_error == 0:
                            return u, v, w
                        if curr_error < best_error:
                            best_error = curr_error
                            best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                    else:
                        u[i*m+j, r] = old_u_val
                        
                elif var_choice == 1:
                    factor = u[i*m+j, r] * w[k*n+i, r]
                    if factor == 0:
                        continue
                    best_delta = None
                    best_local_err = abs(err)
                    for d in [-2, -1, 1, 2]:
                        new_err = err - d * factor
                        if abs(new_err) < best_local_err:
                            best_local_err = abs(new_err)
                            best_delta = d
                    if best_delta is None:
                        continue
                    
                    old_v_val = v[j*p+k, r]
                    v[j*p+k, r] = old_v_val + best_delta
                    new_E = compute_residuals(u, v, w)
                    new_error = total_error(new_E)
                    
                    if new_error < curr_error:
                        curr_error = new_error
                        E = new_E
                        if curr_error == 0:
                            return u, v, w
                        if curr_error < best_error:
                            best_error = curr_error
                            best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                    else:
                        v[j*p+k, r] = old_v_val
                        
                elif var_choice == 2:
                    factor = u[i*m+j, r] * v[j*p+k, r]
                    if factor == 0:
                        continue
                    best_delta = None
                    best_local_err = abs(err)
                    for d in [-2, -1, 1, 2]:
                        new_err = err - d * factor
                        if abs(new_err) < best_local_err:
                            best_local_err = abs(new_err)
                            best_delta = d
                    if best_delta is None:
                        continue
                    
                    old_w_val = w[k*n+i, r]
                    w[k*n+i, r] = old_w_val + best_delta
                    new_E = compute_residuals(u, v, w)
                    new_error = total_error(new_E)
                    
                    if new_error < curr_error:
                        curr_error = new_error
                        E = new_E
                        if curr_error == 0:
                            return u, v, w
                        if curr_error < best_error:
                            best_error = curr_error
                            best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                    else:
                        w[k*n+i, r] = old_w_val
                        
        return best_u, best_v, best_w

    # Search for minimum rank
    # Lower bound is 12.
    # We try ranks from 12 upwards.
    
    deadline = time.time() + 115 # Leave 5s margin
    
    for R in range(12, 25):
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this rank
        # Give more time to lower ranks
        time_budget = min(10, deadline - time.time())
        
        u, v, w = local_search(R, time_budget)
        
        if u is not None:
            # Verify one last time
            if verify_int(u, v, w):
                # Convert back to half-integers
                U = (u / 2.0).tolist()
                V = (v / 2.0).tolist()
                W = (w / 2.0).tolist()
                return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook rank 24
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

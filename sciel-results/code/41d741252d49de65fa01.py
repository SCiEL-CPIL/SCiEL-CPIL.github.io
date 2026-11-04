import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 100  # 50s margin
    
    # Pre-compute equation indices
    # Equation (i, j, k) corresponds to index eq_idx
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For fast error calculation, we store U, V, W as numpy arrays
    # U: (n*m, r)
    # V: (m*p, r)
    # W: (p*n, r)
    
    def compute_error_numpy(U, V, W, r):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # We need to compute sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for all i,j,k
        # This is a tensor contraction.
        # We can loop over i, j, k since n*m*p is small (72)
        err = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U[a], V[b] * W[c])
                    eq_idx = (i * m + j) * p + k
                    err[eq_idx] = s - 1.0
        return err

    def check_integer_half(U, V, W, r, tol=1e-9):
        # Check if all entries are multiples of 0.5
        # We check if 2*x is close to an integer
        if not np.all(np.abs(2*U - np.round(2*U)) < tol): return False
        if not np.all(np.abs(2*V - np.round(2*V)) < tol): return False
        if not np.all(np.abs(2*W - np.round(2*W)) < tol): return False
        return True

    def search_rank(r, time_limit):
        # Try to find a solution of rank r
        # Values to try: integers and half-integers
        # We use a local search on continuous values, then snap to grid?
        # Or just search on the grid directly?
        # Grid search is too big.
        # Let's do local search on continuous, then check if close to grid.
        # If close, snap and verify.
        
        best_err = float('inf')
        best_sol = None
        
        max_restarts = 500
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Initialize with random small values
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Add some structure: maybe start from schoolbook and randomize?
            # For r < 72, we can't start from schoolbook directly.
            
            curr_err = np.sum(compute_error_numpy(U, V, W, r) ** 2)
            
            if curr_err < best_err:
                best_err = curr_err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if curr_err < 1e-10:
                if check_integer_half(U, V, W, r):
                    return U, V, W

            # Local Search
            no_improve = 0
            max_iters = 2000
            step_size = 0.5
            
            for _ in range(max_iters):
                if time.time() > time_limit:
                    break
                
                # Pick random variable
                var_type = np.random.randint(3)
                if var_type == 0:
                    a = np.random.randint(n * m)
                    t = np.random.randint(r)
                    old_val = U[a, t]
                    U[a, t] += np.random.choice([-1, 1]) * step_size
                    new_val = U[a, t]
                elif var_type == 1:
                    b = np.random.randint(m * p)
                    t = np.random.randint(r)
                    old_val = V[b, t]
                    V[b, t] += np.random.choice([-1, 1]) * step_size
                    new_val = V[b, t]
                else:
                    c = np.random.randint(p * n)
                    t = np.random.randint(r)
                    old_val = W[c, t]
                    W[c, t] += np.random.choice([-1, 1]) * step_size
                    new_val = W[c, t]
                
                new_err = np.sum(compute_error_numpy(U, V, W, r) ** 2)
                
                if new_err < curr_err:
                    curr_err = new_err
                    no_improve = 0
                    if curr_err < 1e-10:
                        if check_integer_half(U, V, W, r):
                            return U, V, W
                else:
                    # Revert
                    if var_type == 0:
                        U[a, t] = old_val
                    elif var_type == 1:
                        V[b, t] = old_val
                    else:
                        W[c, t] = old_val
                    no_improve += 1
                    if no_improve > 50:
                        break
                        
            # Snap to grid and check
            if curr_err < 1.0: # If close to solution
                U_snap = np.round(2 * U) / 2.0
                V_snap = np.round(2 * V) / 2.0
                W_snap = np.round(2 * W) / 2.0
                snap_err = np.sum(compute_error_numpy(U_snap, V_snap, W_snap, r) ** 2)
                if snap_err < 1e-9:
                    return U_snap, V_snap, W_snap

        return None

    # Try ranks lower than schoolbook
    # We try a few ranks. If we find one, we return it.
    # We prioritize finding *any* improvement.
    # Given the difficulty, we try 71, 70, etc.
    # But usually, if a lower rank exists, it might be significantly lower.
    # However, for this specific size, it's unknown.
    # Let's try 71 first.
    
    for r in [71, 70, 69, 68]:
        if time.time() > deadline:
            break
        sol = search_rank(r, deadline)
        if sol:
            U, V, W = sol
            # Convert to lists of lists for output
            U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
            V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
            W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END

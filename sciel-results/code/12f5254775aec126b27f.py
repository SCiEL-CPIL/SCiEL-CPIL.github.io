import numpy as np
import time

DIMENSIONS = [3, 3, 8]

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
    deadline = start_time + 110

    def compute_error_vectorized(U, V, W, r, n, m, p):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Compute T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # Note: V index is (j,k), W index is (k,i)
        # So we need to align indices.
        # U: i, j, t
        # V: j, k, t
        # W: k, i, t
        
        # einsum: 'ijt,jkt,kit->ijk'
        T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        
        # Target is 1 for all i,j,k
        target = np.ones((n, m, p))
        err = T - target
        return err

    def check_integer_half(U, V, W, r, tol=1e-9):
        if not np.all(np.abs(2*U - np.round(2*U)) < tol): return False
        if not np.all(np.abs(2*V - np.round(2*V)) < tol): return False
        if not np.all(np.abs(2*W - np.round(2*W)) < tol): return False
        return True

    def snap_to_grid(U, V, W):
        U_snap = np.round(2 * U) / 2.0
        V_snap = np.round(2 * V) / 2.0
        W_snap = np.round(2 * W) / 2.0
        return U_snap, V_snap, W_snap

    def verify_solution(U, V, W, r, n, m, p):
        err = compute_error_vectorized(U, V, W, r, n, m, p)
        if np.max(np.abs(err)) < 1e-8:
            if check_integer_half(U, V, W, r):
                return True
        return False

    def search_rank(r, n, m, p, time_limit):
        best_err = float('inf')
        best_sol = None
        
        max_restarts = 2000 # Increase restarts
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Initialize with small random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            curr_err = np.sum(compute_error_vectorized(U, V, W, r, n, m, p) ** 2)
            
            if curr_err < best_err:
                best_err = curr_err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if curr_err < 1e-10:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                    return U_snap, V_snap, W_snap

            no_improve = 0
            max_iters = 500 # Decrease iters per restart, more restarts
            step_size = 0.1
            
            for _ in range(max_iters):
                if time.time() > time_limit:
                    break
                
                var_type = np.random.randint(3)
                if var_type == 0:
                    a = np.random.randint(n * m)
                    t = np.random.randint(r)
                    old_val = U[a, t]
                    U[a, t] += np.random.choice([-1, 1]) * step_size
                elif var_type == 1:
                    b = np.random.randint(m * p)
                    t = np.random.randint(r)
                    old_val = V[b, t]
                    V[b, t] += np.random.choice([-1, 1]) * step_size
                else:
                    c = np.random.randint(p * n)
                    t = np.random.randint(r)
                    old_val = W[c, t]
                    W[c, t] += np.random.choice([-1, 1]) * step_size
                
                new_err = np.sum(compute_error_vectorized(U, V, W, r, n, m, p) ** 2)
                
                if new_err < curr_err:
                    curr_err = new_err
                    no_improve = 0
                    if np.random.rand() < 0.05:
                        step_size *= 1.1
                    if curr_err < 1e-10:
                        U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                        if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                            return U_snap, V_snap, W_snap
                else:
                    if var_type == 0:
                        U[a, t] = old_val
                    elif var_type == 1:
                        V[b, t] = old_val
                    else:
                        W[c, t] = old_val
                    no_improve += 1
                    if no_improve % 10 == 0:
                        step_size *= 0.9
                    if no_improve > 50:
                        break
                        
            if curr_err < 0.1:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                snap_err = np.sum(compute_error_vectorized(U_snap, V_snap, W_snap, r, n, m, p) ** 2)
                if snap_err < 1e-8:
                    if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                        return U_snap, V_snap, W_snap

        return None

    # Target rank 20 for 3x3x4, which gives rank 40 for 3x3x8
    n4, m4, p4 = 3, 3, 4
    sol_4 = search_rank(20, n4, m4, p4, deadline)
    
    if sol_4:
        U4, V4, W4 = sol_4
        r4 = 20
        U8 = np.hstack([U4, U4])
        V8 = np.zeros((24, 40))
        W8 = np.zeros((24, 40))
        
        V8[0:12, 0:20] = V4
        V8[12:24, 20:40] = V4
        
        W8[0:12, 0:20] = W4
        W8[12:24, 20:40] = W4
        
        if verify_solution(U8, V8, W8, 40, n, m, p):
            U_list = [[int(x) if x == int(x) else x for x in row] for row in U8]
            V_list = [[int(x) if x == int(x) else x for x in row] for row in V8]
            W_list = [[int(x) if x == int(x) else x for x in row] for row in W8]
            return {"U": U_list, "V": V_list, "W": W_list}

    # If rank 20 fails, try rank 21
    sol_4 = search_rank(21, n4, m4, p4, deadline)
    if sol_4:
        U4, V4, W4 = sol_4
        r4 = 21
        U8 = np.hstack([U4, U4])
        V8 = np.zeros((24, 42))
        W8 = np.zeros((24, 42))
        
        V8[0:12, 0:21] = V4
        V8[12:24, 21:42] = V4
        
        W8[0:12, 0:21] = W4
        W8[12:24, 21:42] = W4
        
        if verify_solution(U8, V8, W8, 42, n, m, p):
            U_list = [[int(x) if x == int(x) else x for x in row] for row in U8]
            V_list = [[int(x) if x == int(x) else x for x in row] for row in V8]
            W_list = [[int(x) if x == int(x) else x for x in row] for row in W8]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to seed's logic if needed, but we expect to find 40 or 42
    return {"U": U_school, "V": V_school, "W": W_school}

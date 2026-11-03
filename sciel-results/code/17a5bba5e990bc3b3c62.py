import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1: (A11+A22)(B11+B22)
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2: A22(B21-B11)
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3: (A12-A22)(B21+B22)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4: (A11-A12)(B12+B22)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5: (A11+A12)B22
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6: A11(B12-B22)
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7: (A21-A11)(B11+B12)
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1
        
        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj), col] = 1
                    V[(j0 + dj)*p + k0, col] = 1
                    W[k0*n + (i0 + di), col] = 1
                    col += 1
        
        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)
        
        return U, V, W

    U36, V36, W36 = get_initial_solution_rank36()
    
    def verify_solution(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                val = 0.0
                                for mu in range(R):
                                    val += U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu]
                                if b == c and a == i and d == k:
                                    if abs(val - 1.0) > tol:
                                        return False
                                else:
                                    if abs(val) > tol:
                                        return False
        return True

    def snap_to_half_integers(U, V, W):
        return (np.round(U * 2) / 2.0, 
                np.round(V * 2) / 2.0, 
                np.round(W * 2) / 2.0)

    # Precompute constraint indices for fast residual
    cons_idx = []
    cons_target = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_idx.append((a*m+b, c*p+d, k*n+i))
                            cons_target.append(target)
    
    cons_idx = np.array(cons_idx)
    cons_target = np.array(cons_target)
    
    def residual_vector(x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
        
        # Vectorized: for each constraint (u_idx, v_idx, w_idx), compute dot(U[u_idx], V[v_idx]*W[w_idx])
        u_rows = cons_idx[:, 0]
        v_rows = cons_idx[:, 1]
        w_rows = cons_idx[:, 2]
        
        U_sub = U[u_rows]  # (N_cons, R)
        V_sub = V[v_rows]  # (N_cons, R)
        W_sub = W[w_rows]  # (N_cons, R)
        
        vals = np.sum(U_sub * V_sub * W_sub, axis=1)
        return vals - cons_target
    
    best_rank = 36
    best_solution = (U36, V36, W36)
    
    # Try to find lower ranks
    for R in range(35, 19, -1):
        if time.time() - start_time > time_limit:
            break
        
        max_attempts = 8
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Truncate
                U_init = U36[:, :R]
                V_init = V36[:, :R]
                W_init = W36[:, :R]
            elif attempt == 1:
                # Random small
                U_init = np.random.normal(0, 0.3, (n*m, R))
                V_init = np.random.normal(0, 0.3, (m*p, R))
                W_init = np.random.normal(0, 0.3, (p*n, R))
            elif attempt == 2:
                # Random with some structure from rank 36
                U_init = U36[:, :R] * 0.5 + np.random.normal(0, 0.2, (n*m, R))
                V_init = V36[:, :R] * 0.5 + np.random.normal(0, 0.2, (m*p, R))
                W_init = W36[:, :R] * 0.5 + np.random.normal(0, 0.2, (p*n, R))
            else:
                # Pure random
                U_init = np.random.normal(0, 0.5, (n*m, R))
                V_init = np.random.normal(0, 0.5, (m*p, R))
                W_init = np.random.normal(0, 0.5, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residual_vector, x0, args=(R,), 
                                          method='lm', max_nfev=5000, ftol=1e-12)
                res_norm = np.linalg.norm(result.fun)
                
                if res_norm < 1e-4:
                    x_opt = result.x
                    size_U = n * m * R
                    size_V = m * p * R
                    size_W = p * n * R
                    U_opt = x_opt[0:size_U].reshape(n*m, R)
                    V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                    W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                    
                    # Polish with tighter tolerance
                    try:
                        result2 = opt.least_squares(residual_vector, x_opt, args=(R,),
                                                    method='lm', max_nfev=2000, ftol=1e-15)
                        if np.linalg.norm(result2.fun) < 1e-8:
                            x_opt = result2.x
                            U_opt = x_opt[0:size_U].reshape(n*m, R)
                            V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                            W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                    except:
                        pass
                    
                    # Snap to half-integers
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_snap, V_snap, W_snap, tol=1e-8):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    
                    # Try integer snapping too
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if verify_solution(U_int, V_int, W_int, tol=1e-8):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                pass
    
    # Return the best known solution (rank 36)
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    if verify_solution(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}

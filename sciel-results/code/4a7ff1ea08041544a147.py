import numpy as np
import time
from scipy.optimize import least_squares
from scipy.optimize import nnls

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = n * m * p
    
    # Precompute indices for vectorized verification
    idx_u = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    idx_v = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    idx_w = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    def verify_solution(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        res = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.allclose(res, np.ones(total_entries), atol=tol, rtol=0)
    
    def get_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    r += 1
        return U, V, W
    
    def try_find_rank(R, time_limit=10.0, seed=0):
        """
        Attempt to find a rank R decomposition using alternating NNLS and least squares.
        Combines the structural insight of fixing one matrix and solving for others
        with the robustness of least squares polishing.
        """
        if R >= n * m * p:
            return None
        
        rng = np.random.RandomState(seed)
        
        # We use multiple restarts
        num_restarts = 8
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small random values
            # Scale matters for convergence
            U = rng.uniform(-1, 1, (nm, R))
            V = rng.uniform(-1, 1, (mp, R))
            W = rng.uniform(-1, 1, (pn, R))
            
            # Phase 1: Alternating NNLS (from inspiration 1)
            # This helps find a good local minimum by respecting the structure
            # where each row of U, V, W is solved independently given the others.
            for _ in range(50):
                if time.time() - start_time > time_limit * 0.7:
                    break
                
                # Update U: for each (i,j), solve A * U_row = ones(p)
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, R))
                        for k in range(p):
                            for c in range(R):
                                A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        b = np.ones(p)
                        try:
                            sol, _ = nnls(A, b)
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V: for each (j,k), solve A * V_row = ones(n)
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R))
                        for i in range(n):
                            for c in range(R):
                                A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        b = np.ones(n)
                        try:
                            sol, _ = nnls(A, b)
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W: for each (k,i), solve A * W_row = ones(m)
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R))
                        for j in range(m):
                            for c in range(R):
                                A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        b = np.ones(m)
                        try:
                            sol, _ = nnls(A, b)
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check error periodically
                if _ % 10 == 0:
                    U_sel = U[idx_u, :]
                    V_sel = V[idx_v, :]
                    W_sel = W[idx_w, :]
                    res = np.sum(U_sel * V_sel * W_sel, axis=1)
                    err = np.sum((res - 1.0)**2)
                    if err < 1e-12:
                        break
            
            # Phase 2: Least Squares Polish (from current program)
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_flat = x[:nm*R]
                V_flat = x[nm*R:nm*R+mp*R]
                W_flat = x[nm*R+mp*R:]
                
                U_mat = U_flat.reshape(nm, R)
                V_mat = V_flat.reshape(mp, R)
                W_mat = W_flat.reshape(pn, R)
                
                U_sel = U_mat[idx_u, :]
                V_sel = V_mat[idx_v, :]
                W_sel = W_mat[idx_w, :]
                
                res = np.sum(U_sel * V_sel * W_sel, axis=1)
                return res - 1.0
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=2000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
                
                if result.success or result.cost < 1e-10:
                    x_sol = result.x
                    U_sol = x_sol[:nm*R].reshape(nm, R)
                    V_sol = x_sol[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_sol = x_sol[nm*R+mp*R:].reshape(pn, R)
                    
                    # Check if it's already close to half-integers
                    # Round to nearest half-integer
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    if verify_solution(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    
                    # If direct rounding fails, try local search
                    if result.cost < 1e-6:
                        # Local search: perturb entries to nearest half-integers
                        current_U, current_V, current_W = U_r, V_r, W_r
                        
                        improved = True
                        while improved and (time.time() - start_time < time_limit):
                            improved = False
                            # Try flipping signs or adding 0.5 to random entries
                            for _ in range(20):
                                U_p = current_U.copy()
                                V_p = current_V.copy()
                                W_p = current_W.copy()
                                
                                # Perturb a few entries
                                for _ in range(3):
                                    # Randomly choose which matrix to perturb
                                    mat_choice = rng.randint(3)
                                    if mat_choice == 0:
                                        r_idx, c_idx = np.unravel_index(rng.randint(nm*R), (nm, R))
                                        U_p[r_idx, c_idx] += rng.choice([-0.5, 0.5])
                                    elif mat_choice == 1:
                                        r_idx, c_idx = np.unravel_index(rng.randint(mp*R), (mp, R))
                                        V_p[r_idx, c_idx] += rng.choice([-0.5, 0.5])
                                    else:
                                        r_idx, c_idx = np.unravel_index(rng.randint(pn*R), (pn, R))
                                        W_p[r_idx, c_idx] += rng.choice([-0.5, 0.5])
                                        
                                if verify_solution(U_p, V_p, W_p):
                                    return U_p, V_p, W_p
                                    
            except Exception as e:
                pass
                
        return None

    # Main Logic
    U_sb, V_sb, W_sb = get_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    timeout = 110.0
    
    # Try ranks from 29 down to 25
    for R in range(29, 24, -1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        for seed in range(15):
            if time.time() - start_time > timeout:
                break
            res = try_find_rank(R, time_limit=3.0, seed=seed)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            # If we found a solution for R, we can try to go lower
            continue
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

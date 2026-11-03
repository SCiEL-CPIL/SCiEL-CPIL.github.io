import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Build the target tensor T[a, b, c, d, k, i] = delta_b_c * delta_a_i * delta_d_k
    # Shape: (n*m, m*p, p*n) -> (8, 20, 10)
    # T[a*m+b, c*p+d, k*n+i] = 1 if b==c and a==i and d==k, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                a = i  # delta_a_i
                d = k  # delta_d_k
                c = b  # delta_b_c
                T[a*m+b, c*p+d, k*n+i] = 1.0
    
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7
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
    
    def verify_solution_copy(U, V, W, tol=1e-9):
        T_copy = T.copy()
        R = U.shape[1]
        for mu in range(R):
            T_copy -= np.einsum('a,b,c->abc', U[:,mu], V[:,mu], W[:,mu])
        return np.max(np.abs(T_copy)) < tol

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    def residual_vector_vectorized(x, R):
        """Compute residual using vectorized tensor operations."""
        size_U = n * m * R
        size_V = m * p * R
        
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return (T_rec - T).flatten()
    
    def gradient_residual_vectorized(x, R):
        """Compute gradient of 0.5*||residual||^2 using vectorized operations."""
        size_U = n * m * R
        size_V = m * p * R
        
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        residual = T_rec - T
        
        grad_U = np.einsum('abc,bm,cm->am', residual, V, W, optimize=True)
        grad_V = np.einsum('abc,am,cm->bm', residual, U, W, optimize=True)
        grad_W = np.einsum('abc,am,bm->cm', residual, U, V, optimize=True)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    start_time = time.time()
    time_limit = 140
    
    best_rank = 36
    best_result = None
    
    def try_optimize(U_init, V_init, W_init, R, maxiter=3000, maxfun=6000):
        """Run L-BFGS-B optimization and return (U, V, W, max_res)."""
        size_U = n * m * R
        size_V = m * p * R
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.minimize(
                lambda x: 0.5 * np.sum(residual_vector_vectorized(x, R)**2),
                x0,
                method='L-BFGS-B',
                jac=lambda x: gradient_residual_vectorized(x, R),
                maxiter=maxiter,
                maxfun=maxfun,
                options={'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 80}
            )
            
            x_opt = result.x
            U_opt = x_opt[0:size_U].reshape(n*m, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
            W_opt = x_opt[size_U+size_V:].reshape(p*n, R)
            
            T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
            max_res = np.max(np.abs(T_rec - T))
            
            return U_opt, V_opt, W_opt, max_res
        except Exception:
            return None, None, None, np.inf
    
    def polish(U_init, V_init, W_init, R):
        """High-precision polish pass."""
        size_U = n * m * R
        size_V = m * p * R
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.minimize(
                lambda x: 0.5 * np.sum(residual_vector_vectorized(x, R)**2),
                x0,
                method='L-BFGS-B',
                jac=lambda x: gradient_residual_vectorized(x, R),
                maxiter=15000,
                maxfun=30000,
                options={'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            
            x_opt = result.x
            U_opt = x_opt[0:size_U].reshape(n*m, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
            W_opt = x_opt[size_U+size_V:].reshape(p*n, R)
            
            T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
            max_res = np.max(np.abs(T_rec - T))
            
            return U_opt, V_opt, W_opt, max_res
        except Exception:
            return None, None, None, np.inf
    
    # Try ranks from 35 down to 24
    for R in range(35, 23, -1):
        if time.time() - start_time > time_limit:
            break
        
        size_U = n * m * R
        size_V = m * p * R
        
        # Number of attempts based on remaining time
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        n_attempts = min(15, max(5, int(remaining / 8)))
        
        found = False
        for attempt in range(n_attempts):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0 and R <= 36:
                # Truncate the known rank-36 solution
                U_init = U36[:, :R].copy()
                V_init = V36[:, :R].copy()
                W_init = W36[:, :R].copy()
            elif attempt == 1:
                U_init = np.random.normal(0, 1.0, (n*m, R))
                V_init = np.random.normal(0, 1.0, (m*p, R))
                W_init = np.random.normal(0, 1.0, (p*n, R))
            elif attempt == 2:
                U_init = np.random.normal(0, 0.5, (n*m, R))
                V_init = np.random.normal(0, 0.5, (m*p, R))
                W_init = np.random.normal(0, 0.5, (p*n, R))
            elif attempt == 3:
                U_init = np.random.choice([-1, 0, 0, 0, 1], (n*m, R)).astype(float)
                V_init = np.random.choice([-1, 0, 0, 0, 1], (m*p, R)).astype(float)
                W_init = np.random.choice([-1, 0, 0, 0, 1], (p*n, R)).astype(float)
            elif attempt == 4:
                U_init = np.random.normal(0, 2.0, (n*m, R))
                V_init = np.random.normal(0, 2.0, (m*p, R))
                W_init = np.random.normal(0, 2.0, (p*n, R))
            elif attempt == 5:
                # Sparse random initialization
                U_init = np.zeros((n*m, R))
                V_init = np.zeros((m*p, R))
                W_init = np.zeros((p*n, R))
                for mu in range(R):
                    idx_u = np.random.choice(n*m, size=3, replace=False)
                    idx_v = np.random.choice(m*p, size=3, replace=False)
                    idx_w = np.random.choice(p*n, size=3, replace=False)
                    U_init[idx_u, mu] = np.random.choice([-1, 1], size=3)
                    V_init[idx_v, mu] = np.random.choice([-1, 1], size=3)
                    W_init[idx_w, mu] = np.random.choice([-1, 1], size=3)
            else:
                scale = 0.2 + 0.15 * (attempt % 6)
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            U_opt, V_opt, W_opt, max_res = try_optimize(U_init, V_init, W_init, R)
            
            if U_opt is None:
                continue
            
            if max_res < 1e-6:
                # Try polishing
                U_pol, V_pol, W_pol, max_res_pol = polish(U_opt, V_opt, W_opt, R)
                if U_pol is not None and max_res_pol < 1e-8:
                    U_opt, V_opt, W_opt = U_pol, V_pol, W_pol
                    max_res = max_res_pol
                
                U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
                if np.max(np.abs(T_rec_snap - T)) < 1e-9:
                    best_rank = R
                    best_result = (U_snap, V_snap, W_snap)
                    found = True
                    break
            
            elif max_res < 1e-3:
                # Close but not quite — try polishing
                U_pol, V_pol, W_pol, max_res_pol = polish(U_opt, V_opt, W_opt, R)
                if U_pol is not None and max_res_pol < 1e-8:
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_pol, V_pol, W_pol)
                    T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
                    if np.max(np.abs(T_rec_snap - T)) < 1e-9:
                        best_rank = R
                        best_result = (U_snap, V_snap, W_snap)
                        found = True
                        break
        
        if found:
            break
    
    if best_result is not None:
        U_snap, V_snap, W_snap = best_result
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: return the rank-36 solution
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    T_rec = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
    if np.max(np.abs(T_rec - T)) < 1e-9:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}

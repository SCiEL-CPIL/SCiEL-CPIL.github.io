import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Start with the known rank-36 solution (Strassen blocks + schoolbook).
    2. Attempt to reduce the rank using vectorized least-squares optimization.
    3. Verify exactness with integer/half-integer snapping.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Precompute constraint structure
    # For each (i, k, a, b, c, d), target = 1 if (b==c and a==i and d==k) else 0
    # Total: n*p*n*m*m*p = 2*5*2*4*4*5 = 1600 constraints
    
    # Precompute index arrays for vectorized computation
    cons_i = []
    cons_k = []
    cons_a = []
    cons_b = []
    cons_c = []
    cons_d = []
    cons_target = []
    
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            cons_i.append(i)
                            cons_k.append(k)
                            cons_a.append(a)
                            cons_b.append(b)
                            cons_c.append(c)
                            cons_d.append(d)
                            cons_target.append(1.0 if (b == c and a == i and d == k) else 0.0)
    
    cons_i = np.array(cons_i)
    cons_k = np.array(cons_k)
    cons_a = np.array(cons_a)
    cons_b = np.array(cons_b)
    cons_c = np.array(cons_c)
    cons_d = np.array(cons_d)
    cons_target = np.array(cons_target)
    n_cons = len(cons_target)
    
    def get_initial_solution_rank36():
        # Reconstruct the known rank 36 solution
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0, col_start):
            nonlocal col
            # M1
            U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
            col += 1
            # M5
            U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1)][col] += 1; U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1
            return col_start + 7

        def add_schoolbook_2x4x1(i0, j0, k0, col_start):
            c = col_start
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj)][c] = 1
                    V[(j0 + dj)*p + k0][c] = 1
                    W[k0*n + (i0 + di)][c] = 1
                    c += 1
            return c

        col = add_strassen_222(0, 0, 0, 0)
        col = add_strassen_222(0, 2, 0, col)
        col = add_strassen_222(0, 0, 2, col)
        col = add_strassen_222(0, 2, 2, col)
        col = add_schoolbook_2x4x1(0, 0, 4, col)
        
        return U, V, W

    U36, V36, W36 = get_initial_solution_rank36()
    
    # Helper to verify a solution using vectorized computation
    def verify_solution(U, V, W, tol=1e-9):
        # Check if U, V, W satisfy the tensor equations exactly (with tolerance)
        R = U.shape[1]
        # U shape: (n*m, R), V shape: (m*p, R), W shape: (p*n, R)
        # For constraint (i,k,a,b,c,d): sum_mu U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu]
        
        # Build index arrays
        idx_a = cons_a * m + cons_b  # index into U
        idx_c = cons_c * p + cons_d  # index into V
        idx_w = cons_k * n + cons_i  # index into W
        
        U_sel = U[idx_a]  # shape: (n_cons, R)
        V_sel = V[idx_c]  # shape: (n_cons, R)
        W_sel = W[idx_w]  # shape: (n_cons, R)
        
        # Compute sum_mu U_sel * V_sel * W_sel for each constraint
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        residuals = vals - cons_target
        return np.max(np.abs(residuals)) < tol
    
    # Helper to check if entries are close to half-integers
    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped
    
    # Fast vectorized residual computation
    def residual_vector_fast(x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        # Build index arrays
        idx_a = cons_a * m + cons_b  # index into U
        idx_c = cons_c * p + cons_d  # index into V
        idx_w = cons_k * n + cons_i  # index into W
        
        U_sel = U[idx_a]  # shape: (n_cons, R)
        V_sel = V[idx_c]  # shape: (n_cons, R)
        W_sel = W[idx_w]  # shape: (n_cons, R)
        
        # Compute sum_mu U_sel * V_sel * W_sel for each constraint
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        return vals - cons_target
    
    start_time = time.time()
    time_limit = 110 # Leave margin
    
    # Precompute index arrays once
    idx_a = cons_a * m + cons_b
    idx_c = cons_c * p + cons_d
    idx_w = cons_k * n + cons_i
    
    def residual_vector_fast_precomputed(x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        U_sel = U[idx_a]
        V_sel = V[idx_c]
        W_sel = W[idx_w]
        
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        return vals - cons_target
    
    # Try to find lower ranks
    for R in range(35, 25, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize x
            if attempt == 0:
                # Truncate rank 36 solution
                U_init = U36[:, :R]
                V_init = V36[:, :R]
                W_init = W36[:, :R]
            elif attempt == 1:
                # Scaled random initialization
                U_init = np.random.normal(0, 1.0, (n*m, R))
                V_init = np.random.normal(0, 1.0, (m*p, R))
                W_init = np.random.normal(0, 1.0, (p*n, R))
            elif attempt == 2:
                # Different random scale
                U_init = np.random.normal(0, 0.5, (n*m, R))
                V_init = np.random.normal(0, 0.5, (m*p, R))
                W_init = np.random.normal(0, 0.5, (p*n, R))
            else:
                # Random with varying scales
                scale = 0.3 + 0.1 * (attempt % 5)
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use 'lm' for small problems
                result = opt.least_squares(residual_vector_fast_precomputed, x0, args=(R,), method='lm', max_nfev=30000)
                
                x_opt = result.x
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                
                # Check residual
                if verify_solution(U_opt, V_opt, W_opt, tol=1e-6):
                    # Snap to half-integers
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_snap, V_snap, W_snap, tol=1e-9):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            except Exception as e:
                pass
    
    # If no better rank found, return the rank 36 solution (snapped)
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    if verify_solution(U_snap, V_snap, W_snap, tol=1e-9):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: return original
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}

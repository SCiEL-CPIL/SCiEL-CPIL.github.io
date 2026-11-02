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
    2. Attempt to reduce the rank to 35, 34, etc. using least-squares optimization.
    3. Verify exactness with integer/half-integer snapping.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Build the target tensor T[i, k, a, b, c, d] = delta_b_c * delta_a_i * delta_d_k
    # Actually, we just need the constraints for the optimization.
    # Constraint for (i, k, a, b, c, d):
    # sum_mu U[a, b, mu] V[c, d, mu] W[k, i, mu] == delta[b, c] * delta[a, i] * delta[d, k]
    
    # Precompute indices for constraints
    # i in [0, n), k in [0, p)
    # a in [0, n), b in [0, m), c in [0, m), d in [0, p)
    
    constraints = []
    target_vals = []
    
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            # Target value
                            if b == c and a == i and d == k:
                                target = 1.0
                            else:
                                target = 0.0
                            if target != 0.0:
                                constraints.append((i, k, a, b, c, d, target))
    
    # There are n*p*n*m*m*p constraints, but many are 0.
    # Only n*p constraints are non-zero?
    # delta[b,c] is 1 if b==c.
    # delta[a,i] is 1 if a==i.
    # delta[d,k] is 1 if d==k.
    # So for each (i,k), only the terms where b==c, a==i, d==k are non-zero.
    # Number of non-zero constraints: sum_{i,k} (1 * 1 * 1 * m) = n * p * m = 2 * 5 * 4 = 40.
    
    # Wait, the sum is over mu. The equation is:
    # sum_mu U[a,b,mu] V[c,d,mu] W[k,i,mu] = 0 for all (a,b,c,d) NOT satisfying the deltas.
    # And = 1 for the ones satisfying.
    # So we have many constraints equal to 0.
    # Total constraints: n*p*n*m*m*p = 2*5*2*4*4*5 = 1600.
    # We need to satisfy all of them.
    
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
    
    # Helper to verify a solution
    def verify_solution(U, V, W):
        # Check if U, V, W satisfy the tensor equations exactly (with tolerance)
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
                                    if abs(val - 1.0) > 1e-9:
                                        return False
                                else:
                                    if abs(val - 0.0) > 1e-9:
                                        return False
        return True

    # Helper to check if entries are close to half-integers
    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    # Objective function for optimization
    def residual_vector(x, R):
        # x is flattened [U, V, W]
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        res = []
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                # Compute sum_mu U[a,b,mu] V[c,d,mu] W[k,i,mu]
                                # Vectorized over mu
                                val = np.dot(U[a*m+b], np.multiply(V[c*p+d], W[k*n+i]))
                                if b == c and a == i and d == k:
                                    res.append(val - 1.0)
                                else:
                                    res.append(val - 0.0)
        return np.array(res)
    
    # Precompute constraint structure for speed
    # indices: (i, k, a, b, c, d, target)
    cons_list = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_list.append((i, k, a, b, c, d, target))
    
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
        
        res = np.zeros(len(cons_list))
        idx = 0
        for (i, k, a, b, c, d, target) in cons_list:
            val = np.dot(U[a*m+b], np.multiply(V[c*p+d], W[k*n+i]))
            res[idx] = val - target
            idx += 1
        return res

    start_time = time.time()
    time_limit = 110 # Leave margin
    
    # Try to find lower ranks
    for R in range(35, 25, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random initializations with more attempts
        for attempt in range(15):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize x
            # Scale: values in rank 36 are -1, 0, 1.
            # Random initialization with small values might work better for finding new structures
            # Or truncate the rank 36 solution
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
            else:
                # Random with varying scales
                scale = 0.5 * (1 + attempt/10)
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use 'lm' for small problems, fall back to 'trf' if needed
                result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=20000)
                if not result.success:
                    # Try with 'trf' method as fallback
                    result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='trf', max_nfev=20000)
                
                x_opt = result.x
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                
                # Check residual
                res = residual_vector_fast(x_opt, R)
                if np.max(np.abs(res)) < 1e-6:
                    # Snap to half-integers
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_snap, V_snap, W_snap):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            except Exception as e:
                pass

    # If no better rank found, return the rank 36 solution (snapped)
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    if verify_solution(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: return original
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}

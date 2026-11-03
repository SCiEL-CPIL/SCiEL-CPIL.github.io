import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    
    # The multiplication tensor: T[a, b, c] = 1 iff a=(i,j), b=(j,k), c=(k,i)
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    
    # Build the constraint system
    all_cons_a = []
    all_cons_b = []
    all_cons_c = []
    all_cons_target = []
    
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if i_a == i_c and j_a == j_b and k_b == k_c:
                    target = 1.0
                else:
                    target = 0.0
                all_cons_a.append(a)
                all_cons_b.append(b)
                all_cons_c.append(c)
                all_cons_target.append(target)
    
    all_cons_a = np.array(all_cons_a, dtype=np.int32)
    all_cons_b = np.array(all_cons_b, dtype=np.int32)
    all_cons_c = np.array(all_cons_c, dtype=np.int32)
    all_cons_target = np.array(all_cons_target, dtype=np.float64)
    
    n_cons = len(all_cons_target)
    
    def residual(x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        U_sel = U[all_cons_a]
        V_sel = V[all_cons_b]
        W_sel = W[all_cons_c]
        
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return vals - all_cons_target
    
    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_sel = U[all_cons_a]
        V_sel = V[all_cons_b]
        W_sel = W[all_cons_c]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(vals - all_cons_target)) < tol
    
    def snap(U, V, W):
        return np.round(U*2)/2.0, np.round(V*2)/2.0, np.round(W*2)/2.0
    
    def polish(U, V, W, R, max_iter=100, tol=1e-14):
        """High-precision polish using Gauss-Newton with exact Jacobian"""
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        for it in range(max_iter):
            res = residual(x, R)
            if np.max(np.abs(res)) < tol:
                break
            
            U_mat = x[0:size_U].reshape(n*m, R)
            V_mat = x[size_U:size_U+size_V].reshape(m*p, R)
            W_mat = x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
            
            U_sel = U_mat[all_cons_a]
            V_sel = V_mat[all_cons_b]
            W_sel = W_mat[all_cons_c]
            
            # Build Jacobian
            J = np.zeros((n_cons, len(x)))
            
            # For U: d/dU[a,r] = V_sel[:,r] * W_sel[:,r] where all_cons_a == a
            for r in range(R):
                idx = all_cons_a * R + r
                J[np.arange(n_cons), idx] += V_sel[:, r] * W_sel[:, r]
            
            # For V
            for r in range(R):
                idx = size_U + all_cons_b * R + r
                J[np.arange(n_cons), idx] += U_sel[:, r] * W_sel[:, r]
            
            # For W
            for r in range(R):
                idx = size_U + size_V + all_cons_c * R + r
                J[np.arange(n_cons), idx] += U_sel[:, r] * V_sel[:, r]
            
            # Solve least squares
            try:
                dx, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
            except:
                break
            
            x = x + dx
        
        U_opt = x[0:size_U].reshape(n*m, R)
        V_opt = x[size_U:size_U+size_V].reshape(m*p, R)
        W_opt = x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
        
        return U_opt, V_opt, W_opt
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Build rank-36 Strassen solution as baseline
    def get_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2: A11(B11-B01)
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3: A00(B01)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4: A11(B10)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5: (A00+A01)B10
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6: (A10+A11)(B00)
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7: (A10+A11)(B00+B10)
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        # Four 2x2x2 blocks for j=0,1 and j=2,3, k=0,1 and k=2,3
        add_strassen_222(0, 0, 0)  # i=0, j=0-1, k=0-1
        add_strassen_222(0, 2, 0)  # i=0, j=2-3, k=0-1
        add_strassen_222(0, 0, 2)  # i=0, j=0-1, k=2-3
        add_strassen_222(0, 2, 2)  # i=0, j=2-3, k=2-3
        
        # Schoolbook for k=4
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 4, col] = 1
                W[4*n + i, col] = 1
                col += 1
        
        return U, V, W
    
    U36, V36, W36 = get_rank36()
    
    best_result = None
    best_rank = 36
    
    # Try to find solutions at various ranks
    for R in range(20, 37):
        if time.time() > deadline - 8:
            break
        
        max_attempts = 20 if R <= 24 else 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 4:
                break
            
            # Various initialization strategies
            if attempt == 0:
                # Block-structured: two independent 2x2x5 blocks
                U_init = np.zeros((n*m, R))
                V_init = np.zeros((m*p, R))
                W_init = np.zeros((p*n, R))
                
                half = R // 2
                for block in range(2):
                    cs = block * half
                    ce = cs + half if block < 1 else R
                    # Block 1: j=0,1 ; Block 2: j=2,3
                    U_init[block*2:block*2+2, cs:ce] = np.random.normal(0, 1.0, (2, ce-cs))
                    V_init[block*2:block*2+2, cs:ce] = np.random.normal(0, 1.0, (2, ce-cs))
                    W_init[:, cs:ce] = np.random.normal(0, 1.0, (p*n, ce-cs))
                    
            elif attempt == 1:
                # Truncate from rank 36
                if R <= 36:
                    U_init = U36[:, :R] + np.random.normal(0, 0.1, (n*m, R))
                    V_init = V36[:, :R] + np.random.normal(0, 0.1, (m*p, R))
                    W_init = W36[:, :R] + np.random.normal(0, 0.1, (p*n, R))
                else:
                    U_init = np.random.normal(0, 1.0, (n*m, R))
                    V_init = np.random.normal(0, 1.0, (m*p, R))
                    W_init = np.random.normal(0, 1.0, (p*n, R))
                    
            elif attempt == 2:
                # Random initialization with moderate scale
                scale = 0.5 + 0.5 * np.random.random()
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
                
            elif attempt == 3:
                # Structured with some zeros to encourage sparsity
                U_init = np.random.choice([-1, 0, 1], (n*m, R)) * (np.random.random((n*m, R)) < 0.3)
                V_init = np.random.choice([-1, 0, 1], (m*p, R)) * (np.random.random((m*p, R)) < 0.3)
                W_init = np.random.choice([-1, 0, 1], (p*n, R)) * (np.random.random((p*n, R)) < 0.3)
                
            elif attempt == 4:
                # Use rank-36 as seed with noise
                if R <= 36:
                    U_init = U36[:, :R].copy()
                    V_init = V36[:, :R].copy()
                    W_init = W36[:, :R].copy()
                    # Add small noise to break degeneracy
                    U_init += np.random.normal(0, 0.05, (n*m, R))
                    V_init += np.random.normal(0, 0.05, (m*p, R))
                    W_init += np.random.normal(0, 0.05, (p*n, R))
                else:
                    U_init = np.random.normal(0, 1.0, (n*m, R))
                    V_init = np.random.normal(0, 1.0, (m*p, R))
                    W_init = np.random.normal(0, 1.0, (p*n, R))
                    
            elif attempt == 5:
                # Two-block structure with different random seeds
                U_init = np.zeros((n*m, R))
                V_init = np.zeros((m*p, R))
                W_init = np.zeros((p*n, R))
                # Try to have each block use roughly half the rank
                for block in range(2):
                    cs = block * (R // 2)
                    ce = cs + (R // 2) if block == 0 else R
                    U_init[block*2:block*2+2, cs:ce] = np.random.normal(0, 1.0, (2, ce-cs))
                    V_init[block*2:block*2+2, cs:ce] = np.random.normal(0, 1.0, (2, ce-cs))
                    W_init[:, cs:ce] = np.random.normal(0, 1.0, (p*n, ce-cs))
                    
            else:
                # Scaled random with varied scales
                s1 = 0.3 + 0.4 * np.random.random()
                s2 = 0.3 + 0.4 * np.random.random()
                s3 = 0.3 + 0.4 * np.random.random()
                U_init = np.random.normal(0, s1, (n*m, R))
                V_init = np.random.normal(0, s2, (m*p, R))
                W_init = np.random.normal(0, s3, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use Levenberg-Marquardt for speed
                result = opt.least_squares(residual, x0, args=(R,), method='lm', max_nfev=2000)
                
                if result.cost < 1e-10:
                    x_opt = result.x
                    size_U = n * m * R
                    size_V = m * p * R
                    size_W = p * n * R
                    U_opt = x_opt[0:size_U].reshape(n*m, R)
                    V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                    W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                    
                    if verify(U_opt, V_opt, W_opt, tol=1e-6):
                        # Polish to high precision
                        U_pol, V_pol, W_pol = polish(U_opt, V_opt, W_opt, R, max_iter=150, tol=1e-15)
                        
                        # Try snapping to half-integers
                        U_snap, V_snap, W_snap = snap(U_pol, V_pol, W_pol)
                        if verify(U_snap, V_snap, W_snap, tol=1e-9):
                            if R < best_rank:
                                best_rank = R
                                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                            # Continue searching for even lower rank
                            
                        else:
                            # Try direct snap
                            U_snap2, V_snap2, W_snap2 = snap(U_opt, V_opt, W_opt)
                            if verify(U_snap2, V_snap2, W_snap2, tol=1e-9):
                                if R < best_rank:
                                    best_rank = R
                                    best_result = {"U": U_snap2.tolist(), "V": V_snap2.tolist(), "W": W_snap2.tolist()}
                            
            except Exception:
                pass
    
    # If we found a better result, return it
    if best_result is not None:
        return best_result
    
    # Fallback to rank 36
    U_snap, V_snap, W_snap = snap(U36, V36, W36)
    if verify(U_snap, V_snap, W_snap, tol=1e-9):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}

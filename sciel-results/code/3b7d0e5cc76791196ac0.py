import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    
    # Build constraint indices
    # For 2x4x5 multiplication: T[i,j,k] = 1 if and only if it's the multiplication tensor
    # U has shape (n*m, R) = (8, R), indexed by (i,j) -> i*m+j
    # V has shape (m*p, R) = (20, R), indexed by (j,k) -> j*p+k  
    # W has shape (p*n, R) = (10, R), indexed by (k,i) -> k*n+i
    
    # The constraint: for all (i,k), sum_j U[i,j] * V[j,k] * W[k,i] pattern
    # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (j==j and it's the right entry)
    
    # The multiplication tensor: T_{(i,j),(j,k),(k,i)} = 1 for all valid i,j,k
    # So for each (i,j,k) triple, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations, it should be 0.
    
    # Let's enumerate all constraints
    cons_i = []
    cons_j = []
    cons_k = []
    cons_target = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This is a "1" constraint
                cons_i.append(i)
                cons_j.append(j)
                cons_k.append(k)
                cons_target.append(1.0)
    
    # Also need "0" constraints for all other (i,j,k) where the tensor entry is 0
    # The tensor T_{a,b,c} where a=(i,j) in U, b=(j',k) in V, c=(k',i') in W
    # T = 1 iff i==i', j==j', k==k'
    # So we need all combinations of (a_idx, b_idx, c_idx) where they don't all match
    
    # Let's be more careful. The tensor is:
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    # T[a, b, c] = 0 otherwise
    
    # a ranges over 0..7 (i*m+j), b ranges over 0..19 (j*p+k), c ranges over 0..9 (k*n+i)
    
    # Build all constraints
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
                # T = 1 iff i_a == i_c, j_a == j_b, k_b == k_c
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
        
        U_sel = U[all_cons_a]  # (n_cons, R)
        V_sel = V[all_cons_b]  # (n_cons, R)
        W_sel = W[all_cons_c]  # (n_cons, R)
        
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
    
    def polish(U, V, W, R, max_iter=50, tol=1e-12):
        """High-precision polish using Gauss-Newton"""
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        for it in range(max_iter):
            res = residual(x, R)
            if np.max(np.abs(res)) < tol:
                break
            
            # Compute Jacobian analytically
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            
            U_mat = x[0:size_U].reshape(n*m, R)
            V_mat = x[size_U:size_U+size_V].reshape(m*p, R)
            W_mat = x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
            
            U_sel = U_mat[all_cons_a]  # (n_cons, R)
            V_sel = V_mat[all_cons_b]  # (n_cons, R)
            W_sel = W_mat[all_cons_c]  # (n_cons, R)
            
            # d/dU[a,r] of sum_r' U_sel[:,r'] * V_sel[:,r'] * W_sel[:,r'] 
            # = V_sel[:,r] * W_sel[:,r] for the matching row
            # Jacobian column for U[a,r]: V_sel[:,r] * W_sel[:,r] where all_cons_a == a
            
            J = np.zeros((n_cons, len(x)))
            
            # For U
            for r in range(R):
                mask = (all_cons_a == np.arange(n*m)[None, :])  # This won't work well
                # Better: for each constraint row, the derivative w.r.t. U[all_cons_a[row], r] is V_sel[row,r]*W_sel[row,r]
                # So J[row, all_cons_a[row]*R + r] += V_sel[row,r]*W_sel[row,r]
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
        
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U_opt = x[0:size_U].reshape(n*m, R)
        V_opt = x[size_U:size_U+size_V].reshape(m*p, R)
        W_opt = x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
        
        return U_opt, V_opt, W_opt
    
    start_time = time.time()
    deadline = 140.0
    
    # First, establish a known good solution at rank 36 (Strassen)
    def get_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
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

        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj)][col] = 1
                    V[(j0 + dj)*p + k0][col] = 1
                    W[k0*n + (i0 + di)][col] = 1
                    col += 1

        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)
        
        return U, V, W
    
    U36, V36, W36 = get_rank36()
    
    # Now try to find lower rank solutions
    # Key insight: the problem decomposes into two 2x2x5 sub-problems (j=0,1 and j=2,3)
    # Each 2x2x5 has known optimal rank of 10, so total should be 20
    
    # Let's try rank 20 first with structural initialization
    def get_rank20_structural():
        """Initialize using the block structure: two 2x2x5 blocks"""
        R = 20
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Block 1: j in {0,1}, uses columns 0-9
        # Block 2: j in {2,3}, uses columns 10-19
        # For each block, we need a 2x2x5 decomposition of rank 10
        
        # Use Strassen-like for 2x2x5:
        # For 2x2 x 2x5, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # Strassen: M1 = (A00+A11)(B0+B1), M2 = A11(B1-B0), M3 = A00(B0), M4 = A01(B1), 
        # M5 = (A00+A01)(B1), M6 = (A10+A11)(B0), M7 = (A10+A11)(B0+B1)
        # But this gives 7*5=35 which is too much. 
        # Actually for 2x2x5, we need rank 10.
        # Standard: 2*2*5 = 20. Strassen on the 2x2 part: 7*5 = 35. That's worse!
        # So the optimal rank-10 must use a different structure.
        
        # Let me just use random initialization and optimize
        U_block = np.random.normal(0, 1.0, (4, 10))  # 2x2 = 4 entries
        V_block = np.random.normal(0, 1.0, (10, 10))  # 2x5 = 10 entries
        W_block = np.random.normal(0, 1.0, (10, 10))  # 5x2 = 10 entries
        
        # Fill block 1
        U[:, :10] = U_block
        V[:10, :10] = V_block  # j=0,1 -> indices 0-9
        W[:, :10] = W_block
        
        # Fill block 2
        U[4:, 10:] = U_block  # j=2,3 -> indices 4-7
        V[10:, 10:] = V_block  # j=2,3 -> indices 10-19
        W[:, 10:] = W_block
        
        return U, V, W
    
    best_result = None
    best_rank = 36
    
    # Try ranks from 20 upward
    for R in range(20, 37):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 30 if R <= 22 else 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Structural: block diagonal
                U_init, V_init, W_init = get_rank20_structural()
                if R != 20:
                    U_init = U_init[:, :R]
                    V_init = V_init[:, :R]
                    W_init = W_init[:, :R]
            elif attempt == 1:
                # Truncate from rank 36
                U_init = U36[:, :R]
                V_init = V36[:, :R]
                W_init = W36[:, :R]
            elif attempt == 2:
                # Random
                U_init = np.random.normal(0, 1.0, (n*m, R))
                V_init = np.random.normal(0, 1.0, (m*p, R))
                W_init = np.random.normal(0, 1.0, (p*n, R))
            elif attempt == 3:
                # Perturbed structural
                U_s, V_s, W_s = get_rank20_structural()
                U_init = U_s[:, :R] + np.random.normal(0, 0.3, (n*m, R))
                V_init = V_s[:, :R] + np.random.normal(0, 0.3, (m*p, R))
                W_init = W_s[:, :R] + np.random.normal(0, 0.3, (p*n, R))
            elif attempt == 4:
                # Block-structured with different random seeds
                U_init = np.zeros((n*m, R))
                V_init = np.zeros((m*p, R))
                W_init = np.zeros((p*n, R))
                # Try to fill with block structure
                for block in range(2):
                    col_start = block * (R // 2)
                    col_end = (block + 1) * (R // 2) if block < 1 else R
                    U_init[block*2: block*2+2, col_start:col_end] = np.random.normal(0, 1.0, (2, col_end-col_start))
                    V_init[block*2: block*2+2, col_start:col_end] = np.random.normal(0, 1.0, (2, col_end-col_start))
                    W_init[:, col_start:col_end] = np.random.normal(0, 1.0, (p*n, col_end-col_start))
            else:
                # Scaled random
                scale = 0.3 + 0.7 * np.random.random()
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residual, x0, args=(R,), method='lm', max_nfev=3000)
                
                if result.success or result.cost < 1e-8:
                    x_opt = result.x
                    size_U = n * m * R
                    size_V = m * p * R
                    size_W = p * n * R
                    U_opt = x_opt[0:size_U].reshape(n*m, R)
                    V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                    W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                    
                    if verify(U_opt, V_opt, W_opt, tol=1e-6):
                        # Polish
                        U_pol, V_pol, W_pol = polish(U_opt, V_opt, W_opt, R)
                        
                        # Try snapping
                        U_snap, V_snap, W_snap = snap(U_pol, V_pol, W_pol)
                        if verify(U_snap, V_snap, W_snap, tol=1e-9):
                            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        else:
                            U_snap2, V_snap2, W_snap2 = snap(U_opt, V_opt, W_opt)
                            if verify(U_snap2, V_snap2, W_snap2, tol=1e-9):
                                return {"U": U_snap2.tolist(), "V": V_snap2.tolist(), "W": W_snap2.tolist()}
                            else:
                                # Try with higher precision polish
                                U_pol2, V_pol2, W_pol2 = polish(U_opt, V_opt, W_opt, R, max_iter=100, tol=1e-14)
                                U_snap3, V_snap3, W_snap3 = snap(U_pol2, V_pol2, W_pol2)
                                if verify(U_snap3, V_snap3, W_snap3, tol=1e-9):
                                    return {"U": U_snap3.tolist(), "V": V_snap3.tolist(), "W": W_snap3.tolist()}
            except Exception:
                pass
    
    # Fallback to rank 36
    U_snap, V_snap, W_snap = snap(U36, V36, W36)
    if verify(U_snap, V_snap, W_snap, tol=1e-9):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}

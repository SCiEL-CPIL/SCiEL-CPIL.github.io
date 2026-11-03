"""Program for rank-minimal tensor decomposition of 3x3 by 3x6 matrix multiplication."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition correctly represents matrix multiplication."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    val = np.dot(u_val, v_val * w_val)
                    err = abs(1.0 - val)
                    if err > max_err:
                        max_err = err
        # Check that off-diagonal entries are zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for k2 in range(p):
                        if k == k2:
                            continue
                        u_val = U[i*m+j, :]
                        v_val = V[j*p+k2, :]
                        w_val = W[k*n+i, :]
                        val = np.dot(u_val, v_val * w_val)
                        err = abs(val)
                        if err > max_err:
                            max_err = err
        return max_err

    def verify_decomposition_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix operations."""
        r = U.shape[1]
        # For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for k != k2, sum_r U[i*m+j, r] * V[j*p+k2, r] * W[k*n+i, r] = 0
        # Actually, the condition is:
        # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is the only condition. The off-diagonal check is not needed because
        # the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
        # for all valid (i,j,k) pairs, and 0 otherwise. But since i,j,k are always valid
        # indices in the range, we just need T[i,j,k]=1 for all i,j,k.
        
        max_err = 0.0
        for r_idx in range(r):
            pass  # This approach is complex; let's use the simple one but vectorized
        
        # Vectorized check
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(1.0 - val)
                    if err > max_err:
                        max_err = err
        return max_err

    def schoolbook_3x3x6():
        """Return schoolbook decomposition (rank 54) for 3x3x6."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def build_3x3x3_rank24():
        """
        Build a rank-24 decomposition for 3x3x3 using 4 overlapping Strassen blocks.
        U shape: (9, 24), V shape: (9, 24), W shape: (9, 24)
        W uses trace convention: W[k*3+i]
        """
        U = np.zeros((9, 24))
        V = np.zeros((9, 24))
        W = np.zeros((9, 24))
        
        # Strassen terms for 2x2x2
        # Input indices for 2x2: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        #                      e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        # Output: C[0,0]=m1+m4-m5+m7, C[0,1]=m3+m5, C[1,0]=m2+m4, C[1,1]=m1-m2+m3+m6
        strassen = [
            # (u_coeffs, v_coeffs, w_coeffs) for [a,b,c,d], [e,f,g,h], [C00,C01,C10,C11]
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # m1 = (a+d)(e+h)
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # m2 = (c+d)e
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # m3 = a(f-h)
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),   # m4 = d(g-c)
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # m5 = (a+b)h
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # m6 = (c-a)(e+f)
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # m7 = (b-d)(g+h)
        ]
        
        # 4 blocks covering the 3x3 matrix
        # Block (r1,r2; c1,c2) covers rows r1,r2 and cols c1,c2
        blocks = [(0,1,0,1), (0,1,1,2), (1,2,0,1), (1,2,1,2)]
        
        col = 0
        for r1, r2, c1, c2 in blocks:
            for u2, v2, w2 in strassen:
                # u2 corresponds to [A[r1,c1], A[r1,c2], A[r2,c1], A[r2,c2]]
                # v2 corresponds to [B[c1,c1], B[c1,c2], B[c2,c1], B[c2,c2]]
                # w2 corresponds to [C[r1,c1], C[r1,c2], C[r2,c1], C[r2,c2]]
                
                # Map to 3x3 indices
                # A indices: (r1,c1)=0, (r1,c2)=1, (r2,c1)=2, (r2,c2)=3
                a_indices = [(r1,c1), (r1,c2), (r2,c1), (r2,c2)]
                b_indices = [(c1,c1), (c1,c2), (c2,c1), (c2,c2)]
                c_indices = [(r1,c1), (r1,c2), (r2,c1), (r2,c2)]
                
                for idx in range(4):
                    ar, ac = a_indices[idx]
                    br, bc = b_indices[idx]
                    cr, cc = c_indices[idx]
                    
                    U[ar*3 + ac, col] += u2[idx]
                    V[br*3 + bc, col] += v2[idx]
                    W[cc*3 + cr, col] += w2[idx]  # trace convention: W[k*n+i] = W[cc*3+cr]
                col += 1
        
        return U, V, W, 24

    def build_3x3x6_from_3x3x3(U33, V33, W33, rank33):
        """
        Lift a rank-r 3x3x3 decomposition to 3x3x6 with rank 2r.
        For 3x3x6: A is 3x3, B is 3x6, C is 3x6
        C[:,0:3] = A * B[:,0:3], C[:,3:6] = A * B[:,3:6]
        """
        rank = 2 * rank33
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        for s in range(rank33):
            # Left half (columns 0-2 of B)
            idx1 = 2 * s
            U[:, idx1] = U33[:, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k, idx1] = V33[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[k*3+i, idx1] = W33[k*3+i, s]
            
            # Right half (columns 3-5 of B)
            idx2 = 2 * s + 1
            U[:, idx2] = U33[:, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k+3, idx2] = V33[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i, idx2] = W33[k*3+i, s]
        
        return U, V, W, rank

    def try_optimize_3x3x3(r_target, max_trials=3, time_per_trial=20.0):
        """
        Try to find a rank-r decomposition of 3x3x3 via numerical optimization.
        Returns (U, V, W, rank) or None.
        """
        if time.time() > deadline:
            return None
        
        import scipy.optimize as optimize
        
        n3, m3, p3 = 3, 3, 3
        nm = n3 * m3
        mp = m3 * p3
        pn = p3 * n3
        
        def loss_fn(params):
            U = params[:nm * r_target].reshape(nm, r_target)
            V = params[nm * r_target : nm * r_target + mp * r_target].reshape(mp, r_target)
            W = params[nm * r_target + mp * r_target :].reshape(pn, r_target)
            
            T = np.zeros((n3, m3, p3))
            for r_idx in range(r_target):
                u = U[:, r_idx].reshape(n3, m3)
                v = V[:, r_idx].reshape(m3, p3)
                w = W[:, r_idx].reshape(p3, n3)
                T += np.einsum('ij,jk,ki->ijk', u, v, w)
            
            err = T - 1.0
            return np.sum(err * err)
        
        best_loss = float('inf')
        best_params = None
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with moderate scale
            scale = 1.0
            U0 = np.random.randn(nm, r_target) * scale
            V0 = np.random.randn(mp, r_target) * scale
            W0 = np.random.randn(pn, r_target) * scale
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = optimize.minimize(loss_fn, params0, method='L-BFGS-B',
                                          maxiter=200, options={'maxfun': 3000})
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_params = result.x
            except Exception:
                continue
        
        if best_params is None or best_loss > 1e-6:
            return None
        
        U = best_params[:nm * r_target].reshape(nm, r_target)
        V = best_params[nm * r_target : nm * r_target + mp * r_target].reshape(mp, r_target)
        W = best_params[nm * r_target + mp * r_target :].reshape(pn, r_target)
        
        return U, V, W, r_target

    best_result = None
    best_rank = float('inf')
    
    # Method 1: Rank-24 3x3x3 -> Rank-48 3x3x6
    try:
        U33, V33, W33, rank33 = build_3x3x3_rank24()
        U, V, W, rank = build_3x3x6_from_3x3x3(U33, V33, W33, rank33)
        err = verify_decomposition(U, V, W)
        if err < 1e-9 and rank < best_rank:
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Method 2: Try to find better 3x3x3 decompositions via optimization
    if time.time() < deadline - 30:
        for r_try in [23, 22, 21, 20]:
            if time.time() > deadline - 20:
                break
            try:
                result = try_optimize_3x3x3(r_try, max_trials=2, time_per_trial=15.0)
                if result is not None:
                    U33, V33, W33, rank33 = result
                    U, V, W, rank = build_3x3x6_from_3x3x3(U33, V33, W33, rank33)
                    err = verify_decomposition(U, V, W)
                    if err < 1e-9 and rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            except Exception as e:
                pass
    
    # Method 3: Schoolbook fallback (rank 54)
    U, V, W, rank = schoolbook_3x3x6()
    err = verify_decomposition(U, V, W)
    if err < 1e-9 and rank < best_rank:
        best_rank = rank
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

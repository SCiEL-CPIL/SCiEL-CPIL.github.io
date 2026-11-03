import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10
    
    start_time = time.time()
    time_limit = 140  # safety margin for 150s limit
    
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                target = 1.0 if (b == c and a == i and d == k) else 0.0
                                val = 0.0
                                for mu in range(R):
                                    val += U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu]
                                if abs(val - target) > 1e-9:
                                    return False
        return True
    
    def fast_verify(U, V, W):
        R = U.shape[1]
        # Build the full tensor and check
        # T[a,b,c,d,k,i] = sum_mu U[a*m+b,mu]*V[c*p+d,mu]*W[k*n+i,mu]
        # Should equal 1 if b==c, a==i, d==k else 0
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        # Check all constraints via einsum
        # T_abc = sum_d sum_mu U[ab,mu]*V[cd,mu]*W[dk,mu] should have structure
        # Simpler: check each (i,k) output
        for i in range(n):
            for k in range(p):
                # Output[i,k] = sum_j A[i,j]*B[j,k]
                # = sum_mu W[k*n+i,mu] * sum_j (sum_mu2 U[i*m+j,mu2]*W... 
                # Let's just do the full check
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                target = 1.0 if (b == c and a == i and d == k) else 0.0
                                val = np.dot(U[a*m+b], np.multiply(V[c*p+d], W[k*n+i]))
                                if abs(val - target) > 1e-9:
                                    return False
        return True
    
    # Strategy: Use block decomposition with Strassen in i (2x2) and 
    # FFT/polynomial method in j (size 4) and k (size 5)
    
    # Block 1: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # Use Strassen for the 2x2 block in (i, j_pairs)
    
    # Actually, let me try a direct construction:
    # Decompose into 7 Strassen-like terms for the 2x2x2 core,
    # then extend to handle the full 4x5
    
    # Method: Split j into two groups of 2, k into groups
    # Block (j0 in {0,1}, j1 in {2,3}), (k0 in {0,1,2}, k1 in {3,4})
    
    # For each (j_group, k_group) block of size 2x2:
    #   Use Strassen: 7 multiplications
    # 4 blocks * 7 = 28
    
    # But we can do better with polynomial methods on j
    
    # Let me try: evaluate A at points in j-dimension
    # A(x) = A[0] + A[1]*x + A[2]*x^2 + A[3]*x^3 (vector-valued)
    # B(y) = B[0] + B[1]*y + B[2]*y^2 + B[3]*y^3 + B[4]*y^4
    # Product has degree 7, needs 8 points
    
    # This gives rank 8 * (Strassen in i: 7) ... no, that's not right either
    
    # Let me just try the 4-block Strassen approach (rank 28) and verify
    # Then try to reduce
    
    def build_4block_strassen():
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        # Blocks: (i0, j0, k0) where j0 in {0,2}, k0 in {0,2}
        # Each block is 2x2x2 Strassen
        for i0 in [0]:  # n=2, so only i0=0
            for j0 in [0, 2]:
                for k0 in [0, 2]:
                    # Strassen for 2x2x2 at offset (i0, j0, k0)
                    # M1: (A[i0,j0]+A[i0+1,j0+1])(B[j0,k0]+B[j0+1,k0+1])
                    U[i0*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
                    V[j0*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
                    W[k0*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
                    col += 1
                    # M2: (A[i0+1,j0]+A[i0+1,j0+1])B[j0,k0]
                    U[(i0+1)*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
                    V[j0*p+k0, col] += 1
                    W[k0*n+(i0+1), col] += 1; W[(k0+1)*n+(i0+1), col] -= 1
                    col += 1
                    # M3: A[i0,j0](B[j0+1,k0]-B[j0+1,k0+1])
                    U[i0*m+j0, col] += 1
                    V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] -= 1
                    W[k0*n+i0, col] += 1; W[(k0+1)*n+i0, col] += 1
                    col += 1
                    # M4: A[i0+1,j0+1](B[j0,k0+1]-B[j0,k0])
                    U[(i0+1)*m+(j0+1), col] += 1
                    V[j0*p+(k0+1), col] += 1; V[j0*p+k0, col] -= 1
                    W[(k0+1)*n+i0, col] += 1; W[k0*n+i0, col] += 1
                    col += 1
                    # M5: (A[i0,j0]+A[i0,j0+1])B[j0+1,k0+1]
                    U[i0*m+j0, col] += 1; U[i0*m+(j0+1), col] += 1
                    V[(j0+1)*p+(k0+1), col] += 1
                    W[(k0+1)*n+i0, col] += 1; W[k0*n+i0, col] -= 1
                    col += 1
                    # M6: (A[i0+1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k0+1])
                    U[(i0+1)*m+j0, col] += 1; U[i0*m+j0, col] -= 1
                    V[j0*p+k0, col] += 1; V[j0*p+(k0+1), col] += 1
                    W[(k0+1)*n+(i0+1), col] += 1
                    col += 1
                    # M7: (A[i0,j0+1]-A[i0+1,j0+1])(B[j0+1,k0]+B[j0+1,k0+1])
                    U[i0*m+(j0+1), col] += 1; U[(i0+1)*m+(j0+1), col] -= 1
                    V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
                    W[k0*n+i0, col] += 1
                    col += 1
        
        # Now handle k=4 (the 5th column) separately
        # For k=4, we need A[i,:]*B[:,4] for i=0,1
        # This is 2x4x1 = 4 multiplications (schoolbook)
        for i in range(2):
            for j in range(4):
                U[i*m+j, col] = 1
                V[j*p+4, col] = 1
                W[4*n+i, col] = 1
                col += 1
        
        assert col == rank, f"col={col}, rank={rank}"
        return U, V, W
    
    U, V, W = build_4block_strassen()
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 28
    best_solution = (U.copy(), V.copy(), W.copy())
    
    # Try to reduce rank via optimization
    def residual_vector(x, R):
        size_U = nm * R
        size_V = mp * R
        Ux = x[0:size_U].reshape(nm, R)
        Vx = x[size_U:size_U + size_V].reshape(mp, R)
        Wx = x[size_U + size_V:].reshape(pn, R)
        
        # Build constraint arrays
        vals = []
        targets = []
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                target = 1.0 if (b == c and a == i and d == k) else 0.0
                                vals.append(np.dot(Ux[a*m+b], np.multiply(Vx[c*p+d], Wx[k*n+i])))
                                targets.append(target)
        return np.array(vals) - np.array(targets)
    
    # Precompute constraint indices for speed
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
    
    cons_U = np.array([x[0] for x in cons_idx])
    cons_V = np.array([x[1] for x in cons_idx])
    cons_W = np.array([x[2] for x in cons_idx])
    cons_t = np.array(cons_target)
    
    def residual_fast(x, R):
        size_U = nm * R
        size_V = mp * R
        Ux = x[0:size_U].reshape(nm, R)
        Vx = x[size_U:size_U + size_V].reshape(mp, R)
        Wx = x[size_U + size_V:].reshape(pn, R)
        
        U_rows = Ux[cons_U]  # (N, R)
        V_rows = Vx[cons_V]  # (N, R)
        W_rows = Wx[cons_W]  # (N, R)
        
        vals = np.einsum('im,im,im->i', U_rows, V_rows, W_rows)
        return vals - cons_t
    
    import scipy.optimize as opt
    
    for R in range(27, 20, -1):
        if time.time() - start_time > time_limit:
            break
        
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0:
                U_init = U[:, :R]
                V_init = V[:, :R]
                W_init = W[:, :R]
            else:
                U_init = U[:, :R] + np.random.normal(0, 0.05, (nm, R))
                V_init = V[:, :R] + np.random.normal(0, 0.05, (mp, R))
                W_init = W[:, :R] + np.random.normal(0, 0.05, (pn, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residual_fast, x0, args=(R,), method='lm', max_nfev=5000)
                if np.max(np.abs(result.fun)) < 1e-8:
                    x_opt = result.x
                    size_U = nm * R
                    size_V = mp * R
                    U_opt = x_opt[0:size_U].reshape(nm, R)
                    V_opt = x_opt[size_U:size_U + size_V].reshape(mp, R)
                    W_opt = x_opt[size_U + size_V:].reshape(pn, R)
                    
                    # Snap to half-integers
                    U_s = np.round(U_opt * 2) / 2.0
                    V_s = np.round(V_opt * 2) / 2.0
                    W_s = np.round(W_opt * 2) / 2.0
                    
                    if verify(U_s, V_s, W_s):
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                    elif verify(U_opt, V_opt, W_opt):
                        best_rank = R
                        best_solution = (U_opt, V_opt, W_opt)
                        break
            except Exception:
                pass
    
    if best_rank < 28:
        return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

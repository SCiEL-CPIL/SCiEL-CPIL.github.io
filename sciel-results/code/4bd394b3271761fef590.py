import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 145s deadline, 5s safety margin before 150s kill

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    # Target tensor: T[a,b,c] = 1 where a=2i+j, b=6j+k, c=2k+i
    T_exp = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_exp[a, b, c] = 1.0

    # Strassen 2x2x2 steps
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]

    def build_strassen_block(g, U, V, W, col_offset):
        """Add a Strassen block for k=2g, 2g+1 at column offset."""
        for s in range(7):
            r = col_offset + s
            step = strassen_steps[s]
            k0 = 2 * g
            k1 = 2 * g + 1
            b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
            c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
            for l in range(4):
                U[l, r] = step['u'][l]
                V[b_map[l], r] = step['v'][l]
                W[c_map[l], r] = step['w'][l]

    def try_round_and_verify(U, V, W, T_target, scales=(1, 2, 4, 8)):
        """Try rounding to integers/half-integers at various scales."""
        for scale in scales:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_calc, T_target, atol=1e-7):
                return U_r, V_r, W_r
        # If no rounding works but values are already close
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_calc, T_target, atol=1e-7):
            return U, V, W
        return None

    def polish(U, V, W, T_target, max_iter=200, tol=1e-12):
        """Gauss-Newton polish: minimize ||sum_r u_r v_r w_r^T - T||^2.
        Uses alternating least squares per rank term."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            old_U, old_V, old_W = U.copy(), V.copy(), W.copy()
            
            # Solve for U: for each r, minimize ||T - sum_s U_s V_s W_s^T||
            # Fix V, W; solve for U via linear system
            # T = sum_r U[:,r] (V[:,r] (W[:,r]^T))
            # For each (a,b,c): T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            
            # Build system for U:
            # For each r: U[:,r] is unknown (4 values)
            # Residual: R[a,b,c] = T[a,b,c] - sum_s U[a,s] V[b,s] W[c,s]
            # d(R)/d(U[a,r]) = -V[b,r] W[c,r]
            # Normal equations: sum_{b,c} V[b,r]W[c,r] V[b,r']W[c,r'] U[a,r'] ... 
            # This couples a and r. Let's do ALS instead.
            
            # ALS: fix V,W; for each a, solve for U[a,:]
            # T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
            # For fixed a: T[a,:,:] = U[a,:] (V (W^T))  ->  T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is: T[a] = sum_r U[a,r] (v_r w_r^T) where v_r = V[:,r], w_r = W[:,r]
            # So T[a] is a matrix (b,c), and we write it as sum_r U[a,r] (v_r w_r^T)
            # This is a rank-r decomposition of T[a] with fixed factors v_r, w_r.
            # For each a: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Flatten: T[a] (size mp*pn) = sum_r U[a,r] (V[:,r] * W[:,r]) (elementwise, size mp*pn)
            # So: T_flat[a] = U[a,:] @ M[:,r] where M[b*pn+c, r] = V[b,r]*W[c,r]
            
            M = np.zeros((mp * pn, rank))
            for r in range(rank):
                M[:, r] = (V[:, r] * W[:, r]).flatten()
            
            # T_flat[a] = U[a,:] @ M  (size pn*mp... wait)
            # T[a,b,c]: a fixed, b in [0,mp), c in [0,pn)
            # T_flat[a, b*pn+c] = sum_r U[a,r] V[b,r] W[c,r]
            # So T_flat[a] (vector of length mp*pn) = U[a,:] (row vector, length rank) @ M (mp*pn x rank)? 
            # No: sum_r U[a,r] M[b*pn+c, r] = (U[a,:] @ M)[b*pn+c]
            # So T_flat[a] = U[a,:] @ M  -- but U[a,:] is 1xrank, M is (mp*pn)xrank, product is 1x(mp*pn)
            # So T_flat[a] = U[a,:] @ M  where T_flat[a] is length mp*pn
            
            T_flat = T_target.flatten()  # size nm*mp*pn, indexed [a*mp*pn + b*pn + c]
            for a in range(nm):
                T_a = T_flat[a*mp*pn:(a+1)*mp*pn]
                # Solve: U[a,:] @ M = T_a  ->  M^T U[a,:]^T = T_a^T
                # U[a,:] is 1xrank. M^T is rank x (mp*pn).
                # M^T U[a,:]^T = T_a^T
                try:
                    U[a, :] = np.linalg.lstsq(M.T, T_a, rcond=None)[0]
                except:
                    pass
            
            # Now solve for V: fix U, W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed b: T[:,b,:] = sum_r V[b,r] (U[:,r] W[:,r]^T)
            # T_flat[b] (size nm*pn) = V[b,:] @ N where N[a*pn+c, r] = U[a,r]*W[c,r]
            N = np.zeros((nm * pn, rank))
            for r in range(rank):
                N[:, r] = (U[:, r] * W[:, r]).flatten()
            
            for b in range(mp):
                T_b = T_target[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(N.T, T_b, rcond=None)[0]
                except:
                    pass
            
            # Now solve for W: fix U, V
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed c: T[:,:,c] = sum_r W[c,r] (U[:,r] V[:,r]^T)
            # T_flat[c] (size nm*mp) = W[c,:] @ P where P[a*mp+b, r] = U[a,r]*V[b,r]
            P = np.zeros((nm * mp, rank))
            for r in range(rank):
                P[:, r] = (U[:, r] * V[:, r]).flatten()
            
            for c in range(pn):
                T_c = T_target[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(P.T, T_c, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T_target)
            if err < tol:
                break
            if np.linalg.norm(U - old_U) + np.linalg.norm(V - old_V) + np.linalg.norm(W - old_W) < 1e-14:
                break
        
        return U, V, W

    def try_rank(R, T_target, time_budget=30.0, n_restarts=10, seed=42):
        """Try to find rank-R decomposition via optimization + polish."""
        np.random.seed(seed)
        best_err = np.inf
        best_result = None
        
        def objective(params):
            U = params[:R*nm].reshape(R, nm)
            V = params[R*nm:R*nm+R*mp].reshape(R, mp)
            W = params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            res = (T_calc - T_target).flatten()
            return res
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE or time.time() - start_time > time_budget + 60:
                break
            
            # Random init
            x0 = np.random.randn(R * (nm + mp + pn)) * 0.5
            
            try:
                sol = opt.least_squares(objective, x0, method='lm', max_nfev=1500)
                if sol.cost < 1e-8:
                    U = sol.x[:R*nm].reshape(R, nm)
                    V = sol.x[R*nm:R*nm+R*mp].reshape(R, mp)
                    W = sol.x[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn)
                    
                    # Polish
                    U, V, W = polish(U, V, W, T_target, max_iter=100)
                    
                    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(T_calc - T_target)
                    
                    if err < best_err:
                        best_err = err
                        best_result = (U, V, W)
                    
                    if err < 1e-10:
                        rounded = try_round_and_verify(U, V, W, T_target)
                        if rounded is not None:
                            return rounded
                        return (U, V, W)
            except Exception:
                pass
        
        return best_result

    # Strategy 1: Try rank 18 (6 Strassen + 12 for remaining)
    # Actually, let's try to find a decomposition for the full tensor directly.
    # Try rank 19, 18, 17
    
    for target_rank in [19, 18, 17]:
        if time.time() > DEADLINE - 20:
            break
        result = try_rank(target_rank, T_exp, time_budget=25, n_restarts=5, seed=target_rank)
        if result is not None:
            U, V, W = result
            rounded = try_round_and_verify(U, V, W, T_exp)
            if rounded is not None:
                U, V, W = rounded
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_calc, T_exp, atol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strategy 2: Block decomposition - Strassen for first 2 cols + rank 13 for rest
    # Build Strassen for k=0,1
    U_s = np.zeros((nm, 7))
    V_s = np.zeros((mp, 7))
    W_s = np.zeros((pn, 7))
    build_strassen_block(0, U_s, V_s, W_s, 0)
    
    # Local tensor for k=2,3,4,5
    T_local = np.zeros((nm, 8, 8))
    for i in range(n):
        for j in range(m):
            for kp in range(4):
                a = i * m + j
                b = j * 4 + kp  # local b
                c = i * 4 + kp  # local c
                T_local[a, b, c] = 1.0
    
    def try_rank13_local(time_budget=40.0):
        """Find rank-13 decomposition for T_local (4x8x8)."""
        R = 13
        np.random.seed(77)
        
        def objective(params):
            U = params[:R*nm].reshape(R, nm)
            V = params[R*nm:R*nm+R*8].reshape(R, 8)
            W = params[R*nm+R*8:R*nm+R*8+R*8].reshape(R, 8)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_calc - T_local).flatten()
        
        for restart in range(8):
            if time.time() > DEADLINE:
                break
            x0 = np.random.randn(R * (nm + 8 + 8)) * 0.5
            try:
                sol = opt.least_squares(objective, x0, method='lm', max_nfev=2000)
                if sol.cost < 1e-9:
                    U = sol.x[:R*nm].reshape(R, nm)
                    V = sol.x[R*nm:R*nm+R*8].reshape(R, 8)
                    W = sol.x[R*nm+R*8:R*nm+R*8+R*8].reshape(R, 8)
                    
                    # Polish
                    # ALS for local
                    for it in range(50):
                        M = np.zeros((8*8, R))
                        for r in range(R):
                            M[:, r] = (V[:, r] * W[:, r]).flatten()
                        for a in range(nm):
                            T_a = T_local[a, :, :].flatten()
                            try:
                                U[a, :] = np.linalg.lstsq(M.T, T_a, rcond=None)[0]
                            except:
                                pass
                        N = np.zeros((nm*8, R))
                        for r in range(R):
                            N[:, r] = (U[:, r] * W[:, r]).flatten()
                        for b in range(8):
                            T_b = T_local[:, b, :].flatten()
                            try:
                                V[b, :] = np.linalg.lstsq(N.T, T_b, rcond=None)[0]
                            except:
                                pass
                        P = np.zeros((nm*8, R))
                        for r in range(R):
                            P[:, r] = (U[:, r] * V[:, r]).flatten()
                        for c in range(8):
                            T_c = T_local[:, :, c].flatten()
                            try:
                                W[c, :] = np.linalg.lstsq(P.T, T_c, rcond=None)[0]
                            except:
                                pass
                        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                        if np.linalg.norm(T_calc - T_local) < 1e-12:
                            break
                    
                    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.allclose(T_calc, T_local, atol=1e-8):
                        # Round
                        for scale in [1, 2, 4]:
                            U_r = np.round(U * scale) / scale
                            V_r = np.round(V * scale) / scale
                            W_r = np.round(W * scale) / scale
                            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                            if np.allclose(T_check, T_local, atol=1e-7):
                                return U_r, V_r, W_r
                        return U, V, W
            except Exception:
                pass
        return None
    
    result13 = try_rank13_local()
    if result13 is not None:
        U13, V13, W13 = result13
        # Assemble full rank-20
        U_full = np.zeros((nm, 20))
        V_full = np.zeros((mp, 20))
        W_full = np.zeros((pn, 20))
        
        U_full[:, :7] = U_s
        V_full[:, :7] = V_s
        W_full[:, :7] = W_s
        U_full[:, 7:20] = U13
        
        # Map V13 (local b = j*4+kp) to global V (b = j*6+k, k=2+kp)
        for r in range(13):
            for lb in range(8):
                val = V13[lb, r]
                if val != 0:
                    j = lb // 4
                    kp = lb % 4
                    k = 2 + kp
                    gb = j * 6 + k
                    V_full[gb, 7+r] = val
        
        # Map W13 (local c = i*4+kp) to global W (c = k*2+i, k=2+kp)
        for r in range(13):
            for lc in range(8):
                val = W13[lc, r]
                if val != 0:
                    i = lc // 4
                    kp = lc % 4
                    k = 2 + kp
                    gc = k * 2 + i
                    W_full[gc, 7+r] = val
        
        T_full = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        if np.allclose(T_full, T_exp, atol=1e-6):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: Rank 21 via 3 Strassen blocks
    U = np.zeros((nm, 21))
    V = np.zeros((mp, 21))
    W = np.zeros((pn, 21))
    for g in range(3):
        build_strassen_block(g, U, V, W, 7 * g)
    
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_check, T_exp, atol=1e-6):
        # Should not happen, but safety
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

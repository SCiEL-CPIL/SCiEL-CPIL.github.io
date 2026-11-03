import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def verify_fast(U, V, W):
        T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(T_check - T))

    def polish(U, V, W, max_iters=200, tol=1e-12):
        """High-precision polish pass to converge to exact solution."""
        U = U.copy().astype(np.float64)
        V = V.copy().astype(np.float64)
        W = W.copy().astype(np.float64)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Solve for U: for each i, U[i,:] should satisfy sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
            # Fix V,W and solve linear system for U
            M_U = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, -1)
            T_flat = T.reshape(nm, mp * pn)
            try:
                U_new = T_flat @ np.linalg.lstsq(M_U.T, rcond=None)[0]
            except:
                break
            
            # Solve for V
            M_V = np.einsum('ir,kr->ikr', U_new, W).reshape(nm * pn, -1)
            T_flat_v = T.reshape(mp, nm * pn)
            try:
                V_new = T_flat_v @ np.linalg.lstsq(M_V.T, rcond=None)[0]
            except:
                break
            
            # Solve for W
            M_W = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, -1)
            T_flat_w = T.reshape(pn, nm * mp)
            try:
                W_new = T_flat_w @ np.linalg.lstsq(M_W.T, rcond=None)[0]
            except:
                break
            
            err = verify_fast(U_new, V_new, W_new)
            if err < tol:
                U, V, W = U_new, V_new, W_new
                break
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W, verify_fast(U, V, W)

    def try_round(U, V, W, step=2):
        """Try rounding to multiples of 1/step and verify."""
        U_r = np.round(U * step) / step
        V_r = np.round(V * step) / step
        W_r = np.round(W * step) / step
        
        err = verify_fast(U_r, V_r, W_r)
        if err < 1e-9:
            return U_r, V_r, W_r, err
        return None, None, None, err

    # Rank-55 decomposition: j=0 separately + Strassen for j=1,2
    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank

    # Alternative: schoolbook rank-60
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    U_best, V_best, W_best, rank_best = rank55_decomposition()
    err = verify(U_best, V_best, W_best)
    
    # Try to reduce rank using improved ALS with multiple strategies
    best_score = rank_best
    best_U, best_V, best_W = U_best, V_best, W_best
    
    # Strategy 1: ALS from rank-55 initialization, trying lower ranks
    for r in range(rank_best - 1, 40, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_restarts = 15
        
        for trial in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 7919 + r * 131)
            
            # Initialize near rank-55 solution
            U_c = U_best[:, :r].copy()
            V_c = V_best[:, :r].copy()
            W_c = W_best[:, :r].copy()
            
            # Add noise to escape local minima
            noise_scale = 0.05
            U_c += noise_scale * np.random.randn(nm, r)
            V_c += noise_scale * np.random.randn(mp, r)
            W_c += noise_scale * np.random.randn(pn, r)
            
            # ALS iterations
            converged = False
            for it in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U
                M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
                try:
                    U_c = T.reshape(nm, mp * pn) @ np.linalg.lstsq(M.T, rcond=None)[0]
                except:
                    break
                
                # Solve for V
                N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
                try:
                    V_c = T.reshape(mp, nm * pn) @ np.linalg.lstsq(N.T, rcond=None)[0]
                except:
                    break
                
                # Solve for W
                P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
                try:
                    W_c = T.reshape(pn, nm * mp) @ np.linalg.lstsq(P.T, rcond=None)[0]
                except:
                    break
                
                err = verify_fast(U_c, V_c, W_c)
                if err < 1e-8:
                    converged = True
                    break
            
            if not converged:
                continue
            
            # Polish
            U_p, V_p, W_p, err_p = polish(U_c, V_c, W_c, max_iters=100)
            
            if err_p < 1e-10:
                # Try rounding
                U_r, V_r, W_r, err_r = try_round(U_p, V_p, W_p, step=2)
                if U_r is not None:
                    best_U, best_V, best_W, best_score = U_r, V_r, W_r, r
                    found = True
                    break
                else:
                    # Check if unrounded is acceptable
                    if err_p < 1e-10:
                        best_U, best_V, best_W, best_score = U_p, V_p, W_p, r
                        found = True
                        break
        
        if found:
            break
    
    # Strategy 2: Random initialization for various ranks
    if time.time() < deadline - 20:
        for r in range(50, 40, -1):
            if time.time() > deadline - 15:
                break
            
            found = False
            max_restarts = 10
            
            for trial in range(max_restarts):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(trial * 104729 + r)
                
                U_c = np.random.randn(nm, r) * 0.5
                V_c = np.random.randn(mp, r) * 0.5
                W_c = np.random.randn(pn, r) * 0.5
                
                for it in range(25):
                    if time.time() > deadline - 10:
                        break
                    
                    M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
                    try:
                        U_c = T.reshape(nm, mp * pn) @ np.linalg.lstsq(M.T, rcond=None)[0]
                    except:
                        break
                    
                    N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
                    try:
                        V_c = T.reshape(mp, nm * pn) @ np.linalg.lstsq(N.T, rcond=None)[0]
                    except:
                        break
                    
                    P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
                    try:
                        W_c = T.reshape(pn, nm * mp) @ np.linalg.lstsq(P.T, rcond=None)[0]
                    except:
                        break
                    
                    err = verify_fast(U_c, V_c, W_c)
                    if err < 1e-8:
                        break
                
                if time.time() > deadline - 10:
                    break
                
                U_p, V_p, W_p, err_p = polish(U_c, V_c, W_c, max_iters=50)
                
                if err_p < 1e-10:
                    U_r, V_r, W_r, err_r = try_round(U_p, V_p, W_p, step=2)
                    if U_r is not None and r < best_score:
                        best_U, best_V, best_W, best_score = U_r, V_r, W_r, r
                        found = True
                        break
                    elif err_p < 1e-10 and r < best_score:
                        best_U, best_V, best_W, best_score = U_p, V_p, W_p, r
                        found = True
                        break
            
            if found:
                break
    
    # Final verification and fallback
    final_err = verify(best_U, best_V, best_W)
    if final_err > 1e-8:
        U_best, V_best, W_best, rank_best = schoolbook()
        final_err = verify(U_best, V_best, W_best)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

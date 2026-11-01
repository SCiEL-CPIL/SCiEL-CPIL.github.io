import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Build target tensor
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    def verify(U, V, W):
        """Verify exact decomposition against full tensor."""
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        """Snap to nearest 0.5 and verify."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def als_decompose(rank, max_iter=200, tol=1e-12):
        """Alternating Least Squares for tensor decomposition."""
        U = np.random.randn(U_size, rank)
        V = np.random.randn(V_size, rank)
        W = np.random.randn(W_size, rank)
        
        prev_cost = np.inf
        for it in range(max_iter):
            # Update U: for each u, T[u,:,:] = sum_c U[u,c] * (V⊗W)[:,c]
            # M (V*W, rank): M[(j,k),c] = V[j,c]*W[k,c]
            M = np.einsum('jc,kc->jkc', V, W).reshape(V_size * W_size, rank)
            try:
                U_new = np.linalg.lstsq(M, T.reshape(U_size, -1).T, rcond=None)[0].T
            except:
                return None

            # Update V: for each v, T[:,v,:] = sum_c V[v,c] * (U⊗W)[:,c]
            N = np.einsum('ic,kc->ikc', U_new, W).reshape(U_size * W_size, rank)
            try:
                V_new = np.linalg.lstsq(N, T.reshape(-1, V_size).T, rcond=None)[0].T
            except:
                return None

            # Update W: for each w, T[:,:,w] = sum_c W[w,c] * (U⊗V)[:,c]
            P = np.einsum('ic,jc->ijc', U_new, V_new).reshape(U_size * V_size, rank)
            try:
                W_new = np.linalg.lstsq(P, T.reshape(-1, W_size).T, rcond=None)[0].T
            except:
                return None

            U, V, W = U_new, V_new, W_new
            
            # Compute cost
            T_calc = np.zeros((U_size, V_size, W_size))
            for c in range(rank):
                T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            cost = np.sum((T_calc - T) ** 2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost

        # Final verification
        result = snap_and_verify(U, V, W)
        if result:
            return result
        # Also check without snapping in case it's already close to integers
        if verify(U, V, W):
            return U, V, W
        return None

    def least_squares_solve(rank, max_nfev=500):
        """Solve using scipy least_squares on full tensor."""
        total_vars = rank * (U_size + V_size + W_size)
        
        def residual(params):
            U = params[:rank*U_size].reshape(U_size, rank)
            V = params[rank*U_size:rank*U_size+rank*V_size].reshape(V_size, rank)
            W = params[rank*U_size+rank*V_size:].reshape(W_size, rank)
            
            T_calc = np.zeros((U_size, V_size, W_size))
            for c in range(rank):
                T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            
            return (T_calc - T).flatten()
        
        # Random init
        params = np.random.randn(total_vars) * 0.5
        
        try:
            res = least_squares(residual, params, method='trf', max_nfev=max_nfev, 
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if res.cost < 1e-10:
                U_opt = res.x[:rank*U_size].reshape(U_size, rank)
                V_opt = res.x[rank*U_size:rank*U_size+rank*V_size].reshape(V_size, rank)
                W_opt = res.x[rank*U_size+rank*V_size:].reshape(W_size, rank)
                
                result = snap_and_verify(U_opt, V_opt, W_opt)
                if result:
                    return result
                if verify(U_opt, V_opt, W_opt):
                    return U_opt, V_opt, W_opt
        except:
            pass
        return None

    # Try ranks from low to high
    # Known lower bound: for 2x3x3, rank is at least 10 (based on literature)
    # Try from 10 upward
    for rank in range(10, 19):
        if time.time() > deadline:
            break
        
        # Try ALS first (faster per iteration)
        max_als_restarts = 30 if rank <= 12 else 15
        for restart in range(max_als_restarts):
            if time.time() > deadline:
                break
            result = als_decompose(rank)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # Try least_squares as backup
        if time.time() > deadline:
            break
        max_ls_restarts = 10 if rank <= 12 else 5
        for restart in range(max_ls_restarts):
            if time.time() > deadline:
                break
            result = least_squares_solve(rank)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = np.zeros((U_size, rank_school))
    V_school = np.zeros((V_size, rank_school))
    W_school = np.zeros((W_size, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

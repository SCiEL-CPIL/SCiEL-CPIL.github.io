import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target multiplication tensor T[u, v, w]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def check_error(U, V, W):
        """Compute ||sum_r U[u,r]V[v,r]W[w,r] - T||"""
        Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(Rec - T)
    
    def optimize(U, V, W, max_iter=100, tol=1e-8, start_time=0, deadline=0):
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        R = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline - 5: # Safety margin
                break
                
            # Update W: for each w, solve A_w @ W[w,:] = T[:,:,w].reshape(-1)
            # A_w[u*mp+v, r] = U[u,r] * V[v,r]
            P_UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (nm, mp, R)
            A_W = P_UV.reshape(nm * mp, R)
            
            W_new = np.zeros((pn, R), dtype=np.float64)
            for w in range(pn):
                b = T[:, :, w].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                    W_new[w] = sol
                except np.linalg.LinAlgError:
                    W_new[w] = 0.0
            W = W_new
            
            # Update V: for each v, solve A_v @ V[v,:] = T[:,v,:].reshape(-1)
            # A_v[u*pn+w, r] = U[u,r] * W[w,r]
            P_UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (nm, pn, R)
            A_V = P_UW.reshape(nm * pn, R)
            
            V_new = np.zeros((mp, R), dtype=np.float64)
            for v in range(mp):
                b = T[:, v, :].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                    V_new[v] = sol
                except np.linalg.LinAlgError:
                    V_new[v] = 0.0
            V = V_new
            
            # Update U: for each u, solve A_u @ U[u,:] = T[u,:,:].reshape(-1)
            # A_u[v*pn+w, r] = V[v,r] * W[w,r]
            P_VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (mp, pn, R)
            A_U = P_VW.reshape(mp * pn, R)
            
            U_new = np.zeros((nm, R), dtype=np.float64)
            for u in range(nm):
                b = T[u, :, :].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                    U_new[u] = sol
                except np.linalg.LinAlgError:
                    U_new[u] = 0.0
            U = U_new
            
            # Check error
            err = check_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
                
        return best_U, best_V, best_W, best_err
    
    def polish(U, V, W, max_iter=20, tol=1e-10, deadline=0):
        """High-precision polish pass"""
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        R = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
                
            # Update W
            P_UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            A_W = P_UV.reshape(nm * mp, R)
            W_new = np.zeros((pn, R), dtype=np.float64)
            for w in range(pn):
                b = T[:, :, w].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                    W_new[w] = sol
                except np.linalg.LinAlgError:
                    W_new[w] = 0.0
            W = W_new
            
            # Update V
            P_UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
            A_V = P_UW.reshape(nm * pn, R)
            V_new = np.zeros((mp, R), dtype=np.float64)
            for v in range(mp):
                b = T[:, v, :].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                    V_new[v] = sol
                except np.linalg.LinAlgError:
                    V_new[v] = 0.0
            V = V_new
            
            # Update U
            P_VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
            A_U = P_VW.reshape(mp * pn, R)
            U_new = np.zeros((nm, R), dtype=np.float64)
            for u in range(nm):
                b = T[u, :, :].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                    U_new[u] = sol
                except np.linalg.LinAlgError:
                    U_new[u] = 0.0
            U = U_new
            
            err = check_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
                
        return best_U, best_V, best_W, best_err
    
    deadline = time.time() + 140  # 10s safety margin
    
    best_solution = None
    best_rank = n * m * p  # Fallback to schoolbook rank 48
    best_err = float('inf')
    
    # Try different rank targets from low to high
    rank_targets = [44, 45, 46, 47]
    
    for R_target in rank_targets:
        if time.time() > deadline - 20:
            break
            
        max_restarts = 30 if R_target == 44 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Different initialization strategies
            if restart % 3 == 0:
                # Random initialization
                U = np.random.randn(nm, R_target)
                V = np.random.randn(mp, R_target)
                W = np.random.randn(pn, R_target)
            elif restart % 3 == 1:
                # Sparse initialization (based on finite field insight)
                U = np.zeros((nm, R_target))
                V = np.zeros((mp, R_target))
                W = np.zeros((pn, R_target))
                for r in range(R_target):
                    # Random sparse pattern
                    u_idx = np.random.choice(nm, size=3, replace=False)
                    v_idx = np.random.choice(mp, size=3, replace=False)
                    w_idx = np.random.choice(pn, size=3, replace=False)
                    U[u_idx, r] = np.random.choice([-1, 1], size=3)
                    V[v_idx, r] = np.random.choice([-1, 1], size=3)
                    W[w_idx, r] = np.random.choice([-1, 1], size=3)
            else:
                # Structured initialization based on tensor structure
                U = np.zeros((nm, R_target))
                V = np.zeros((mp, R_target))
                W = np.zeros((pn, R_target))
                
                # Try to match the schoolbook pattern but with fewer terms
                for r in range(R_target):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    
                    U[u, r] = 1.0
                    V[v, r] = 1.0
                    W[w, r] = 1.0
                    
                    # Add some random perturbation
                    if r < R_target - 5:
                        U[np.random.randint(nm), r] += np.random.randn() * 0.1
                        V[np.random.randint(mp), r] += np.random.randn() * 0.1
                        W[np.random.randint(pn), r] += np.random.randn() * 0.1
            
            # Run optimization
            U, V, W, err = optimize(U, V, W, max_iter=40, tol=1e-8, start_time=time.time(), deadline=deadline)
            
            # If we get close, do a polish pass
            if err < 1e-4:
                U, V, W, err = polish(U, V, W, max_iter=20, tol=1e-10, deadline=deadline)
            
            if err < best_err:
                best_err = err
                best_solution = (U, V, W)
                best_rank = R_target
            
            if err < 1e-6:
                break
        
        if best_err < 1e-6:
            break
    
    if best_solution is None or best_err > 1e-5:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        best_solution = (U, V, W)
        best_rank = rank

    U, V, W = best_solution
    
    # Convert to list of lists
    def to_list(matrix):
        L = []
        for row in matrix:
            r = []
            for val in row:
                # Check if close to half-integer
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) < 1e-6:
                    r.append(nearest)
                else:
                    r.append(float(val))
            L.append(r)
        return L

    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }

import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor support
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                support.append((u, v, w))
    
    target_rank = 12
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    start_time = time.time()
    deadline = 140.0
    
    def build_target():
        T = np.zeros((nm, mp, pn))
        for (u, v, w) in support:
            T[u, v, w] = 1.0
        return T
    
    T = build_target()
    
    def compute_tensor(U, V, W):
        R = U.shape[1]
        C = np.zeros((nm, mp, pn))
        for c in range(R):
            C += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        return C
    
    def verify_exact(U, V, W, atol=1e-8):
        """Check if U,V,W exactly decompose T with half-integer entries."""
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=atol):
                return False
        C = compute_tensor(U, V, W)
        return np.allclose(C, T, atol=atol)
    
    def snap_half(x):
        return np.round(x * 2) / 2.0
    
    def solve_als(U, V, W, T, max_iters=100, tol=1e-10):
        """
        Alternating Least Squares for tensor decomposition.
        """
        R = U.shape[1]
        # Preallocate
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        for it in range(max_iters):
            # 1. Solve for U
            # T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
            # For fixed u, T[u, :, :] = sum_r U[u, r] * (V[:, r] outer W[:, r])
            # Let M_r = V[:, r] outer W[:, r]
            # T_u_vec = sum_r U[u, r] * vec(M_r)
            # A_U is matrix with columns vec(M_r)
            
            # Construct A_U (mp*pn x R)
            # M_r is mp x pn
            # vec(M_r) is mp*pn
            
            A_U = np.zeros((mp * pn, R))
            for r in range(R):
                M_r = np.outer(V_curr[:, r], W_curr[:, r])
                A_U[:, r] = M_r.flatten()
            
            # Solve for each u
            # T_u_vec is mp*pn
            # U_curr[u, :] = (A_U^T A_U)^-1 A_U^T T_u_vec
            
            try:
                # Use lstsq for stability
                # We can solve all u at once if we stack T
                # T_flat is nm x (mp*pn)
                T_flat = T.reshape(nm, mp * pn)
                # U_curr is nm x R
                # T_flat = U_curr @ A_U^T
                # U_curr = T_flat @ (A_U^T)^+
                U_new, _, _, _ = np.linalg.lstsq(A_U.T, T_flat.T, rcond=None)
                U_curr = U_new.T
            except Exception:
                pass
            
            # 2. Solve for V
            # T[u, v, w] = sum_r V[v, r] * (U[u, r] outer W[w, r])
            # N_r = U[:, r] outer W[:, r]
            A_V = np.zeros((nm * pn, R))
            for r in range(R):
                N_r = np.outer(U_curr[:, r], W_curr[:, r])
                A_V[:, r] = N_r.flatten()
            
            T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V_new, _, _, _ = np.linalg.lstsq(A_V.T, T_flat_v.T, rcond=None)
                V_curr = V_new.T
            except Exception:
                pass
            
            # 3. Solve for W
            # T[u, v, w] = sum_r W[w, r] * (U[u, r] outer V[v, r])
            # P_r = U[:, r] outer V[:, r]
            A_W = np.zeros((nm * mp, R))
            for r in range(R):
                P_r = np.outer(U_curr[:, r], V_curr[:, r])
                A_W[:, r] = P_r.flatten()
            
            T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W_new, _, _, _ = np.linalg.lstsq(A_W.T, T_flat_w.T, rcond=None)
                W_curr = W_new.T
            except Exception:
                pass
            
            # Check convergence
            C = compute_tensor(U_curr, V_curr, W_curr)
            err = np.linalg.norm(C - T)
            if err < tol:
                break
        
        return U_curr, V_curr, W_curr, err

    def try_rank(rank):
        if time.time() > deadline:
            return None
        
        best_sol = None
        best_err = float('inf')
        
        # Try multiple random initializations
        num_trials = 50
        for t in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random sparse initialization
            # Each rank-1 term should ideally cover a small part of the support
            # Let's try initializing with random values in a sparse pattern
            # Or just random dense with small magnitude
            
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Run ALS
            U_sol, V_sol, W_sol, err = solve_als(U, V, W, T, max_iters=200)
            
            if err < best_err:
                best_err = err
                best_sol = (U_sol, V_sol, W_sol)
            
            # Try to snap and verify
            if err < 1e-4:
                U_snap = snap_half(U_sol)
                V_snap = snap_half(V_sol)
                W_snap = snap_half(W_sol)
                
                if verify_exact(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        # If we found a good solution but didn't verify, try a final polish
        if best_sol is not None and best_err < 1e-2:
            U_sol, V_sol, W_sol = best_sol
            # Run more iterations on the best solution
            U_pol, V_pol, W_pol, err_pol = solve_als(U_sol, V_sol, W_sol, T, max_iters=500)
            if err_pol < 1e-6:
                U_snap = snap_half(U_pol)
                V_snap = snap_half(V_pol)
                W_snap = snap_half(W_pol)
                if verify_exact(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        return None

    # Attempt rank 12
    sol = try_rank(12)
    if sol:
        return sol
    
    # Attempt rank 13? Unlikely to be integer/half-integer exact if 12 fails and 24 works.
    # But let's try rank 14, 16 just in case? No, rank must be minimal.
    # If 12 fails, 24 is the next known bound.
    
    return fallback

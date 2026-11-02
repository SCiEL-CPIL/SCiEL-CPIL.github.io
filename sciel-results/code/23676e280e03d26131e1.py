import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s
    
    deadline = time.time() + 140
    
    def verify(U, V, W, tol=1e-10):
        S = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(S, T, atol=tol)

    def try_rank(R, max_iters=100, n_attempts=5):
        """
        Try to find a rank-R decomposition.
        Uses ALS with rounding.
        """
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with random integers in {-1, 0, 1} or similar
            # Sparse initialization is often better for tensor rank
            density = 0.2
            Ur = np.random.choice([-1, 0, 1], (nm, R))
            Ur[np.random.rand(nm, R) > density] = 0
            Vr = np.random.choice([-1, 0, 1], (mp, R))
            Vr[np.random.rand(mp, R) > density] = 0
            Wr = np.random.choice([-1, 0, 1], (pn, R))
            Wr[np.random.rand(pn, R) > density] = 0
            
            # Ensure not all zeros
            if not np.any(Ur): Ur[0,0] = 1
            if not np.any(Vr): Vr[0,0] = 1
            if not np.any(Wr): Wr[0,0] = 1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # ALS: Fix V, W, update U
                # T = sum_r U_r V_r W_r^T
                # For each r, we want to minimize || T - sum_s U_s V_s W_s^T ||^2
                # This is not linear in U_r if we update all at once, but if we fix others, it is linear?
                # No, it's quadratic.
                # However, we can use a greedy update:
                # Compute residual R = T - sum_{s != r} U_s V_s W_s^T
                # We want to find U_r to minimize || R - U_r (V_r W_r^T) ||_F^2
                # This is a rank-1 approximation problem.
                # U_r = argmin || R - U_r X_r ||_F^2 where X_r = V_r W_r^T
                # Solution: U_r = (R X_r^T) (X_r X_r^T)^{-1}
                
                # This is O(R * nm * mp * pn) which is too slow if done naively for each r.
                # But nm=12, mp=28, pn=21. R is small.
                # Let's try it.
                
                for r_idx in range(R):
                    # Compute S_other
                    S_other = np.zeros((nm, mp, pn))
                    for s in range(R):
                        if s != r_idx:
                            S_other += np.einsum('a,b,c->abc', Ur[:, s], Vr[:, s], Wr[:, s])
                    
                    Res = T - S_other
                    X_r = np.outer(Vr[:, r_idx], Wr[:, r_idx]) # Shape (mp, pn)
                    
                    # Solve for Ur[:, r_idx]
                    # Res = Ur[:, r_idx] @ X_r  (flattened)
                    # Res_mat (nm, mp*pn) = Ur_col (nm, 1) @ X_r_flat (1, mp*pn)
                    # Ur_col = Res_mat @ X_r_flat^T / (X_r_flat @ X_r_flat^T)
                    
                    X_flat = X_r.flatten()
                    X_norm_sq = np.dot(X_flat, X_flat)
                    if X_norm_sq < 1e-15:
                        continue
                    
                    Res_mat = Res.reshape(nm, mp * pn)
                    Ur[:, r_idx] = (Res_mat @ X_flat) / X_norm_sq
                    
                # Similarly for V and W
                for r_idx in range(R):
                    S_other = np.zeros((nm, mp, pn))
                    for s in range(R):
                        if s != r_idx:
                            S_other += np.einsum('a,b,c->abc', Ur[:, s], Vr[:, s], Wr[:, s])
                    
                    Res = T - S_other
                    X_r = np.outer(Ur[:, r_idx], Wr[:, r_idx]) # Shape (nm, pn)
                    X_flat = X_r.flatten()
                    X_norm_sq = np.dot(X_flat, X_flat)
                    if X_norm_sq < 1e-15:
                        continue
                    
                    Res_mat = Res.reshape(mp, nm * pn)
                    Vr[:, r_idx] = (Res_mat @ X_flat) / X_norm_sq
                    
                for r_idx in range(R):
                    S_other = np.zeros((nm, mp, pn))
                    for s in range(R):
                        if s != r_idx:
                            S_other += np.einsum('a,b,c->abc', Ur[:, s], Vr[:, s], Wr[:, s])
                    
                    Res = T - S_other
                    X_r = np.outer(Ur[:, r_idx], Vr[:, r_idx]) # Shape (nm, mp)
                    X_flat = X_r.flatten()
                    X_norm_sq = np.dot(X_flat, X_flat)
                    if X_norm_sq < 1e-15:
                        continue
                    
                    Res_mat = Res.reshape(pn, nm * mp)
                    Wr[:, r_idx] = (Res_mat @ X_flat) / X_norm_sq
                
                # Check for exact solution after rounding
                # We round to nearest integer (or half-integer)
                Ur_int = np.round(Ur)
                Vr_int = np.round(Vr)
                Wr_int = np.round(Wr)
                
                if verify(Ur_int, Vr_int, Wr_int):
                    return Ur_int, Vr_int, Wr_int
                    
        return None

    # Try to find a rank lower than schoolbook
    # Start from a reasonable guess, e.g., 50, and go down
    # Or start from 28 and go up until found?
    # Going up is safer because lower ranks are harder to find.
    # But we want the MINIMAL rank.
    # Let's try a few specific ranks that are known to be feasible or likely.
    
    # We know rank >= 28.
    # Let's try to find a solution for R=40 first to ensure we beat 84.
    # Then try to lower it.
    
    found = False
    for R in range(40, 27, -1):
        if time.time() > deadline:
            break
            
        res = try_rank(R, max_iters=50, n_attempts=10)
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = R
            best_U, best_V, best_W = Ur, Vr, Wr
            found = True
            # Continue searching for lower ranks?
            # The loop goes downwards, so it will try 39, 38...
            # This is good.
            
    # If we didn't find anything < 40, we still have the schoolbook.
    # But we should try a few higher ranks if 40-28 failed?
    # No, 40 is already better than 84.
    # If 40 fails, try 50, 60...
    
    if not found:
        for R in [50, 60, 70, 80]:
            if time.time() > deadline:
                break
            res = try_rank(R, max_iters=50, n_attempts=10)
            if res is not None:
                Ur, Vr, Wr = res
                best_rank = R
                best_U, best_V, best_W = Ur, Vr, Wr
                break

    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 10s margin
    
    # Build target tensor T
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    # Shape: (n*m, m*p, p*n) = (9, 21, 21)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Flatten T for easier handling in least squares
    # T_flat[a, b*p*n + c]
    T_flat = T.reshape(n*m, m*p * p*n)
    
    def verify(U, V, W):
        """
        Verify if U, V, W (as lists of lists) form a valid decomposition.
        Returns True if valid, False otherwise.
        Also checks if entries are integers or half-integers.
        """
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Check half-integer property
        U_check = U_arr * 2
        V_check = V_arr * 2
        W_check = W_arr * 2
        
        if not np.allclose(U_check, np.round(U_check), atol=1e-9):
            return False
        if not np.allclose(V_check, np.round(V_check), atol=1e-9):
            return False
        if not np.allclose(W_check, np.round(W_check), atol=1e-9):
            return False
            
        U_int = np.round(U_check).astype(np.int64)
        V_int = np.round(V_check).astype(np.int64)
        W_int = np.round(W_check).astype(np.int64)
        
        rank = U_int.shape[1]
        
        # Check all entries of T
        # T_calc[a, b, c] = sum_k U_int[a, k] * V_int[b, k] * W_int[c, k]
        # We need T_calc to match T exactly.
        # Since T is 0 or 1, and U,V,W are half-integers (scaled by 2 to int),
        # The product U*V*W is multiple of 1/8.
        # Sum is multiple of 1/8.
        # So T_calc * 8 must be integer and match T * 8.
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = 0
                    for k in range(rank):
                        val += U_int[a, k] * V_int[b, k] * W_int[c, k]
                    target = int(T[a, b, c] * 8)
                    if val != target:
                        return False
        return True

    def solve_rank(r, time_limit, n_init=10, max_iter=200):
        """
        Try to find a rank-r decomposition.
        """
        for init in range(n_init):
            if time.time() > time_limit:
                return None
            
            # Random initialization
            # Scale matters for convergence.
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize to avoid exploding/decaying values
            # We can normalize each column or row, but let's just start with standard normal.
            
            for it in range(max_iter):
                if time.time() > time_limit:
                    return None
                
                # Update U
                # For each a, solve M_a U[a,:] = T_flat[a,:]
                # M_a[b*p*n+c, k] = V[b,k] * W[c,k]
                # We can construct M_a for each a.
                # To save memory/time, we construct M once if V,W don't change? No, V,W change.
                # But for a fixed iteration step, V and W are fixed while updating U.
                
                # Construct M_U: shape (n*m, m*p*p*n, r)
                # M_U[a, b*p*n+c, k] = V[b,k] * W[c,k]
                # This is large: 9 * 441 * 60 ~ 238k floats. OK.
                M_U = np.zeros((n*m, m*p * p*n, r))
                for b in range(m*p):
                    for c in range(p*n):
                        # V[b, :] is (r,), W[c, :] is (r,)
                        # Element-wise product is (r,)
                        vec = V[b, :] * W[c, :]
                        for a in range(n*m):
                            M_U[a, b * p*n + c, :] = vec
                            
                # Solve least squares for each a
                # U[a, :] = argmin || M_U[a] @ U[a,:] - T_flat[a] ||
                # We can do this in a loop or vectorize.
                # np.linalg.lstsq does not vectorize over the first dimension easily in older versions,
                # but we can loop over a (only 9 iterations).
                for a in range(n*m):
                    U[a, :], _, _, _ = np.linalg.lstsq(M_U[a], T_flat[a], rcond=None)
                
                # Update V
                # For each b, solve M_V[b] V[b,:] = T_flat[:, b, :].flatten()
                # M_V[b, a*p*n+c, k] = U[a,k] * W[c,k]
                M_V = np.zeros((m*p, n*m * p*n, r))
                for a in range(n*m):
                    for c in range(p*n):
                        vec = U[a, :] * W[c, :]
                        for b in range(m*p):
                            M_V[b, a * p*n + c, :] = vec
                            
                for b in range(m*p):
                    # Target is T[:, b, :] flattened
                    t_b = T[:, b, :].flatten()
                    V[b, :], _, _, _ = np.linalg.lstsq(M_V[b], t_b, rcond=None)
                    
                # Update W
                # For each c, solve M_W[c] W[c,:] = T[:, :, c].flatten()
                # M_W[c, a*m*p+b, k] = U[a,k] * V[b,k]
                M_W = np.zeros((p*n, n*m * m*p, r))
                for a in range(n*m):
                    for b in range(m*p):
                        vec = U[a, :] * V[b, :]
                        for c in range(p*n):
                            M_W[c, a * m*p + b, :] = vec
                            
                for c in range(p*n):
                    # Target is T[:, :, c] flattened
                    t_c = T[:, :, c].flatten()
                    W[c, :], _, _, _ = np.linalg.lstsq(M_W[c], t_c, rcond=None)
                    
                # Optional: Check convergence
                # Calculate residual
                # This is expensive, so maybe check every 10 iterations or at the end.
                
            # After iterations, round to half-integers
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            # Verify
            if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
        return None

    # Try to find a decomposition with rank < 63
    # Start from 62 and go down.
    # If we find one, return it.
    # If time runs out, return schoolbook.
    
    schoolbook_rank = n * m * p
    best_result = None
    best_rank = schoolbook_rank
    
    # Schoolbook solution for fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    # Try ranks 62, 61, ...
    # We need to be careful with time.
    # Let's try a few ranks.
    for r in range(schoolbook_rank - 1, max(20, schoolbook_rank - 20), -1):
        if time.time() > deadline - 10:
            break
        res = solve_rank(r, deadline - 5, n_init=5, max_iter=100)
        if res is not None:
            U, V, W = res
            # This is a valid decomposition with rank r < schoolbook_rank
            return {"U": U, "V": V, "W": W}
            
    # If no better rank found, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}

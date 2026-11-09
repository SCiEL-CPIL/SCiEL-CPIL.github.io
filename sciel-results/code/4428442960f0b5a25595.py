import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Construct the tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize best with schoolbook
    best_U = np.zeros((nm, schoolbook_rank))
    best_V = np.zeros((mp, schoolbook_rank))
    best_W = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1.0
                best_V[j*p+k, col] = 1.0
                best_W[k*n+i, col] = 1.0
                col += 1
                
    deadline = time.time() + 140 # Safety margin
    
    def check_error(U, V, W, R):
        # Compute sum_s U[:,s] V[:,s] W[:,s]
        # Use einsum for speed
        # T_rec = np.einsum('as,bs,cs->abc', U, V, W)
        # This might be slow if R is large, but R <= 60.
        # nm=6, mp=30, pn=20.
        # 6*30*20 = 3600.
        # R * 3600 ops.
        T_rec = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T_rec - T))
        
    def try_rank(R, max_iter=100, num_seeds=20):
        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                return None
            np.random.seed(seed * 12345 + R * 6789)
            
            # Initialize
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Normalize to prevent overflow/underflow
            U /= np.sqrt(np.sum(U**2))
            V /= np.sqrt(np.sum(V**2))
            W /= np.sqrt(np.sum(W**2))
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                # M_U[s, bc] = V[b,s] * W[c,s]
                # M_U is R x (mp*pn)
                M_U = np.einsum('bs,cs->sbc', V, W).reshape(R, mp * pn)
                Y_U = T.reshape(nm, mp * pn)
                # Solve U M_U = Y_U
                # U = Y_U M_U^T (M_U M_U^T)^-1
                MMt = M_U @ M_U.T
                Y_Mt = Y_U @ M_U.T
                try:
                    U = np.linalg.solve(MMt, Y_Mt.T).T
                except np.linalg.LinAlgError:
                    U = np.linalg.pinv(MMt) @ Y_Mt.T
                    U = U.T

                # Update V
                # M_V[s, ac] = U[a,s] * W[c,s]
                # M_V is R x (nm*pn)
                M_V = np.einsum('as,cs->sac', U, W).reshape(R, nm * pn)
                Y_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                MMt = M_V @ M_V.T
                Y_Mt = Y_V @ M_V.T
                try:
                    V = np.linalg.solve(MMt, Y_Mt.T).T
                except np.linalg.LinAlgError:
                    V = np.linalg.pinv(MMt) @ Y_Mt.T
                    V = V.T

                # Update W
                # M_W[s, ab] = U[a,s] * V[b,s]
                # M_W is R x (nm*mp)
                M_W = np.einsum('as,bs->sab', U, V).reshape(R, nm * mp)
                Y_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                MMt = M_W @ M_W.T
                Y_Mt = Y_W @ M_W.T
                try:
                    W = np.linalg.solve(MMt, Y_Mt.T).T
                except np.linalg.LinAlgError:
                    W = np.linalg.pinv(MMt) @ Y_Mt.T
                    W = W.T

                # Check convergence
                err = check_error(U, V, W, R)
                if err < 1e-10:
                    # Try to round to half-integers
                    for scale in [1, 2, 4, 8]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale
                        err_r = check_error(U_r, V_r, W_r, R)
                        if err_r < 1e-9:
                            return (U_r, V_r, W_r)
                    # Return float if no simple rational found
                    return (U, V, W)
            
            # End of iterations for this seed
            err = check_error(U, V, W, R)
            if err < 1e-8:
                # Try rounding
                for scale in [1, 2, 4, 8]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    err_r = check_error(U_r, V_r, W_r, R)
                    if err_r < 1e-9:
                        return (U_r, V_r, W_r)
                return (U, V, W)
        
        return None
        
    # Try ranks from schoolbook-1 down
    for R in range(schoolbook_rank - 1, 34, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(R, max_iter=50, num_seeds=10)
        if result is not None:
            U, V, W = result
            err = check_error(U, V, W, R)
            if err < 1e-9:
                best_U = U
                best_V = V
                best_W = W
                best_rank = R
                # We found a lower rank, so we can stop if we want the minimum?
                # No, we want the MINIMUM rank. So we should continue to lower ranks.
                # But lower ranks are harder.
                # If we found R=50, we should try 49, 48...
                # So we do NOT break. We continue the loop.
                # However, if we find a very low rank, we should be careful.
                # The loop goes down to 35.
                pass
            else:
                # Error too high, ignore
                pass
                
    # If we didn't find any better than schoolbook, use schoolbook
    if best_rank == schoolbook_rank:
        U = best_U[:, :best_rank]
        V = best_V[:, :best_rank]
        W = best_W[:, :best_rank]
    else:
        U = best_U[:, :best_rank]
        V = best_V[:, :best_rank]
        W = best_W[:, :best_rank]
        
    # Convert to list of lists of floats
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

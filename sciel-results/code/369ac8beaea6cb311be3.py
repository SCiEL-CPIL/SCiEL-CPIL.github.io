import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed

    # Generate all constraints (a, b, c, target)
    # a in [0, nm), b in [0, mp), c in [0, pn)
    # Target is 1 if (a,b,c) corresponds to (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # Else 0.
    
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    
    valid_set = set(valid_triples)
    
    all_a = np.arange(n * m)
    all_b = np.arange(m * p)
    all_c = np.arange(p * n)
    
    # Create arrays for all 576 constraints
    # We can vectorize the check
    A_all = np.repeat(all_a, (m * p) * (p * n))
    B_all = np.tile(np.repeat(all_b, p * n), n * m)
    C_all = np.tile(all_c, (n * m) * (m * p))
    
    Targets = np.zeros(len(A_all))
    for a, b, c in valid_triples:
        # Find indices where a, b, c match
        mask = (A_all == a) & (B_all == b) & (C_all == c)
        Targets[mask] = 1.0

    def error_function(U, V, W):
        # Reshape U, V, W to 3D for broadcasting?
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # We need sum_l U[a,l] V[b,l] W[c,l]
        # This is equivalent to element-wise product summed over l.
        # To do this for all a,b,c efficiently:
        # We can loop over l? No, r is small.
        # Or use einsum?
        # T[a,b,c] = sum_l U[a,l] V[b,l] W[c,l]
        # This is a tensor of size nm x mp x pn.
        # We can compute it as:
        # Temp = U[:, :, None, None] * V[None, :, :, None] * W[None, None, :, :]
        # This creates a tensor of size nm x mp x pn x r.
        # Sum over r.
        # nm=6, mp=12, pn=8, r~20. Size 6*12*8*20 = 11520. Very small.
        
        U3 = U[:, :, np.newaxis, np.newaxis] # nm x r x 1 x 1
        V3 = V[np.newaxis, :, :, np.newaxis] # 1 x mp x r x 1
        W3 = W[np.newaxis, np.newaxis, :, :] # 1 x 1 x pn x r
        
        # Product
        Prod = U3 * V3 * W3 # nm x mp x pn x r
        T = np.sum(Prod, axis=3) # nm x mp x pn
        
        T_flat = T.flatten()
        return T_flat - Targets

    def check_exact(U, V, W):
        vals = error_function(U, V, W)
        return np.allclose(vals, 0.0, atol=1e-9)

    def normalize_columns(X):
        norms = np.linalg.norm(X, axis=0)
        norms[norms < 1e-12] = 1.0
        return X / norms

    def try_rank_als(r, max_iters=50, num_restarts=100, seed_base=42):
        if time.time() > deadline - 10.0:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                return None
                
            np.random.seed(seed_base + restart)
            
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            U = normalize_columns(U)
            V = normalize_columns(V)
            W = normalize_columns(W)
            
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    return None
                
                # Fast ALS update using the vectorized trick derived earlier
                # Update U
                # For each l, U[:, l] is updated.
                # U[:, l] = argmin sum_{a,b,c} (Target[a,b,c] - sum_{l'} U[a,l'] V[b,l'] W[c,l'])^2
                # Fix l. Let R = Target - sum_{l' != l} U[:, l'] V[:, l'] W[:, l'] (broadcast)
                # Minimize sum_{a,b,c} (R[a,b,c] - U[a,l] V[b,l] W[c,l])^2
                # This is sum_a sum_{b,c} (R[a,b,c] - U[a,l] C[b,c])^2 where C[b,c] = V[b,l] W[c,l]
                # For each a, U[a,l] = sum_{b,c} R[a,b,c] C[b,c] / sum_{b,c} C[b,c]^2
                
                U_new = np.zeros_like(U)
                V_new = np.zeros_like(V)
                W_new = np.zeros_like(W)
                
                # Compute residuals for each l
                # We can compute the full tensor T once?
                # T = sum_l U[:,l] V[:,l] W[:,l]
                # It's easier to just recompute the error term for each l update.
                
                for l in range(r):
                    # Compute C_l
                    C_l = np.outer(V[:, l], W[:, l]) # mp x pn
                    
                    # Compute R_l = Targets - sum_{l' != l} U[:, l'] V[:, l'] W[:, l']
                    # This is T - U[:, l] V[:, l] W[:, l]
                    # We can compute T_full = sum_l' U[:, l'] V[:, l'] W[:, l']
                    # But doing this inside the loop is O(r * nm * mp * pn * r) = O(r^2 * 576).
                    # r=20, 400 * 576 = 230k ops. Fast.
                    
                    T_full = np.zeros((n*m, m*p, p*n))
                    for l_prime in range(r):
                        if l_prime == l: continue
                        T_full += np.einsum('ar,br,cr->abc', U[:, l_prime], V[:, l_prime], W[:, l_prime])
                    
                    R_l = Targets.reshape(n*m, m*p, p*n) - T_full
                    
                    # Numerator: sum_{b,c} R_l[a,b,c] C_l[b,c]
                    # Denominator: sum_{b,c} C_l[b,c]^2
                    Denom = np.sum(C_l ** 2)
                    if Denom < 1e-12:
                        continue
                    Numer = np.einsum('abc,bc->a', R_l, C_l)
                    U_new[:, l] = Numer / Denom
                    
                    # Update V
                    # Minimize sum_{a,b,c} (R_l[a,b,c] - U_new[a,l] V[b,l] W[c,l])^2
                    # For each b, V[b,l] = sum_{a,c} R_l[a,b,c] U_new[a,l] W[c,l] / sum_{a,c} (U_new[a,l] W[c,l])^2
                    D_V = np.sum((np.outer(U_new[:, l], W[:, l])) ** 2)
                    if D_V < 1e-12: continue
                    Num_V = np.einsum('abc,ac->b', R_l, np.outer(U_new[:, l], W[:, l]))
                    V_new[:, l] = Num_V / D_V
                    
                    # Update W
                    D_W = np.sum((np.outer(U_new[:, l], V_new[:, l])) ** 2)
                    if D_W < 1e-12: continue
                    Num_W = np.einsum('abc,ab->c', R_l, np.outer(U_new[:, l], V_new[:, l]))
                    W_new[:, l] = Num_W / D_W

                U, V, W = U_new, V_new, W_new
                U = normalize_columns(U)
                V = normalize_columns(V)
                W = normalize_columns(W)
                
                err = np.sum(error_function(U, V, W) ** 2)
                
                if err < 1e-8:
                    # Polish with least_squares on all constraints
                    def residuals(params):
                        U_p = params[:n*m*r].reshape(n*m, r)
                        V_p = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                        W_p = params[n*m*r+m*p*r:].reshape(p*n, r)
                        return error_function(U_p, V_p, W_p)
                    
                    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    try:
                        res = least_squares(residuals, x0, method='trf', max_nfev=1000)
                        if res.success or np.sum(res.fun**2) < 1e-10:
                            U_pol = res.x[:n*m*r].reshape(n*m, r)
                            V_pol = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                            W_pol = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                            
                            # Round to 0.5
                            U_r = np.round(U_pol * 2) / 2
                            V_r = np.round(V_pol * 2) / 2
                            W_r = np.round(W_pol * 2) / 2
                            
                            if check_exact(U_r, V_r, W_r):
                                return U_r, V_r, W_r
                    except Exception:
                        pass
                    break

                if err > prev_err * 1.01:
                    break
                prev_err = err

        return None

    ranks_to_try = [23, 22, 21, 20]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
            
        sol = try_rank_als(r, max_iters=30, num_restarts=50, seed_base=r*100)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

<<<<<<< SEARCH
def random_search(rank_target, max_iters):
        """Try to find a decomposition of given rank."""
        np.random.seed(42)
        n, m, p = DIMENSIONS
        nm = n*m
        mp = m*p
        pn = p*n
        
        for trial in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with random values
            U = np.random.randn(nm, rank_target)
            V = np.random.randn(mp, rank_target)
            W = np.random.randn(pn, rank_target)
            
            # Scale to help
            U /= np.sqrt(rank_target)
            V /= np.sqrt(rank_target)
            W /= np.sqrt(rank_target)
            
            # Try a few gradient steps to improve fit
            best_err = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(20):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradient-like update
                # err = T_recon - T_target
                # dU = sum_bc err[a,b,c] * V[b,r] * W[c,r] for each a, r
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Build target
                T_target = np.zeros((nm, mp, pn), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
                err = T_recon - T_target
                current_err = np.sum(err**2)
                
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if current_err < 1e-10:
                    break
                
                # Gradient descent
                lr = 0.1
                dU = np.einsum('abc,br,cr->ar', err, V, W)
                dV = np.einsum('abc,ar,cr->br', err, U, W)
                dW = np.einsum('abc,ar,br->cr', err, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Normalize to prevent divergence
                normU = np.linalg.norm(U)
                normV = np.linalg.norm(V)
                normW = np.linalg.norm(W)
                if normU > 0 and normV > 0 and normW > 0:
                    scale = (normU * normV * normW) ** (1/3)
                    U /= scale * np.sqrt(rank_target)
                    V /= scale * np.sqrt(rank_target)
                    W /= scale * np.sqrt(rank_target)
            
            if best_err < 1e-8:
                # Convert to lists and verify
                U_list = best_U.tolist()
                V_list = best_V.tolist()
                W_list = best_W.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
=======
def random_search(rank_target, max_iters):
        """Try to find a decomposition of given rank using ALS."""
        n, m, p = DIMENSIONS
        nm = n*m
        mp = m*p
        pn = p*n
        
        # Precompute matricizations of the target tensor T
        # T[a, b, c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i
        T_a = np.zeros((nm, mp * pn))
        T_b = np.zeros((mp, nm * pn))
        T_c = np.zeros((pn, nm * mp))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    # Flatten indices for matricizations
                    # T_a[a, b*pn + c] = 1
                    T_a[a, b*pn + c] = 1.0
                    # T_b[b, a*pn + c] = 1
                    T_b[b, a*pn + c] = 1.0
                    # T_c[c, a*mp + b] = 1
                    T_c[c, a*mp + b] = 1.0
        
        for trial in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with random values
            U = np.random.randn(nm, rank_target)
            V = np.random.randn(mp, rank_target)
            W = np.random.randn(pn, rank_target)
            
            # Normalize columns to prevent scale issues
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-9
            V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-9
            W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
            
            best_err = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update U: U = T_a @ (V khatri W)^+
                # Khatri-Rao product of V (mp x R) and W (pn x R) is (mp*pn x R)
                # Column r is V[:, r] * W[:, r] (flattened)
                VW = np.concatenate([V[:, r] * W[:, r] for r in range(rank_target)], axis=1)
                # Solve U @ VW.T = T_a  => U = T_a @ VW^+
                # Using pseudo-inverse
                try:
                    U_new = T_a @ np.linalg.pinv(VW)
                except:
                    break
                
                # Update V: V = T_b @ (W khatri U)^+
                WU = np.concatenate([W[:, r] * U[:, r] for r in range(rank_target)], axis=1)
                try:
                    V_new = T_b @ np.linalg.pinv(WU)
                except:
                    break
                
                # Update W: W = T_c @ (U khatri V)^+
                UV = np.concatenate([U[:, r] * V[:, r] for r in range(rank_target)], axis=1)
                try:
                    W_new = T_c @ np.linalg.pinv(UV)
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Compute error
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                T_target = np.zeros((nm, mp, pn), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_target[i*m+j, j*p+k, k*n+i] = 1.0
                err = T_recon - T_target
                current_err = np.sum(err**2)
                
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if current_err < 1e-10:
                    break
            
            if best_err < 1e-8:
                # Convert to lists and verify
                U_list = best_U.tolist()
                V_list = best_V.tolist()
                W_list = best_W.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
>>>>>>> REPLACE

The ALS approach is significantly more powerful for finding low-rank decompositions than simple gradient descent because it solves for each factor matrix exactly given the others (least squares solution), which is a standard and effective heuristic for tensor decomposition. This increases the chance of finding a valid decomposition with rank < 54 if one exists.

Additionally, I will slightly adjust the search range to start from a theoretically plausible lower bound and increase the number of trials per rank if time permits, but the primary improvement is the solver itself.

I will also add a small optimization: precomputing `T_target` inside the loop is redundant if we do it once per trial or globally. I moved the matricization precomputation inside `random_search` but outside the trial loop to save time.

One potential issue: `np.linalg.pinv` can be slow for large matrices. Here, the matrices are at most $486 \times 54$, which is small. So it should be fast.

Let's also ensure we don't spend too much time on very low ranks that are impossible. The lower bound is 27.
The current loop goes from 27 to 55. This is fine.

I will also update the `construct` function to use a better seed strategy or just rely on the internal seed in `random_search` (which is currently hardcoded to 42, which is bad for diversity). I should remove the hardcoded seed or use a different one per trial.


def _try_rank(rank, target, nm, mp, pn, n, m, p, rng, max_attempts, deadline):
    """Try to find a decomposition of given rank using ALS with various initializations."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        if rank > 40:
            # Perturbed schoolbook-like structure
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            for r in range(rank):
                i = rng.randint(n)
                j = rng.randint(m)
                k = rng.randint(p)
                U[i*m+j, r] = 1.0 + 0.1 * rng.randn()
                V[j*p+k, r] = 1.0 + 0.1 * rng.randn()
                W[k*n+i, r] = 1.0 + 0.1 * rng.randn()
            U += 0.01 * rng.randn(nm, rank)
            V += 0.01 * rng.randn(mp, rank)
            W += 0.01 * rng.randn(pn, rank)
        else:
            # Random integer/half-integer seeds
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = rng.choice(vals, size=(nm, rank))
            V = rng.choice(vals, size=(mp, rank))
            W = rng.choice(vals, size=(pn, rank))
            # Ensure no column is all zero
            for r in range(rank):
                if np.allclose(U[:, r], 0):
                    U[rng.randint(nm), r] = 1.0
                if np.allclose(V[:, r], 0):
                    V[rng.randint(mp), r] = 1.0
                if np.allclose(W[:, r], 0):
                    W[rng.randint(pn), r] = 1.0
        
        # Run ALS
        best_err = np.inf
        stuck_count = 0
        for it in range(200):
            if time.time() > deadline:
                break
            
            result = _als_step(U, V, W, target, nm, mp, pn, rank)
            if result is None:
                break
            U, V, W = result
            
            if it % 5 == 4:
                err = _compute_error(U, V, W, target)
                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        break
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count > 10 and err > best_err * 1.5:
                        break
        
        # Final check
        if best_err < 1e-9:
            decomposed = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(decomposed, target, atol=1e-8, rtol=1e-8):
                # Try to find clean values
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                decomp_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                if np.allclose(decomp_half, target, atol=1e-4, rtol=1e-4):
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                decomp_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                if np.allclose(decomp_int, target, atol=1e-4, rtol=1e-4):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

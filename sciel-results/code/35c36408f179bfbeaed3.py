<<<<<<< SEARCH
    def local_optimize_fast(U, V, W, rank, max_iters=300, atol=1e-9):
        """Fast local optimization using incremental error updates with wider search."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Wider range of valid values
        valid_vals = np.array([x / 2.0 for x in range(-30, 31)])
        
        # Precompute base reconstruction
        Rec_base = compute_reconstruction(U, V, W, rank)
        best_err = np.max(np.abs(Rec_base - 1.0))
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if best_err < atol:
                break
            
            improved = False
            
            # Try adjusting U entries
            for r in range(rank):
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r].reshape(p, n)
                
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        old_val = U[idx, r]
                        
                        # Wider candidate range
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 3.0]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            new_rec_affected = Rec_base[i, j, :] + delta * v_col[j, :] * w_col[:, i]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                # Apply change incrementally
                                U[idx, r] = cand
                                Rec_base[i, j, :] += delta * v_col[j, :] * w_col[:, i]
                                err = np.max(np.abs(Rec_base - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    # Revert
                                    U[idx, r] = old_val
                                    Rec_base[i, j, :] -= delta * v_col[j, :] * w_col[:, i]
                        
                        if best_candidate != old_val:
                            # Recompute Rec_base from scratch to avoid drift
                            Rec_base = compute_reconstruction(U, V, W, rank)
                            best_err = best_candidate_err
                            improved = True
            
            # Try adjusting V entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                w_col = W[:, r].reshape(p, n)
                
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        old_val = V[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 3.0]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            new_rec_affected = Rec_base[:, j, k] + delta * u_col[:, j] * w_col[k, :]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                V[idx, r] = cand
                                Rec_base[:, j, k] += delta * u_col[:, j] * w_col[k, :]
                                err = np.max(np.abs(Rec_base - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    V[idx, r] = old_val
                                    Rec_base[:, j, k] -= delta * u_col[:, j] * w_col[k, :]
                        
                        if best_candidate != old_val:
                            Rec_base = compute_reconstruction(U, V, W, rank)
                            best_err = best_candidate_err
                            improved = True
            
            # Try adjusting W entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r].reshape(m, p)
                
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        old_val = W[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 3.0]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            new_rec_affected = Rec_base[i, :, k] + delta * u_col[i, :] * v_col[:, k]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                W[idx, r] = cand
                                Rec_base[i, :, k] += delta * u_col[i, :] * v_col[:, k]
                                err = np.max(np.abs(Rec_base - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    W[idx, r] = old_val
                                    Rec_base[i, :, k] -= delta * u_col[i, :] * v_col[:, k]
                        
                        if best_candidate != old_val:
                            Rec_base = compute_reconstruction(U, V, W, rank)
                            best_err = best_candidate_err
                            improved = True
            
            if not improved:
                break
        
        if best_err < atol:
            return U, V, W, True
        return U, V, W, False
=======
    def l2_surrogate_err(Rec, beta=2.0):
        """Smooth surrogate for max-norm: (sum(exp(beta*|x-1|)) / N)^(1/beta) - 1."""
        d = np.abs(Rec - 1.0)
        N = d.size
        # Avoid overflow
        d_clipped = np.clip(d, 0, 50.0 / beta)
        s = np.sum(np.exp(beta * d_clipped))
        return (s / N) ** (1.0 / beta) - 1.0

    def local_optimize_fast(U, V, W, rank, max_iters=300, atol=1e-9):
        """Fast local optimization using incremental error updates with wider search."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Wider range of valid values
        valid_vals = np.array([x / 2.0 for x in range(-30, 31)])
        
        # Precompute base reconstruction
        Rec_base = compute_reconstruction(U, V, W, rank)
        best_err = np.max(np.abs(Rec_base - 1.0))
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if best_err < atol:
                break
            
            improved = False
            
            # Phase 1: Greedy max-norm descent (original)
            # Try adjusting U entries
            for r in range(rank):
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r].reshape(p, n)
                
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        old_val = U[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 3.0]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            new_rec_affected = Rec_base[i, j, :] + delta * v_col[j, :] * w_col[:, i]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                U[idx, r] = cand
                                Rec_base[i, j, :] += delta * v_col[j, :] * w_col[:, i]
                                err = np.max(np.abs(Rec_base - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    U[idx, r] = old_val
                                    Rec_base[i, j, :] -= delta * v_col[j, :] * w_col[:, i]
                        
                        if best_candidate != old_val:
                            Rec_base = compute_reconstruction(U, V, W, rank)
                            best_err = best_candidate_err
                            improved = True
            
            # Try adjusting V entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                w_col = W[:, r].reshape(p, n)
                
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        old_val = V[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 3.0]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            new_rec_affected = Rec_base[:, j, k] + delta * u_col[:, j] * w_col[k, :]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                V[idx, r] = cand
                                Rec_base[:, j, k] += delta * u_col[:, j] * w_col[k, :]
                                err = np.max(np.abs(Rec_base - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    V[idx, r] = old_val
                                    Rec_base[:, j, k] -= delta * u_col[:, j] * w_col[k, :]
                        
                        if best_candidate != old_val:
                            Rec_base = compute_reconstruction(U, V, W, rank)
                            best_err = best_candidate_err
                            improved = True
            
            # Try adjusting W entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r].reshape(m, p)
                
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        old_val = W[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 3.0]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            new_rec_affected = Rec_base[i, :, k] + delta * u_col[i, :] * v_col[:, k]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                W[idx, r] = cand
                                Rec_base[i, :, k] += delta * u_col[i, :] * v_col[:, k]
                                err = np.max(np.abs(Rec_base - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    W[idx, r] = old_val
                                    Rec_base[i, :, k] -= delta * u_col[i, :] * v_col[:, k]
                        
                        if best_candidate != old_val:
                            Rec_base = compute_reconstruction(U, V, W, rank)
                            best_err = best_candidate_err
                            improved = True
            
            # Phase 2: Smooth L2 surrogate descent (only if stuck on max-norm)
            if not improved and best_err > atol:
                if time.time() > deadline:
                    break
                
                improved2 = False
                beta = 2.0
                cur_l2 = l2_surrogate_err(Rec_base, beta)
                
                # Try U adjustments with L2 surrogate
                for r in range(rank):
                    v_col = V[:, r].reshape(m, p)
                    w_col = W[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            idx = i * m + j
                            old_val = U[idx, r]
                            candidates = valid_vals[np.abs(valid_vals - old_val) < 2.0]
                            if len(candidates) == 0:
                                continue
                            best_cand = old_val
                            best_l2 = cur_l2
                            for cand in candidates:
                                if abs(cand - old_val) < 1e-12:
                                    continue
                                delta = cand - old_val
                                new_rec_affected = Rec_base[i, j, :] + delta * v_col[j, :] * w_col[:, i]
                                # Quick L2 check on affected slice
                                d_aff = np.abs(new_rec_affected - 1.0)
                                # Approximate: compare affected contribution
                                # Full check is expensive; use affected slice max as proxy
                                # Better: compute L2 on full array only if affected looks promising
                                aff_max = np.max(d_aff)
                                if aff_max < best_err + 0.5:  # only if not making things much worse
                                    U[idx, r] = cand
                                    Rec_base[i, j, :] += delta * v_col[j, :] * w_col[:, i]
                                    new_l2 = l2_surrogate_err(Rec_base, beta)
                                    if new_l2 < best_l2 - 1e-8:
                                        best_l2 = new_l2
                                        best_cand = cand
                                    else:
                                        U[idx, r] = old_val
                                        Rec_base[i, j, :] -= delta * v_col[j, :] * w_col[:, i]
                            if best_cand != old_val:
                                Rec_base = compute_reconstruction(U, V, W, rank)
                                cur_l2 = best_l2
                                best_err = np.max(np.abs(Rec_base - 1.0))
                                improved2 = True
                
                # Try V adjustments with L2 surrogate
                for r in range(rank):
                    u_col = U[:, r].reshape(n, m)
                    w_col = W[:, r].reshape(p, n)
                    for j in range(m):
                        for k in range(p):
                            idx = j * p + k
                            old_val = V[idx, r]
                            candidates = valid_vals[np.abs(valid_vals - old_val) < 2.0]
                            if len(candidates) == 0:
                                continue
                            best_cand = old_val
                            best_l2 = cur_l2
                            for cand in candidates:
                                if abs(cand - old_val) < 1e-12:
                                    continue
                                delta = cand - old_val
                                new_rec_affected = Rec_base[:, j, k] + delta * u_col[:, j] * w_col[k, :]
                                aff_max = np.max(np.abs(new_rec_affected - 1.0))
                                if aff_max < best_err + 0.5:
                                    V[idx, r] = cand
                                    Rec_base[:, j, k] += delta * u_col[:, j] * w_col[k, :]
                                    new_l2 = l2_surrogate_err(Rec_base, beta)
                                    if new_l2 < best_l2 - 1e-8:
                                        best_l2 = new_l2
                                        best_cand = cand
                                    else:
                                        V[idx, r] = old_val
                                        Rec_base[:, j, k] -= delta * u_col[:, j] * w_col[k, :]
                            if best_cand != old_val:
                                Rec_base = compute_reconstruction(U, V, W, rank)
                                cur_l2 = best_l2
                                best_err = np.max(np.abs(Rec_base - 1.0))
                                improved2 = True
                
                # Try W adjustments with L2 surrogate
                for r in range(rank):
                    u_col = U[:, r].reshape(n, m)
                    v_col = V[:, r].reshape(m, p)
                    for k in range(p):
                        for i in range(n):
                            idx = k * n + i
                            old_val = W[idx, r]
                            candidates = valid_vals[np.abs(valid_vals - old_val) < 2.0]
                            if len(candidates) == 0:
                                continue
                            best_cand = old_val
                            best_l2 = cur_l2
                            for cand in candidates:
                                if abs(cand - old_val) < 1e-12:
                                    continue
                                delta = cand - old_val
                                new_rec_affected = Rec_base[i, :, k] + delta * u_col[i, :] * v_col[:, k]
                                aff_max = np.max(np.abs(new_rec_affected - 1.0))
                                if aff_max < best_err + 0.5:
                                    W[idx, r] = cand
                                    Rec_base[i, :, k] += delta * u_col[i, :] * v_col[:, k]
                                    new_l2 = l2_surrogate_err(Rec_base, beta)
                                    if new_l2 < best_l2 - 1e-8:
                                        best_l2 = new_l2
                                        best_cand = cand
                                    else:
                                        W[idx, r] = old_val
                                        Rec_base[i, :, k] -= delta * u_col[i, :] * v_col[:, k]
                            if best_cand != old_val:
                                Rec_base = compute_reconstruction(U, V, W, rank)
                                cur_l2 = best_l2
                                best_err = np.max(np.abs(Rec_base - 1.0))
                                improved2 = True
                
                if not improved2:
                    break
            else:
                if not improved:
                    break
        
        # Final polish: high-precision local search with tighter tolerance
        if best_err < 0.1 and best_err >= atol:
            if time.time() < deadline - 5.0:
                valid_vals_tight = np.array([x / 2.0 for x in range(-10, 11)])
                for _ in range(200):
                    if time.time() > deadline - 2.0:
                        break
                    if best_err < atol:
                        break
                    imp = False
                    for r in range(rank):
                        v_col = V[:, r].reshape(m, p)
                        w_col = W[:, r].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                idx = i * m + j
                                old_val = U[idx, r]
                                cands = valid_vals_tight[np.abs(valid_vals_tight - old_val) < 1.5]
                                if len(cands) == 0:
                                    continue
                                for cand in cands:
                                    if abs(cand - old_val) < 1e-12:
                                        continue
                                    delta = cand - old_val
                                    Rec_base[i, j, :] += delta * v_col[j, :] * w_col[:, i]
                                    err = np.max(np.abs(Rec_base - 1.0))
                                    if err < best_err - 1e-12:
                                        best_err = err
                                        imp = True
                                    else:
                                        Rec_base[i, j, :] -= delta * v_col[j, :] * w_col[:, i]
                                        break
                                else:
                                    continue
                                break
                    for r in range(rank):
                        u_col = U[:, r].reshape(n, m)
                        w_col = W[:, r].reshape(p, n)
                        for j in range(m):
                            for k in range(p):
                                idx = j * p + k
                                old_val = V[idx, r]
                                cands = valid_vals_tight[np.abs(valid_vals_tight - old_val) < 1.5]
                                if len(cands) == 0:
                                    continue
                                for cand in cands:
                                    if abs(cand - old_val) < 1e-12:
                                        continue
                                    delta = cand - old_val
                                    Rec_base[:, j, k] += delta * u_col[:, j] * w_col[k, :]
                                    err = np.max(np.abs(Rec_base - 1.0))
                                    if err < best_err - 1e-12:
                                        best_err = err
                                        imp = True
                                    else:
                                        Rec_base[:, j, k] -= delta * u_col[:, j] * w_col[k, :]
                                        break
                                else:
                                    continue
                                break
                    for r in range(rank):
                        u_col = U[:, r].reshape(n, m)
                        v_col = V[:, r].reshape(m, p)
                        for k in range(p):
                            for i in range(n):
                                idx = k * n + i
                                old_val = W[idx, r]
                                cands = valid_vals_tight[np.abs(valid_vals_tight - old_val) < 1.5]
                                if len(cands) == 0:
                                    continue
                                for cand in cands:
                                    if abs(cand - old_val) < 1e-12:
                                        continue
                                    delta = cand - old_val
                                    Rec_base[i, :, k] += delta * u_col[i, :] * v_col[:, k]
                                    err = np.max(np.abs(Rec_base - 1.0))
                                    if err < best_err - 1e-12:
                                        best_err = err
                                        imp = True
                                    else:
                                        Rec_base[i, :, k] -= delta * u_col[i, :] * v_col[:, k]
                                        break
                                else:
                                    continue
                                break
                    if not imp:
                        break
                Rec_base = compute_reconstruction(U, V, W, rank)
                best_err = np.max(np.abs(Rec_base - 1.0))
        
        if best_err < atol:
            return U, V, W, True
        return U, V, W, False
>>>>>>> REPLACE


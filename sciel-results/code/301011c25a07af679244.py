    def _reduce_rank_iterative(U, V, W, n, m, p, deadline):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        while time.time() < deadline - 5:
            r = U.shape[1]
            if r == 0: break
            
            # Build column vectors
            col_vecs = np.zeros((r, n * m * p), dtype=float)
            for c in range(r):
                u_2d = U[:, c].reshape(n, m)
                v_2d = V[:, c].reshape(m, p)
                w_2d = W[:, c].reshape(p, n)
                t_c = np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
                col_vecs[c] = t_c.flatten()
            
            # Check for zero columns
            zero_cols = [c for c in range(r) if np.allclose(col_vecs[c], 0, atol=1e-12)]
            if zero_cols:
                keep = [c for c in range(r) if c not in zero_cols]
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                continue
            
            # Find null space
            try:
                U_svd, s_vals, Vt_svd = np.linalg.svd(col_vecs, full_matrices=False)
            except np.linalg.LinAlgError:
                break
                
            tol = 1e-10 * (max(s_vals) if len(s_vals) > 0 else 1.0)
            rank_cv = sum(1 for s in s_vals if s > tol)
            
            if rank_cv < r:
                null_vecs = Vt_svd[rank_cv:]
                # Try to find a valid removal
                removed = False
                for nv in null_vecs:
                    nonzero = np.abs(nv) > 1e-12
                    if not np.any(nonzero): continue
                    for c in range(r):
                        if abs(nv[c]) < 1e-12: continue
                        # Scale nv so nv[c] = -1
                        scale = -1.0 / nv[c]
                        alpha = nv * scale
                        # Remove column c
                        keep = [cc for cc in range(r) if cc != c]
                        U_new = U[:, keep]
                        V_new = V[:, keep]
                        W_new = W[:, keep]
                        # Verify
                        if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                            U, V, W = U_new, V_new, W_new
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    continue
            else:
                # No more dependencies found in this basis
                break
                
        return U, V, W
    
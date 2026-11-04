            # Vectorized search for best w given u, v
            # But we also need to search u, v.
            # Let's fix u, then search v, w jointly?
            # Score(u, v, w) = <T, u v w>
            # For fixed u: Ru = T u.
            # For fixed u, v: Rv = Ru v.
            # Score = Rv w.
            
            # Let's iterate u.
            for ui in u_idx:
                u = U_cands[ui]
                Ru = np.tensordot(T, u, axes=([0], [0]))
                # Now search v, w
                # We can compute a matrix M[v, w] = <Ru, v w> ?
                # M = Ru reshaped?
                # Ru is (m*p, p*n).
                # v is (m*p,). w is (p*n,).
                # Score(v, w) = sum_b,c Ru[b,c] v[b] w[c] = (v^T Ru) w = (Ru^T v)^T w.
                # Let's compute A = Ru^T @ V_cands.T -> shape (p*n, num_v)
                # Then for each w, score = A[:, v_idx] dot w?
                # No, for a specific v, the vector is A[:, v].
                # We want max over v, w of (A[:, v] dot w).
                # This is max over v of (max over w of (A[:, v] dot w)).
                
                # Compute A = Ru.T @ V_cands.T
                A = Ru.T @ V_cands.T
                # For each v candidate (column in A), find best w
                # best_w_score = A @ W_cands.T ?
                # A is (p*n, num_v). W_cands.T is (num_w, p*n).
                # S = A @ W_cands.T -> (num_v, num_w)
                # S[v_idx, w_idx] is the score.
                S = A @ W_cands.T
                # Find max in S
                max_val = np.max(S)
                if max_val > best_score:
                    best_score = max_val
                    v_idx_best, w_idx_best = np.unravel_index(np.argmax(S), S.shape)
                    best_u = u
                    best_v = V_cands[v_idx_best]
                    best_w = W_cands[w_idx_best]

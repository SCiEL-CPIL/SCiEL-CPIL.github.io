<<<<<<< SEARCH
    def _optimize_rank_adaptive(rank, seed, max_iters=5000):
        """Adaptive learning rate gradient descent."""
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, rank) * 0.5
        V = rng.randn(mp, rank) * 0.5
        W = rng.randn(pn, rank) * 0.5

        lr = 0.05
        prev_loss = np.inf

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            recon = np.zeros((nm, mp, pn), dtype=np.float64)
            for r in range(rank):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            diff = recon - T
            loss = 0.5 * np.sum(diff * diff)

            # Adaptive LR
            if loss > prev_loss:
                lr *= 0.5
            elif it > 0 and (loss - prev_loss) < -1e-8:
                lr = min(lr * 1.05, 0.1)
            prev_loss = loss

            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(rank):
                dU[:, r] = np.einsum('abc,b,c->a', diff, V[:, r], W[:, r])
                dV[:, r] = np.einsum('abc,a,c->b', diff, U[:, r], W[:, r])
                dW[:, r] = np.einsum('abc,a,b->c', diff, U[:, r], V[:, r])

            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

            if it % 200 == 0:
                err = np.max(np.abs(recon - T))
                if err < 1e-6:
                    result = _try_round_and_verify(U, V, W)
                    if result is not None:
                        return result

        # Final
        recon = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = np.max(np.abs(recon - T))
        if err < 1e-6:
            result = _try_round_and_verify(U, V, W)
            if result is not None:
                return result
        return None
=======
    def _optimize_rank_als(rank, seed, max_iters=2000):
        """Alternating Least Squares with rounding."""
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, rank) * 0.5
        V = rng.randn(mp, rank) * 0.5
        W = rng.randn(pn, rank) * 0.5

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for r in range(rank):
                # T[:, :, :] = U[:, r] * V[:, r] * W[:, r]
                # Fix V, W. Solve for U[:, r]
                # T = sum_s U_s V_s W_s
                # T - sum_{s!=r} U_s V_s W_s = U_r V_r W_r
                # This is a linear system if we fix the other factors.
                # But we are updating one factor at a time, keeping others fixed.
                # Let's do standard ALS:
                # For U:
                # T = U V W^T (matrix unfolding)
                # T_(1) = U (W \otimes V)
                # U = T_(1) (W \otimes V)^+
                
                # Efficiently:
                # For each r, we want to minimize || T - sum_s U_s V_s W_s ||_F
                # This is equivalent to solving for U[:, r] given V[:, r] and W[:, r] and the rest.
                # Actually, standard ALS updates all rows of U for a fixed V, W.
                
                # Let's compute the residual excluding U[:, r]
                # Res_r = T - sum_{s!=r} U_s V_s W_s
                # We want U[:, r] such that U[:, r] \otimes V[:, r] \otimes W[:, r] approximates Res_r
                # This is a linear least squares problem for U[:, r].
                # Res_r[a, b, c] = U[a, r] V[b, r] W[c, r]
                # Sum_{a,b,c} (Res_r[a,b,c] - U[a,r] V[b,r] W[c,r])^2
                # This is separable.
                # U[a,r] = sum_{b,c} Res_r[a,b,c] V[b,r] W[c,r] / (V[b,r]^2 W[c,r]^2 sum?) No.
                # Let's use the matrix unfolding.
                # T_(1) [a, (b,c)] = T[a,b,c]
                # M = V[:, r] \otimes W[:, r]  (shape mp*pn)
                # T_res_(1) = T_(1) - sum_{s!=r} U_s M_s
                # U[:, r] = T_res_(1) M^+
                
                # This is expensive for each r.
                # Alternative: Update all U at once.
                # T_(1) = U (W \otimes V)
                # U = T_(1) (W \otimes V)^+
                
                # Let's do that.
                # Construct M_WV = W \otimes V (shape pn, mp*rank? No, rank columns)
                # M_WV[:, r] = W \otimes V[:, r] ? No.
                # T_(1) is (nm, mp*pn).
                # U is (nm, rank).
                # M is (mp*pn, rank).
                # M[:, r] = V[:, r] \otimes W  (shape mp*pn)
                
                M = np.zeros((mp * pn, rank))
                for r in range(rank):
                    M[:, r] = np.kron(V[:, r], W) # V is mp, W is pn. kron(V, W) is mp*pn.
                    # Wait, index order in T_(1) is a, b, c.
                    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                    # T_(1)[a, b*pn + c] = sum_r U[a, r] (V[b, r] W[c, r])
                    # So M[:, r] should be vector of length mp*pn where index b*pn+c is V[b,r]*W[c,r].
                    # This is exactly np.kron(V[:, r], W) if we flatten b then c?
                    # np.kron(v, w) where v is mp, w is pn gives mp*pn.
                    # Element (b, c) is v[b] w[c]. Yes.
                
                # Solve U = T_(1) M^+
                T_1 = T.reshape(nm, mp * pn)
                try:
                    U_new = np.linalg.lstsq(M, T_1.T, rcond=None)[0].T
                except:
                    continue
                U = U_new

                # Similarly for V
                # T_(2) [b, a*pn + c] = sum_r V[b, r] U[a, r] W[c, r]
                # M_V[:, r] = U[:, r] \otimes W[:, r] ? No, U is nm, W is pn.
                # Index a*pn + c.
                # M_V[:, r] = np.kron(U[:, r], W)
                M = np.zeros((nm * pn, rank))
                for r in range(rank):
                    M[:, r] = np.kron(U[:, r], W)
                T_2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new = np.linalg.lstsq(M, T_2.T, rcond=None)[0].T
                except:
                    continue
                V = V_new

                # Similarly for W
                # T_(3) [c, a*mp + b] = sum_r W[c, r] U[a, r] V[b, r]
                # M_W[:, r] = U[:, r] \otimes V[:, r]
                M = np.zeros((nm * mp, rank))
                for r in range(rank):
                    M[:, r] = np.kron(U[:, r], V[:, r])
                T_3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new = np.linalg.lstsq(M, T_3.T, rcond=None)[0].T
                except:
                    continue
                W = W_new

            if it % 50 == 0:
                recon = np.zeros((nm, mp, pn), dtype=np.float64)
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = np.max(np.abs(recon - T))
                if err < 1e-6:
                    result = _try_round_and_verify(U, V, W)
                    if result is not None:
                        return result

        recon = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = np.max(np.abs(recon - T))
        if err < 1e-6:
            result = _try_round_and_verify(U, V, W)
            if result is not None:
                return result
        return None
>>>>>>> REPLACE


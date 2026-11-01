<<<<<<< SEARCH
    # Fallback: Return the known 54-rank solution
    n, m, p = DIMENSIONS
    rank_new = 54
=======
    # Fallback: Return the known 54-rank solution
    n, m, p = DIMENSIONS
    rank_new = 54
    # Try a few more greedy iterations with different random seeds if time permits
    if time.time() - start_time < DEADLINE - 30:
        # Try greedy with different candidate ordering (shuffle)
        for trial_seed in [1, 2, 3]:
            if time.time() - start_time > DEADLINE - 20:
                break
            np.random.seed(trial_seed)
            perm = np.random.permutation(num_cands)
            residual2 = T_target.copy()
            sel2 = []
            coeff2 = []
            for _ in range(80):
                if time.time() - start_time > DEADLINE - 20:
                    break
                dots2 = cand_vecs @ residual2
                reds2 = dots2**2 / cand_norms_sq_safe
                # Add small random perturbation to break ties
                reds2 += np.random.randn(num_cands) * 1e-10
                best2 = int(np.argmax(reds2))
                if reds2[best2] < 1e-10:
                    break
                c2 = dots2[best2] / cand_norms_sq_safe[best2]
                sel2.append(best2)
                coeff2.append(c2)
                residual2 -= c2 * cand_vecs[best2]
                if np.linalg.norm(residual2) < 1e-9:
                    break
            if len(sel2) < rank_new and np.linalg.norm(residual2) < 1e-8:
                # Apply rank reduction to this solution
                sv2 = cand_vecs[sel2]
                for _p2 in range(20):
                    if time.time() - start_time > DEADLINE - 20:
                        break
                    if len(sel2) <= 1:
                        break
                    ca2 = np.array([abs(c) for c in coeff2])
                    ord2 = np.argsort(ca2)
                    rem2 = False
                    for tr2 in ord2:
                        if len(sel2) <= 1:
                            break
                        rem_idx2 = [i for i in range(len(sel2)) if i != tr2]
                        if len(rem_idx2) == 0:
                            break
                        rv2 = sv2[rem_idx2]
                        try:
                            xs2 = np.linalg.lstsq(rv2.T, T_target, rcond=None)[0]
                            if np.linalg.norm(rv2 @ xs2 - T_target) < 1e-8:
                                sel2 = [sel2[i] for i in rem_idx2]
                                coeff2 = xs2.tolist()
                                sv2 = cand_vecs[sel2]
                                residual2 = T_target - sv2 @ np.array(coeff2)
                                rem2 = True
                                break
                        except:
                            pass
                    if not rem2:
                        break
                if len(sel2) < rank_new and np.linalg.norm(residual2) < 1e-8:
                    rank_new = len(sel2)
                    U_new = np.zeros((n * m, rank_new))
                    V_new = np.zeros((m * p, rank_new))
                    W_new = np.zeros((p * n, rank_new))
                    for r2, idx2 in enumerate(sel2):
                        c2 = coeff2[r2]
                        _, uc2, vc2, wc2 = candidates[idx2]
                        for (i, j), val in uc2.items():
                            U_new[i * m + j, r2] = val * c2
                        for (j, k), val in vc2.items():
                            V_new[j * p + k, r2] = val
                        for (k, i), val in wc2.items():
                            W_new[k * n + i, r2] = val
                    U_new = np.round(U_new * 2) / 2
                    V_new = np.round(V_new * 2) / 2
                    W_new = np.round(W_new * 2) / 2
                    # Verify
                    np.random.seed(42)
                    At = np.random.randint(-10, 10, (n, m))
                    Bt = np.random.randint(-10, 10, (m, p))
                    Ct = At @ Bt
                    Cc = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            val = 0
                            for j in range(m):
                                for r in range(rank_new):
                                    val += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                            Cc[i, k] = val
                    if np.allclose(Ct, Cc, atol=1e-5):
                        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
>>>>>>> REPLACE


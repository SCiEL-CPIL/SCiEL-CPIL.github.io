<<<<<<< SEARCH
    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 8:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        for temp, iters in ((0.4, 200), (0.15, 200), (0.05, 200)):
            if _now() > deadline - 8:
                break
            try:
                result = minimize(
                    _soft_objective,
                    x_start,
                    method='L-BFGS-B',
                    args=(temp, 256),
                    options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-9},
                )
                s_final = np.tanh(result.x)
                x_start = result.x
                sf = _score_sequence(s_final)
                if sf < best_score:
                    best_score = sf
                    best_s = s_final.copy()
                sf_b = _score_sequence(np.sign(s_final) * 1.0)
                if sf_b < best_score:
                    best_score = sf_b
                    best_s = np.sign(s_final)
                    best_s[best_s == 0] = 1.0
            except Exception:
                pass

    # ---------------- Phase 4: final discrete polish on best solution ----------------
    if best_s is not None and _now() < deadline - 5:
        cur = best_s.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = best_score
        for _ in range(200):
            if _now() > deadline - 5:
                break
            best_j = -1
            best_js = cur_score
            for j in range(N):
                ns = _max_ac_after_flip(cur, cur_ac, cur_sum, j)
                if ns < best_js:
                    best_js = ns
                    best_j = j
            if best_j == -1 or best_js >= cur_score - 1e-12:
                break
            val = cur[best_j]
            delta = -2.0 * val
            cur_ac = cur_ac.copy()
            cur_ac[best_j:best_j+N] += 2.0 * delta * cur
            cur_ac[best_j] += delta * delta
            cur[best_j] = -val
            cur_sum += delta
            cur_score = best_js
            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

    if best_s is None:
        return np.ones(N)
    return best_s
# EVOLVE-BLOCK-END
=======
    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 6:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        for temp, iters in ((0.3, 150), (0.1, 150), (0.03, 150)):
            if _now() > deadline - 6:
                break
            try:
                result = minimize(
                    _soft_objective,
                    x_start,
                    method='L-BFGS-B',
                    args=(temp, 256),
                    options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                s_final = np.tanh(result.x)
                x_start = result.x
                sf = _score_sequence(s_final)
                if sf < best_score:
                    best_score = sf
                    best_s = s_final.copy()
                sf_b = _score_sequence(np.sign(s_final) * 1.0)
                if sf_b < best_score:
                    best_score = sf_b
                    best_s = np.sign(s_final)
                    best_s[best_s == 0] = 1.0
            except Exception:
                pass

    # ---------------- Phase 4: final discrete polish on best solution ----------------
    if best_s is not None and _now() < deadline - 3:
        cur = best_s.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = best_score
        for _ in range(500):
            if _now() > deadline - 3:
                break
            scores = _batch_flip_scores(cur, cur_ac, cur_sum)
            best_j = int(np.argmin(scores))
            best_js = scores[best_j]
            if best_js >= cur_score - 1e-12:
                break
            cur_sum = _apply_flip_inplace(cur, cur_ac, cur_sum, best_j)
            cur_score = best_js
            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

    if best_s is None:
        return np.ones(N)
    return best_s
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

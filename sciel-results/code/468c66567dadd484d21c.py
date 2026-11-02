<<<<<<< SEARCH
def _peak_targeted_polish(x0, rng, max_iters=5000):
    """
    Targeted polish: identify the lag(s) contributing to the peak autoconvolution
    and apply targeted adjustments to reduce them.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    
    for _ in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        s = np.sum(best_x)
        s2 = s * s
        abs_ac = np.abs(ac) / s2
        
        # Find the top few lags contributing to the peak
        top_k = 5
        peak_lags = np.argsort(abs_ac)[-top_k:]
        peak_vals = abs_ac[peak_lags]
        
        # For each peak lag, try to reduce it by adjusting x
        improved = False
        for lag in peak_lags:
            lag = int(lag)
            if lag >= N:
                continue
            
            # ac[lag] = sum_n x[n]*x[n+lag]
            # To reduce |ac[lag]|, we can adjust x[lag] or nearby elements
            # Try small adjustments
            for idx in range(max(0, lag-2), min(N, lag+3)):
                for scale in [0.8, 0.9, 1.1, 1.2]:
                    x_trial = best_x.copy()
                    x_trial[idx] *= scale
                    
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try a broader random perturbation focused on peak regions
            n_perturb = rng.integers(3, 10)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, 0.01, size=n_perturb)
            x_trial = best_x + perturbation
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
            else:
                # No improvement in a while, try a larger perturbation
                if rng.random() < 0.05:
                    x_trial = best_x + rng.normal(0, 0.03, size=N)
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial.copy()
    
    return best_x, best_val
=======
def _peak_targeted_polish(x0, rng, max_iters=5000):
    """
    Targeted polish: identify the lag(s) contributing to the peak autoconvolution
    and apply targeted adjustments to reduce them. Vectorized for speed.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    
    for _ in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        s = np.sum(best_x)
        s2 = s * s
        abs_ac = np.abs(ac) / s2
        
        # Find the top few lags contributing to the peak
        top_k = 8
        peak_lags = np.argsort(abs_ac)[-top_k:]
        
        # Build candidate indices from peak lags (vectorized)
        candidate_indices = set()
        for lag in peak_lags:
            lag = int(lag)
            if lag >= N:
                continue
            for idx in range(max(0, lag - 3), min(N, lag + 4)):
                candidate_indices.add(idx)
        
        if not candidate_indices:
            candidate_indices = set(rng.choice(N, size=min(20, N), replace=False).tolist())
        
        cand_arr = np.array(sorted(candidate_indices))
        n_cand = len(cand_arr)
        
        # Try multiple scales on all candidates at once
        improved = False
        best_local_val = best_val
        best_local_idx = -1
        best_local_scale = 1.0
        
        for scale in [0.75, 0.85, 0.92, 1.08, 1.15, 1.25]:
            x_trial = best_x.copy()
            x_trial[cand_arr] *= scale
            val = _objective_only(x_trial)
            if val < best_local_val:
                best_local_val = val
                best_local_idx = -1  # means "all candidates"
                best_local_scale = scale
                improved = True
        
        # Also try individual index perturbations for finer control
        if not improved:
            for idx in cand_arr:
                for scale in [0.8, 0.9, 1.1, 1.2]:
                    x_trial = best_x.copy()
                    x_trial[idx] *= scale
                    val = _objective_only(x_trial)
                    if val < best_local_val:
                        best_local_val = val
                        best_local_idx = idx
                        best_local_scale = scale
                        improved = True
        
        if improved:
            x_trial = best_x.copy()
            if best_local_idx >= 0:
                x_trial[best_local_idx] *= best_local_scale
            else:
                x_trial[cand_arr] *= best_local_scale
            best_val = best_local_val
            best_x = x_trial.copy()
        else:
            # Try a broader random perturbation
            n_perturb = rng.integers(5, 15)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, 0.01, size=n_perturb)
            x_trial = best_x + perturbation
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
            else:
                if rng.random() < 0.05:
                    x_trial = best_x + rng.normal(0, 0.03, size=N)
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial.copy()
    
    return best_x, best_val
>>>>>>> REPLACE


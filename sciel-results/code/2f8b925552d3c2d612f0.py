"""Strategy: Nadaraya-Watson denoising + degree-5 polynomial fit with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nadaraya_watson(x, y, x_eval, bandwidth):
    """Nadaraya-Watson kernel regression estimator."""
    if bandwidth <= 0:
        return y[0] * np.ones_like(x_eval)
    # Vectorized: for each x_eval, compute weighted average of y
    # Use Gaussian kernel
    # To avoid memory blowup, process in chunks
    n_eval = len(x_eval)
    n_train = len(x)
    result = np.empty(n_eval)
    chunk_size = max(1, min(10000, n_eval))
    
    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        xe = x_eval[start:end]
        # diff[i, j] = xe[i] - x[j]
        diff = xe[:, None] - x[None, :]
        # Gaussian kernel weights
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        # Weighted sum
        num = weights @ y
        den = weights.sum(axis=1)
        # Avoid division by zero
        den_safe = np.where(den > 1e-30, den, 1.0)
        result[start:end] = num / den_safe
    
    return result


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    
    # Basic sanity checks
    if n < 6:
        return np.zeros(6)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.zeros(6)
    
    # Remove any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Sort by x for better behavior
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Compute x range
    x_min, x_max = x_sorted[0], x_sorted[-1]
    x_range = x_max - x_min
    if x_range < 1e-10:
        return np.zeros(6)
    
    # Baseline: unregularized least squares on full data
    V_full = np.vander(x, N=6, increasing=True)
    try:
        coeff_ls = np.linalg.lstsq(V_full, y, rcond=None)[0]
    except:
        coeff_ls = np.zeros(6)
    
    # Try Nadaraya-Watson denoising followed by polynomial fit
    # The idea: the subgrid tendency is a smooth function of x plus noise.
    # NW regression estimates the smooth function, then we fit a degree-5 poly to it.
    
    best_coeff = coeff_ls
    best_score = np.inf
    
    # Holdout validation split
    if n > 12:
        # Use a deterministic split: first 20% as validation
        n_val = max(6, int(0.2 * n))
        # Sort-based split: use sorted x, take extremes for validation to cover range
        # Actually, for a polynomial, random split is better for generalization
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        
        x_val = x[val_idx]
        y_val = y[val_idx]
        x_trn = x[trn_idx]
        y_trn = y[trn_idx]
        
        V_val = np.vander(x_val, N=6, increasing=True)
        
        # Candidate 1: plain LS on training split
        V_trn = np.vander(x_trn, N=6, increasing=True)
        try:
            c_ls_trn = np.linalg.lstsq(V_trn, y_trn, rcond=None)[0]
            if np.all(np.isfinite(c_ls_trn)):
                pred = V_val @ c_ls_trn
                mse = np.mean((pred - y_val) ** 2)
                if mse < best_score:
                    best_score = mse
                    best_coeff = c_ls_trn
        except:
            pass
        
        # Candidate 2: NW denoised then poly fit
        # Try different bandwidths for NW
        # Bandwidth in units of x
        # Use quantile-based bandwidths
        bandwidths = []
        # A few fixed fractions of range
        for frac in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
            bandwidths.append(frac * x_range)
        # Also try IQR-based
        q25, q75 = np.percentile(x_trn, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            bandwidths.append(0.5 * iqr)
        
        # Deduplicate and filter
        bandwidths = list(set([b for b in bandwidths if b > 0]))
        # Limit number to keep time bounded
        bandwidths = bandwidths[:8]
        
        for bw in bandwidths:
            if time.time() > deadline - 15:
                break
            
            # NW denoise the training data
            try:
                y_denoised = _nadaraya_watson(x_trn, y_trn, x_trn, bw)
            except:
                continue
            
            if not np.all(np.isfinite(y_denoised)):
                continue
            
            # Fit degree-5 polynomial to denoised data
            try:
                c_nw = np.linalg.lstsq(V_trn, y_denoised, rcond=None)[0]
            except:
                continue
            
            if not np.all(np.isfinite(c_nw)):
                continue
            
            # Score on holdout
            pred = V_val @ c_nw
            mse = np.mean((pred - y_val) ** 2)
            if mse < best_score:
                best_score = mse
                best_coeff = c_nw
        
        # Candidate 3: Ridge-regularized fits on training split
        alphas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
            
            if alpha == 0.0:
                try:
                    c = np.linalg.lstsq(V_trn, y_trn, rcond=None)[0]
                except:
                    continue
            else:
                A = V_trn.T @ V_trn + alpha * np.eye(6)
                b = V_trn.T @ y_trn
                try:
                    c = np.linalg.solve(A, b)
                except:
                    try:
                        c = np.linalg.lstsq(V_trn, y_trn, rcond=None)[0]
                    except:
                        continue
            
            if not np.all(np.isfinite(c)):
                continue
            
            pred = V_val @ c
            mse = np.mean((pred - y_val) ** 2)
            if mse < best_score:
                best_score = mse
                best_coeff = c
        
        # Also try NW denoised + ridge
        for bw in bandwidths[:4]:
            if time.time() > deadline - 5:
                break
            
            try:
                y_denoised = _nadaraya_watson(x_trn, y_trn, x_trn, bw)
            except:
                continue
            
            if not np.all(np.isfinite(y_denoised)):
                continue
            
            for alpha in [1e-6, 1e-4, 1e-3, 1e-2, 1.0]:
                if time.time() > deadline - 5:
                    break
                
                A = V_trn.T @ V_trn + alpha * np.eye(6)
                b = V_trn.T @ y_denoised
                try:
                    c = np.linalg.solve(A, b)
                except:
                    continue
                
                if not np.all(np.isfinite(c)):
                    continue
                
                pred = V_val @ c
                mse = np.mean((pred - y_val) ** 2)
                if mse < best_score:
                    best_score = mse
                    best_coeff = c
    
    # If we have time, also try fitting on full data with NW denoising
    if time.time() < deadline - 10:
        # Try a few bandwidths on full data
        for frac in [0.02, 0.05, 0.1]:
            if time.time() > deadline - 5:
                break
            bw = frac * x_range
            try:
                y_denoised_full = _nadaraya_watson(x, y, x, bw)
            except:
                continue
            
            if not np.all(np.isfinite(y_denoised_full)):
                continue
            
            try:
                c_full_nw = np.linalg.lstsq(V_full, y_denoised_full, rcond=None)[0]
            except:
                continue
            
            if np.all(np.isfinite(c_full_nw)):
                # We can't easily validate this without a split, but if we have a split,
                # we already did. For now, only use if it's not worse than LS on full data
                # Actually, let's just keep best_coeff from validation
                pass
    
    # Final safety check
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = coeff_ls if np.all(np.isfinite(coeff_ls)) else np.zeros(6)
    
    # Truncate very large coefficients to avoid numerical issues in evaluation
    max_mag = np.max(np.abs(best_coeff))
    if max_mag > 1e6:
        best_coeff = np.clip(best_coeff, -1e6, 1e6)
    
    return best_coeff
# EVOLVE-BLOCK-END

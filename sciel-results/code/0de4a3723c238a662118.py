"""Strategy: Kernel-weighted polynomial fit with NW denoising and polish pass."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nadaraya_watson_fast(x, y, x_eval, bandwidth):
    """Efficient NW kernel regression using sorted data and bisection for window."""
    if bandwidth <= 0:
        return y[0] * np.ones_like(x_eval)
    
    n_train = len(x)
    n_eval = len(x_eval)
    result = np.empty(n_eval)
    
    # Sort training data by x
    sort_idx = np.argsort(x)
    x_s = x[sort_idx]
    y_s = y[sort_idx]
    
    # Precompute cumulative sums for efficient kernel weighting
    # We'll use a sliding window approach with Gaussian kernel
    
    for i in range(n_eval):
        xi = x_eval[i]
        # Find approximate window using bisection
        # Points within ~4*bandwidth contribute meaningfully
        window = 4.0 * bandwidth
        lo = np.searchsorted(x_s, xi - window)
        hi = np.searchsorted(x_s, xi + window)
        
        if hi <= lo:
            # No points in window; use nearest point
            idx = np.clip(np.searchsorted(x_s, xi), 0, n_train - 1)
            result[i] = y_s[idx]
            continue
        
        x_w = x_s[lo:hi]
        y_w = y_s[lo:hi]
        diff = (x_w - xi) / bandwidth
        weights = np.exp(-0.5 * diff * diff)
        
        wsum = weights.sum()
        if wsum < 1e-30:
            result[i] = y_w[np.argmin(np.abs(x_w - xi))]
        else:
            result[i] = np.dot(weights, y_w) / wsum
    
    return result


def _kernel_weighted_polyfit(x, y, x_eval, bandwidth, degree=5):
    """Fit degree-5 polynomial minimizing weighted SSE with Gaussian kernel weights."""
    n = len(x)
    V = np.vander(x, N=degree + 1, increasing=True)
    
    # Compute weights for each training point
    # w_i = sum_j K(x_i - x_j)  (leave-one-out style or full)
    # For simplicity, use full kernel weights
    # This is O(n^2) but we'll do it in chunks
    
    weights = np.ones(n)
    if bandwidth > 0:
        chunk = max(1, min(5000, n))
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            diff = x[start:end, None] - x[None, :]
            w = np.exp(-0.5 * (diff / bandwidth) ** 2)
            # Leave-one-out: subtract self-weight
            w[np.arange(end - start), np.arange(start, end)] -= 1.0
            weights[start:end] = w.sum(axis=1)
    
    # Ensure positive weights
    weights = np.maximum(weights, 1e-10)
    
    # Weighted least squares
    W = np.sqrt(weights)
    Vw = V * W[:, None]
    yw = y * W
    
    try:
        coeff, _, _, _ = np.linalg.lstsq(Vw, yw, rcond=None)
    except:
        coeff = np.linalg.lstsq(V, y, rcond=None)[0]
    
    return coeff


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    
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
    
    # Compute x range
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-10:
        return np.zeros(6)
    
    # Center and scale x for numerical stability
    x_mean = (x_min + x_max) / 2
    x_scale = x_range / 2
    x_centered = (x - x_mean) / x_scale  # in [-1, 1]
    
    # Baseline: unregularized least squares on full data
    V_full = np.vander(x_centered, N=6, increasing=True)
    try:
        coeff_ls = np.linalg.lstsq(V_full, y, rcond=None)[0]
    except:
        coeff_ls = np.zeros(6)
    
    best_coeff = coeff_ls
    best_score = np.inf
    
    # Holdout validation split
    if n > 12:
        n_val = max(6, int(0.2 * n))
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        
        x_val = x[val_idx]
        y_val = y[val_idx]
        x_trn = x[trn_idx]
        y_trn = y[trn_idx]
        
        xc_val = (x_val - x_mean) / x_scale
        xc_trn = (x_trn - x_mean) / x_scale
        
        V_val = np.vander(xc_val, N=6, increasing=True)
        V_trn = np.vander(xc_trn, N=6, increasing=True)
        
        def _score(coeff):
            """Compute validation MSE."""
            pred = V_val @ coeff
            return np.mean((pred - y_val) ** 2)
        
        # Candidate 1: plain LS on training split
        try:
            c_ls_trn = np.linalg.lstsq(V_trn, y_trn, rcond=None)[0]
            if np.all(np.isfinite(c_ls_trn)):
                s = _score(c_ls_trn)
                if s < best_score:
                    best_score = s
                    best_coeff = c_ls_trn
        except:
            pass
        
        # Bandwidths to try (in original x units)
        bandwidths = []
        for frac in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]:
            bandwidths.append(frac * x_range)
        q25, q75 = np.percentile(x_trn, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            bandwidths.append(0.5 * iqr)
            bandwidths.append(1.0 * iqr)
        
        bandwidths = list(set([b for b in bandwidths if b > 0]))
        bandwidths.sort()
        bandwidths = bandwidths[:12]
        
        # Candidate 2: NW denoised then poly fit
        for bw in bandwidths:
            if time.time() > deadline - 20:
                break
            
            try:
                y_denoised = _nadaraya_watson_fast(x_trn, y_trn, x_trn, bw)
            except:
                continue
            
            if not np.all(np.isfinite(y_denoised)):
                continue
            
            try:
                c_nw = np.linalg.lstsq(V_trn, y_denoised, rcond=None)[0]
            except:
                continue
            
            if not np.all(np.isfinite(c_nw)):
                continue
            
            s = _score(c_nw)
            if s < best_score:
                best_score = s
                best_coeff = c_nw
        
        # Candidate 3: Kernel-weighted polynomial fit
        for bw in bandwidths[:8]:
            if time.time() > deadline - 15:
                break
            
            try:
                c_kw = _kernel_weighted_polyfit(x_trn, y_trn, x_trn, bw, degree=5)
            except:
                continue
            
            if not np.all(np.isfinite(c_kw)):
                continue
            
            s = _score(c_kw)
            if s < best_score:
                best_score = s
                best_coeff = c_kw
        
        # Candidate 4: Ridge-regularized fits
        alphas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        
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
                    continue
            
            if not np.all(np.isfinite(c)):
                continue
            
            s = _score(c)
            if s < best_score:
                best_score = s
                best_coeff = c
        
        # Candidate 5: NW denoised + ridge
        for bw in bandwidths[:5]:
            if time.time() > deadline - 8:
                break
            
            try:
                y_denoised = _nadaraya_watson_fast(x_trn, y_trn, x_trn, bw)
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
                
                s = _score(c)
                if s < best_score:
                    best_score = s
                    best_coeff = c
    
    # Polish pass: if we have time, refine using all data with the best method
    if time.time() < deadline - 10:
        # Determine which approach worked best and apply to full data
        # Try NW denoising + LS on full data with a few bandwidths
        for bw in bandwidths[:4]:
            if time.time() > deadline - 5:
                break
            
            try:
                y_denoised_full = _nadaraya_watson_fast(x, y, x, bw)
            except:
                continue
            
            if not np.all(np.isfinite(y_denoised_full)):
                continue
            
            try:
                c_full = np.linalg.lstsq(V_full, y_denoised_full, rcond=None)[0]
            except:
                continue
            
            if np.all(np.isfinite(c_full)):
                # Compare with current best using full-data self-score as proxy
                # Only update if significantly better in self-prediction
                pred_full = V_full @ c_full
                mse_full = np.mean((pred_full - y) ** 2)
                pred_best = V_full @ best_coeff
                mse_best = np.mean((pred_best - y) ** 2)
                # Use full-data fit only if it's comparable (avoid overfitting check)
                if mse_full < mse_best * 1.1:
                    best_coeff = c_full
        
        # Final polish: kernel-weighted fit on full data
        if time.time() < deadline - 3:
            for bw in bandwidths[:3]:
                if time.time() > deadline - 2:
                    break
                try:
                    c_kwf = _kernel_weighted_polyfit(x, y, x, bw, degree=5)
                    if np.all(np.isfinite(c_kwf)):
                        pred = V_full @ c_kwf
                        mse = np.mean((pred - y) ** 2)
                        pred_best = V_full @ best_coeff
                        mse_best = np.mean((pred_best - y) ** 2)
                        if mse < mse_best * 1.05:
                            best_coeff = c_kwf
                except:
                    continue
    
    # Final safety check
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = coeff_ls if np.all(np.isfinite(coeff_ls)) else np.zeros(6)
    
    # Truncate very large coefficients
    max_mag = np.max(np.abs(best_coeff))
    if max_mag > 1e6:
        best_coeff = np.clip(best_coeff, -1e6, 1e6)
    
    return best_coeff
# EVOLVE-BLOCK-END

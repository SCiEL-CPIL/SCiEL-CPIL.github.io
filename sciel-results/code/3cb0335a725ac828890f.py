import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Ensure inputs are 1D numpy arrays
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter out any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n_samples = len(x)
    if n_samples == 0:
        return np.zeros(6)
    
    if n_samples < 10:
        # Fallback to simple fit if data is too sparse
        try:
            deg = min(5, n_samples - 1)
            if deg >= 0:
                coeff = np.polynomial.polynomial.polyfit(x, y, deg)
                out = np.zeros(6)
                out[:len(coeff)] = coeff
                return out
        except Exception:
            pass
        return np.zeros(6)
    
    # Strategy:
    # 1. Nadaraya-Watson Denoising to estimate E[y|x]
    # 2. Initial Ridge Regression on denoised data
    # 3. Iterative Reweighting (Huber-like) on original data to robustly align polynomial
    
    # --- Step 1: Nadaraya-Watson Denoising ---
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.zeros(6)
    
    # Create a grid for density estimation
    # Use a grid size that balances resolution and memory (N * G)
    # If N is very large, reduce G.
    n_grid = 500
    if n_samples > 100000:
        n_grid = 200
        
    grid = np.linspace(x_min, x_max, n_grid)
    
    # Estimate bandwidth. Silverman's rule of thumb: 1.06 * sigma * n^(-1/5)
    # But for NW regression, we might want a slightly wider bandwidth to smooth out fast-scale noise.
    # Let's try a few bandwidths and pick the one that minimizes residual variance on a holdout set?
    # No, we don't have holdout. We have to use the same data.
    # Cross-validation on the denoised curve? Or just a heuristic.
    # The knowledge suggests: "Tuning the Nadaraya-Watson bandwidth via a small grid search with validation-based selection should find a better smoothing level".
    # Since we don't have a separate validation set, we can use Leave-One-Out or just a heuristic based on the scale of the signal.
    # Let's try a small grid of bandwidths and select the one that minimizes the "roughness" of the resulting curve (second derivative norm) or similar.
    # Or, simply use a robust estimate of scale.
    
    x_std = np.std(x)
    if x_std == 0:
        return np.zeros(6)
    
    # Heuristic bandwidths: 0.1, 0.2, 0.5, 1.0 times std
    bw_candidates = [0.1 * x_std, 0.2 * x_std, 0.5 * x_std, 1.0 * x_std]
    
    best_y_smooth = None
    best_score = np.inf
    
    for h_bw in bw_candidates:
        if h_bw <= 0:
            continue
        try:
            # Compute weights for each grid point
            # dist is (N, G)
            dist = np.abs(x[:, np.newaxis] - grid[np.newaxis, :])
            # Gaussian kernel
            w_grid = np.exp(-0.5 * (dist / h_bw) ** 2)
            
            w_sum = np.sum(w_grid, axis=0)
            y_w_sum = np.sum(w_grid * y[:, np.newaxis], axis=0)
            
            # Avoid division by zero
            y_smooth = np.where(w_sum > 1e-12, y_w_sum / w_sum, np.mean(y))
            
            # Score: We want a smooth curve. A common metric is the integrated squared second derivative.
            # Approximate second derivative using finite differences on the grid.
            dy = np.diff(y_smooth)
            d2y = np.diff(dy)
            roughness = np.sum(d2y ** 2)
            
            # We also want the curve to actually fit the data reasonably well.
            # Let's interpolate back to x and compute RMSE?
            # That's expensive. Let's just use roughness for now, assuming the signal is smoother than noise.
            # However, if h_bw is too large, we lose signal.
            # Let's add a term for the variance of the residuals from the smooth curve.
            # Interpolate y_smooth back to x
            y_smooth_at_x = np.interp(x, grid, y_smooth)
            residual_var = np.mean((y - y_smooth_at_x) ** 2)
            
            # Combine: We want low residual var and low roughness.
            # Normalize?
            # Let's just minimize residual_var + lambda * roughness / (grid_spacing^2) ?
            # Grid spacing is (x_max - x_min) / (n_grid - 1)
            grid_spacing = (x_max - x_min) / (n_grid - 1)
            # Second derivative approx: (y[i+1] - 2y[i] + y[i-1]) / h^2
            # So sum(d2y^2) is sum of (numerator^2). To get derivative^2, divide by h^4.
            # Let's just use a simple heuristic score.
            score = residual_var + 1e-6 * roughness / (grid_spacing ** 2)
            
            if score < best_score:
                best_score = score
                best_y_smooth = y_smooth
                
        except Exception:
            continue
            
    if best_y_smooth is None:
        # Fallback: simple binning
        n_bins = 100
        try:
            counts, bin_edges = np.histogram(x, bins=n_bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            valid_bins = counts > 0
            if np.any(valid_bins):
                x_binned = bin_centers[valid_bins]
                weights = counts[valid_bins]
                sums, _ = np.histogram(x, bins=n_bins, weights=y)
                y_binned = sums[valid_bins] / weights
                # Interpolate to original x
                best_y_smooth = np.interp(x, x_binned, y_binned)
            else:
                best_y_smooth = y
        except Exception:
            best_y_smooth = y
            
    # Interpolate the smooth curve back to the original x points
    # If best_y_smooth is already on the grid, we need to map it back.
    # In the loop above, y_smooth was on the grid.
    # We need to be careful: if we selected best_y_smooth from the loop, it's on 'grid'.
    # If we fell back to binning, best_y_smooth is already at x (or we need to interp).
    # Let's assume best_y_smooth is a vector of same length as x.
    # In the loop, we computed y_smooth_at_x for scoring, but stored y_smooth (on grid).
    # Let's re-interpolate the selected best_y_smooth (on grid) to x.
    
    # Wait, the loop stored `y_smooth` which is on the grid.
    # We need `y_denoised` at the original x locations.
    # Let's re-derive it.
    
    # Actually, it's easier to just compute the denoised values for x directly if we pick a bandwidth.
    # But we selected a bandwidth implicitly. Let's just use the last computed one or re-run for the best one.
    # To save time, let's just interpolate the grid result.
    
    # The variable `best_y_smooth` is on `grid`.
    y_denoised = np.interp(x, grid, best_y_smooth)
    
    # --- Step 2: Initial Ridge Regression ---
    V = np.vander(x, N=6, increasing=True)
    
    # Regularization lambda
    # trace(V.T V) / N is a good scale
    VtV = V.T @ V
    trace_VtV = np.trace(VtV)
    lam = 1e-6 * trace_VtV / n_samples
    I = np.eye(6)
    
    try:
        c_init = np.linalg.solve(VtV + lam * I, V.T @ y_denoised)
    except np.linalg.LinAlgError:
        c_init = np.polynomial.polynomial.polyfit(x, y_denoised, 5)
        
    if time.time() > deadline - 5.0:
        return c_init
        
    # --- Step 3: Iterative Reweighting (Robust Fit) ---
    # We want to fit the polynomial to the original data `y`, but downweight points
    # where the fast-scale noise is high.
    # The initial fit `c_init` is based on the denoised data, so it represents the "true" mean trend.
    # Residuals `r = y - V @ c_init` represent the noise + model error.
    # We use Huber weights to downweight large residuals.
    
    # We can iterate this a few times.
    c_current = c_init
    for _ in range(3):
        y_pred = V @ c_current
        residuals = y - y_pred
        r_abs = np.abs(residuals)
        
        # MAD-based scale
        delta_huber = 1.4826 * np.median(r_abs)
        if delta_huber == 0:
            delta_huber = 1e-6
            
        # Compute weights
        w = np.ones(n_samples)
        mask_large = r_abs > delta_huber
        w[mask_large] = delta_huber / r_abs[mask_large]
        
        # Re-fit with weighted least squares
        sqrt_w = np.sqrt(w)
        V_w = V * sqrt_w[:, np.newaxis]
        y_w = y * sqrt_w
        
        # Use a slightly larger lambda for stability in robust fit
        lam_robust = 1e-5 * trace_VtV / n_samples
        
        try:
            VtV_w = V_w.T @ V_w
            VTy_w = V_w.T @ y_w
            c_new = np.linalg.solve(VtV_w + lam_robust * I, VTy_w)
            
            if not np.all(np.isfinite(c_new)):
                break
                
            c_current = c_new
            
        except np.linalg.LinAlgError:
            break
            
    if time.time() > deadline - 2.0:
        return c_current
        
    return c_current

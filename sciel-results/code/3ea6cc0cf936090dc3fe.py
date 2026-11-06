"""Strategy: Robust Polynomial Fit with Outlier Removal via Residual-based Thresholding."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 10:
        return np.zeros(6)
    
    # 1. Initial Least Squares Fit on Scaled Data
    # Scale x to improve conditioning of the Vandermonde matrix
    x_scale = np.max(np.abs(x))
    if x_scale < 1e-12:
        x_scale = 1.0
    x_scaled = x / x_scale
    
    deg = 5
    V = np.vander(x_scaled, N=deg + 1, increasing=True)
    
    try:
        # Initial fit
        coeffs_scaled, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(6)
    
    # 2. Compute Residuals and Identify Outliers
    y_pred = V @ coeffs_scaled
    residuals_vec = y - y_pred
    
    # Use Median Absolute Deviation (MAD) for robust outlier detection
    # MAD is less sensitive to outliers than standard deviation
    med = np.median(residuals_vec)
    mad = np.median(np.abs(residuals_vec - med))
    
    if mad < 1e-12:
        # If MAD is zero (e.g., perfect fit or constant), fall back to std
        std = np.std(residuals_vec)
        if std < 1e-12:
            # No outliers to remove or constant data
            coeffs_final_scaled = coeffs_scaled
        else:
            # Use 3-sigma rule
            threshold = 3.0 * std
            mask = np.abs(residuals_vec - med) < threshold
            if np.sum(mask) > deg + 1:
                V_sub = V[mask]
                y_sub = y[mask]
                try:
                    coeffs_final_scaled, _, _, _ = np.linalg.lstsq(V_sub, y_sub, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs_final_scaled = coeffs_scaled
            else:
                coeffs_final_scaled = coeffs_scaled
    else:
        # Use 4*MAD rule (approx 2.5 sigma for normal distribution, but more robust)
        # 4 * 1.4826 * MAD ~ 2 sigma, 5 * 1.4826 ~ 2.5 sigma
        # Let's use a factor of 4.0 on the MAD-based scale
        threshold = 4.0 * mad
        mask = np.abs(residuals_vec - med) < threshold
        
        if np.sum(mask) > deg + 1:
            V_sub = V[mask]
            y_sub = y[mask]
            try:
                coeffs_final_scaled, _, _, _ = np.linalg.lstsq(V_sub, y_sub, rcond=None)
            except np.linalg.LinAlgError:
                coeffs_final_scaled = coeffs_scaled
        else:
            coeffs_final_scaled = coeffs_scaled
            
    # 3. Transform Coefficients Back to Original x Scale
    # p(x_scaled) = sum(c_i_scaled * (x/scale)^i)
    # p(x) = sum(c_i_scaled * scale^{-i} * x^i)
    powers = np.arange(deg + 1)
    coeffs_original = coeffs_final_scaled * (x_scale ** (-powers))
    
    # 4. Check Time
    if time.time() > deadline:
        return coeffs_original
        
    # 5. Optional: Iterate Robust Fit (Refit on cleaned data, check for new outliers)
    # One more iteration might help if the first pass removed significant outliers
    # but shifted the mean/variance significantly.
    # However, to save time and avoid overfitting to noise, we stop here.
    # If we had more time, we could do 2-3 iterations.
    
    return coeffs_original
# EVOLVE-BLOCK-END

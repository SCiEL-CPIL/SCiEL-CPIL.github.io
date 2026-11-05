# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a degree-5 polynomial using iteratively reweighted least squares (IRLS)
    with Huber loss for robustness against outliers in the subgrid tendency.
    
    Strategy:
    1. Scale x to z in [-1, 1] for numerical conditioning.
    2. Use Chebyshev basis for the polynomial fit (better conditioning than monomial).
    3. Apply IRLS with Huber loss: iteratively update weights based on residuals,
       down-weighting points with large residuals (potential outliers).
    4. Convert Chebyshev coefficients back to monomial coefficients in x.
    
    This is a structural change from binning: instead of discarding information via
    averaging, we robustly fit all data points while reducing the influence of
    outliers that may arise from the chaotic two-scale dynamics.
    """
    t_start = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    n = len(x)
    
    if n < 6:
        deg = min(5, max(0, n - 1))
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full_coeff = np.zeros(6)
            full_coeff[:deg+1] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6)
    
    # Scale x to [-1, 1] for numerical stability
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.array([np.mean(y)] + [0.0]*5)
    
    x_center = (x_min + x_max) / 2.0
    x_scale = x_range / 2.0
    z = (x - x_center) / x_scale
    
    # Chebyshev design matrix: T_k(z) for k=0..5
    # T_0 = 1, T_1 = z, T_2 = 2z^2 - 1, etc.
    # Use recurrence: T_{k+1} = 2z*T_k - T_{k-1}
    T = np.zeros((n, 6))
    T[:, 0] = 1.0
    T[:, 1] = z
    for k in range(2, 6):
        T[:, k] = 2.0 * z * T[:, k-1] - T[:, k-2]
    
    # IRLS with Huber loss
    # Huber: loss(r) = 0.5*r^2 if |r| <= delta, else delta*|r| - 0.5*delta^2
    # Weight: w(r) = 1 if |r| <= delta, else delta/|r|
    
    # Initial fit: ordinary least squares
    try:
        coeff_z, _, _, _ = np.linalg.lstsq(T, y, rcond=None)
    except Exception:
        try:
            coeff_z = np.polynomial.chebyshev.chebfit(z, y, 5)
            if len(coeff_z) < 6:
                coeff_z = np.pad(coeff_z, (0, 6 - len(coeff_z)))
        except Exception:
            return np.zeros(6)
    
    if not np.all(np.isfinite(coeff_z)):
        return np.zeros(6)
    
    # Determine Huber delta from residual scale (MAD-based)
    def _huber_delta(residuals):
        """Estimate robust scale for Huber delta."""
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        # Scale MAD to estimate std (assuming approximate normality)
        sigma_est = 1.4826 * mad
        if sigma_est < 1e-12:
            sigma_est = np.std(residuals) if np.std(residuals) > 1e-12 else 1.0
        return 1.5 * sigma_est  # Huber delta
    
    # IRLS iterations
    max_iter = 30
    tol = 1e-10
    best_coeff_z = coeff_z.copy()
    best_cost = np.inf
    
    for it in range(max_iter):
        if time.time() - t_start > DEADLINE:
            break
        
        # Compute residuals
        pred = T @ coeff_z
        resid = y - pred
        
        # Compute Huber delta
        delta = _huber_delta(resid)
        
        # Compute weights
        abs_r = np.abs(resid)
        w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-15))
        
        # Weighted least squares
        sqrt_w = np.sqrt(w)
        T_w = T * sqrt_w[:, None]
        y_w = y * sqrt_w
        
        try:
            new_coeff_z, _, _, _ = np.linalg.lstsq(T_w, y_w, rcond=None)
        except Exception:
            break
        
        if not np.all(np.isfinite(new_coeff_z)):
            break
        
        # Check convergence
        diff = np.max(np.abs(new_coeff_z - coeff_z))
        coeff_z = new_coeff_z
        
        # Track best based on Huber cost
        pred = T @ coeff_z
        resid = y - pred
        abs_r = np.abs(resid)
        huber_loss = np.where(abs_r <= delta, 0.5 * abs_r**2, delta * abs_r - 0.5 * delta**2)
        cost = np.mean(huber_loss)
        
        if cost < best_cost:
            best_cost = cost
            best_coeff_z = coeff_z.copy()
        
        if diff < tol:
            break
    
    coeff_z = best_coeff_z
    
    # Convert Chebyshev coefficients in z to monomial coefficients in x
    # P(x) = sum_{k=0}^{5} c_k T_k(z), where z = (x - x_center)/x_scale
    # T_k(z) is a polynomial in z of degree k
    # We need to express P as sum_{j=0}^{5} b_j x^j
    
    # First, express each T_k(z) as a polynomial in z:
    # T_0(z) = 1
    # T_1(z) = z
    # T_2(z) = 2z^2 - 1
    # T_3(z) = 4z^3 - 3z
    # T_4(z) = 8z^4 - 8z^2 + 1
    # T_5(z) = 16z^5 - 20z^3 + 5z
    
    # Build the matrix M such that [T_0, T_1, ..., T_5] = M @ [1, z, z^2, z^3, z^4, z^5]
    M = np.zeros((6, 6))
    M[0, 0] = 1.0
    M[1, 1] = 1.0
    M[2, 0] = -1.0
    M[2, 2] = 2.0
    M[3, 1] = -3.0
    M[3, 3] = 4.0
    M[4, 0] = 1.0
    M[4, 2] = -8.0
    M[4, 4] = 8.0
    M[5, 1] = 5.0
    M[5, 3] = -20.0
    M[5, 5] = 16.0
    
    # P(z) = c^T T(z) = c^T M [1, z, ..., z^5]^T
    # So coefficients in z-basis: a_z = M^T @ coeff_z
    a_z = M.T @ coeff_z  # a_z[j] is coefficient of z^j
    
    # Now substitute z = (x - x_center)/x_scale
    # z^j = (x - x_center)^j / x_scale^j
    # (x - x_center)^j = sum_{i=0}^{j} binom(j, i) x^i (-x_center)^(j-i)
    
    b = np.zeros(6)
    for j in range(6):
        if a_z[j] == 0:
            continue
        coeff_scale = a_z[j] / (x_scale ** j)
        for i in range(j + 1):
            # binom(j, i)
            binom_val = 1.0
            j_temp = min(i, j - i)
            for k in range(1, j_temp + 1):
                binom_val *= (j - j_temp + k) / k
            term = coeff_scale * binom_val * ((-x_center) ** (j - i))
            b[i] += term
    
    if not np.all(np.isfinite(b)):
        # Fallback: direct monomial fit on scaled data
        try:
            coeff_direct = np.polynomial.polynomial.polyfit(z, y, 5)
            # Transform from z to x
            b2 = np.zeros(6)
            for j in range(6):
                if coeff_direct[j] == 0:
                    continue
                coeff_scale = coeff_direct[j] / (x_scale ** j)
                for i in range(j + 1):
                    binom_val = 1.0
                    j_temp = min(i, j - i)
                    for k in range(1, j_temp + 1):
                        binom_val *= (j - j_temp + k) / k
                    term = coeff_scale * binom_val * ((-x_center) ** (j - i))
                    b2[i] += term
            b = b2
        except Exception:
            return np.zeros(6)
    
    return b
# EVOLVE-BLOCK-END

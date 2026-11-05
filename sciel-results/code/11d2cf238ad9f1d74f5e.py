"""Robust Cauchy-loss polynomial closure with SVD-based solving."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Standardize x for numerical conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    xs = (x - x_mean) / x_std

    # Standardize y to improve conditioning of the fit
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    ys = (y - y_mean) / y_std

    degree = 5
    V = np.vander(xs, degree + 1, increasing=True)

    # SVD for numerical stability in solving
    # We will solve (V^T W V) c = V^T W ys
    # But V is ill-conditioned. Let's use the property that if we standardize,
    # the Vandermonde is better conditioned.
    # We use an iterative reweighting scheme with Cauchy loss.
    
    w = np.ones(n)
    c_s = np.zeros(degree + 1)
    
    # Pre-compute SVD of V for potential use, but for weighted LS, 
    # we effectively solve a different system each time.
    # However, we can use np.linalg.lstsq which uses SVD internally.
    
    for _ in range(30):
        # Weighted least squares: minimize sum w_i * (V_i c - ys_i)^2
        # This is equivalent to least squares on sqrt(w_i) * V_i c = sqrt(w_i) * ys_i
        sqrt_w = np.sqrt(w)
        V_w = V * sqrt_w[:, np.newaxis]
        ys_w = ys * sqrt_w
        
        try:
            c_s, _, _, _ = np.linalg.lstsq(V_w, ys_w, rcond=None)
        except np.linalg.LinAlgError:
            c_s = np.zeros(degree + 1)
            
        # Compute residuals
        resid = ys - V @ c_s
        
        # Update weights based on Cauchy loss
        # Cauchy loss: rho(r) = 0.5 * log(1 + (r/delta)^2)
        # Gradient weight: w = 1 / (1 + (r/delta)^2)
        # delta is the scale parameter. Use MAD or IQR for robust scale.
        abs_resid = np.abs(resid)
        # Robust scale estimation
        median_resid = np.median(abs_resid)
        if median_resid < 1e-12:
            delta = 1.0
        else:
            # Scale delta to match the typical residual magnitude
            # 1.4826 * MAD is consistent estimator of sigma for normal dist
            delta = 1.4826 * median_resid * 1.5 
            
        w = 1.0 / (1.0 + (abs_resid / delta) ** 2)
        
        # Check for convergence
        if np.allclose(w, np.ones(n), atol=1e-8):
            break

    # Unstandardize the polynomial
    # We fit: ys = sum c_i * xs^i
    # xs = (x - x_mean)/x_std
    # ys = (y - y_mean)/y_std
    # y = y_mean + y_std * sum c_i * ((x - x_mean)/x_std)^i
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    d = degree
    coeff = np.zeros(6)
    
    # Expansion:
    # ((x - mu)/sigma)^i = sum_{k=0}^i binom(i,k) (x/sigma)^k (-mu/sigma)^(i-k)
    # = sum_{k=0}^i binom(i,k) a^k b^(i-k) x^k
    
    from math import comb
    for k in range(6):
        inner = 0.0
        for i in range(k, d + 1):
            inner += c_s[i] * comb(i, k) * (a ** k) * (b ** (i - k))
        coeff[k] = y_std * inner
    coeff[0] += y_mean

    # Ensure finite
    coeff = np.where(np.isfinite(coeff), coeff, 0.0)
    return coeff
# EVOLVE-BLOCK-END

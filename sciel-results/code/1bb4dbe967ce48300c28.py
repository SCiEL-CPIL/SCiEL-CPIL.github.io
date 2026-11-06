import numpy as np

def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # 1. Standardize x to improve numerical stability of monomial fit
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
        
    x_std_scaled = (x - x_mean) / x_std
    
    # 2. Construct Monomial Design Matrix in Standardized Space
    n_samples = len(x)
    D = np.zeros((n_samples, 6))
    D[:, 0] = 1.0
    for i in range(1, 6):
        D[:, i] = D[:, i-1] * x_std_scaled
        
    # 3. Solve for coefficients d in standardized space
    # Use SVD for better stability than direct solve or lstsq with rcond
    try:
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        # Regularization: threshold small singular values
        # Determine rank
        tol = np.finfo(float).eps * max(D.shape) * s[0]
        rank = np.sum(s > tol)
        
        # If rank deficient, use pseudo-inverse with cutoff
        # We want to solve D d = t
        # d = Vt.T * (s / (s^2 + lambda)) * U.T * t
        # Simple ridge regression with small lambda for stability
        lambda_reg = 1e-8 * (s[0] ** 2) if s[0] > 0 else 1e-8
        
        d = Vt.T @ (s / (s**2 + lambda_reg) * (U.T @ t))
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        d, _, _, _ = np.linalg.lstsq(D, t, rcond=None)
        
    if not np.all(np.isfinite(d)):
        return np.zeros(6)
        
    # 4. Convert coefficients from Standardized Space to Original Space
    # P(x_std) = sum_{i=0}^5 d_i * x_std^i
    # x_std = (x - x_mean) / x_std
    # P(x) = sum_{i=0}^5 d_i * ((x - x_mean) / x_std)^i
    #      = sum_{i=0}^5 (d_i / x_std^i) * (x - x_mean)^i
    
    # We need coefficients c_j such that P(x) = sum_{j=0}^5 c_j x^j
    # Expand (x - x_mean)^i = sum_{k=0}^i binom(i, k) x^k (-x_mean)^(i-k)
    
    c = np.zeros(6)
    for i in range(6):
        coeff_i = d[i] / (x_std ** i)
        if coeff_i == 0:
            continue
        # Expand (x - mu)^i
        # Terms are binom(i, k) * x^k * (-mu)^(i-k)
        for k in range(i + 1):
            # binom(i, k)
            binom_val = 1.0
            for m in range(k):
                binom_val *= (i - m)
                binom_val /= (m + 1)
            
            term_coeff = coeff_i * binom_val * ((-x_mean) ** (i - k))
            c[k] += term_coeff
            
    if not np.all(np.isfinite(c)):
        return np.zeros(6)
        
    return c

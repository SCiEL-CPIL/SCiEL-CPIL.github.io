import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # If n is very large, subsample for kernel regression to ensure speed
    # But keep all points for the final fit if possible? 
    # Kernel regression is O(N^2). If N=100k, that's 10^10 ops, too slow.
    # We should subsample for the denoising step.
    max_samples_for_smoothing = 5000
    if n > max_samples_for_smoothing:
        idx = np.random.choice(n, max_samples_for_smoothing, replace=False)
        x_smooth = x[idx]
        t_smooth = t[idx]
    else:
        x_smooth = x
        t_smooth = t
    
    # Estimate bandwidth for Nadaraya-Watson
    # Using Silverman's rule of thumb for 1D
    std_x = np.std(x_smooth)
    if std_x < 1e-10:
        return np.zeros(6)
    n_smooth = len(x_smooth)
    h = 1.06 * std_x * (n_smooth ** (-0.2))
    
    # Compute denoised targets using Gaussian Kernel Regression
    # We compute m(x_i) for each x_i in the smoothed set
    # To be efficient, we can vectorize the calculation or loop.
    # Looping over 5000 points with vectorized inner products is fast.
    t_denoised = np.zeros_like(t_smooth)
    
    # For efficiency, we can compute the kernel weights for all pairs at once if memory allows.
    # 5000x5000 matrix is 25M floats, ~200MB. Acceptable.
    # If memory is a concern, we can chunk it.
    
    # Let's use a chunked approach to be safe with memory and time.
    chunk_size = 1000
    for i in range(0, n_smooth, chunk_size):
        end = min(i + chunk_size, n_smooth)
        x_chunk = x_smooth[i:end]
        # Compute distances
        diff = x_chunk[:, np.newaxis] - x_smooth[np.newaxis, :]
        # Gaussian kernel
        weights = np.exp(-0.5 * (diff / h) ** 2)
        # Sum weights and weighted targets
        w_sum = np.sum(weights, axis=1)
        wt_sum = np.sum(weights * t_smooth[np.newaxis, :], axis=1)
        # Avoid division by zero
        t_denoised[i:end] = wt_sum / np.where(w_sum > 1e-10, w_sum, 1.0)
        
        if time.time() > deadline:
            # Fallback to simple linear fit if taking too long
            coeff = np.zeros(6)
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, t, 1)
            except:
                pass
            return coeff

    # Now fit a degree 5 polynomial to (x, t_denoised)
    # Use all original x points? Or just the smoothed ones?
    # Using the smoothed points (x_smooth, t_denoised) is consistent with the denoising.
    # However, we want the polynomial to represent the global trend.
    # It's better to fit on the denoised data corresponding to the training x's.
    # If we subsampled, we have (x_smooth, t_denoised).
    # If we didn't subsample, we have (x, t_denoised).
    
    # Scale x for numerical stability of polynomial fit
    mu = np.mean(x_smooth)
    sigma = np.std(x_smooth)
    if sigma < 1e-10:
        sigma = 1.0
        
    x_scaled = (x_smooth - mu) / sigma
    
    # Fit degree 5 polynomial
    # V matrix: [1, x, x^2, ..., x^5]
    # Use np.polynomial.polynomial.polyfit
    try:
        # polyfit returns coefficients in ascending order
        coeff_scaled = np.polynomial.polynomial.polyfit(x_scaled, t_denoised, 5)
    except Exception:
        # Fallback to lower degree if rank deficient or error
        try:
            coeff_scaled = np.polynomial.polynomial.polyfit(x_scaled, t_denoised, 3)
            # Pad with zeros
            temp = np.zeros(6)
            temp[:4] = coeff_scaled
            coeff_scaled = temp
        except:
            return np.zeros(6)

    # Transform coefficients from scaled x to original x
    # p_scaled(x_scaled) = sum c_i ( (x-mu)/sigma )^i
    # We need c'_j such that sum c'_j x^j = sum c_i ( (x-mu)/sigma )^i
    # This is a linear transformation of the coefficients.
    # Let y = (x-mu)/sigma => x = sigma*y + mu
    # We want to express (sigma*y + mu)^j in terms of y? No, we have coeffs in y, want coeffs in x.
    # Actually, it's easier to think:
    # P(y) = c0 + c1*y + ... + c5*y^5
    # y = (x-mu)/sigma
    # P(x) = c0 + c1*(x-mu)/sigma + ... + c5*((x-mu)/sigma)^5
    # We need to expand this in powers of x.
    
    # Method: Compute the transformation matrix or use polynomial arithmetic.
    # Let's use the property that if we have coefficients in basis B1 (powers of x_scaled)
    # and want coefficients in basis B2 (powers of x), we can use the conversion.
    # Or simpler: 
    # Let Q(x) = P((x-mu)/sigma).
    # We can evaluate P at 6 points, interpolate, get coeffs? No, too error-prone.
    
    # Use np.polynomial.polynomial.polyfromroots? No.
    # Use the conversion formula.
    # The change of basis from (x-mu)/sigma to x is affine.
    # Let z = x. y = (z-mu)/sigma.
    # We have P(y) = sum_{i=0}^5 c_i y^i.
    # Substitute y:
    # P(z) = sum_{i=0}^5 c_i ( (z-mu)/sigma )^i
    #       = sum_{i=0}^5 c_i sigma^{-i} (z-mu)^i
    # Let Q(u) = sum_{i=0}^5 c_i sigma^{-i} u^i, where u = z-mu.
    # Now we need to expand Q(u) in powers of z, given u = z-mu.
    # This is a Taylor shift.
    # We can use the binomial expansion or a simple iterative multiplication.
    
    # Let's compute the coefficients of Q(u) first:
    c_u = coeff_scaled / (sigma ** np.arange(6))
    
    # Now convert from powers of (z-mu) to powers of z.
    # Let R(z) = Q(z-mu).
    # We can use the polynomial shift algorithm.
    # Or, since degree is low (5), we can just expand manually or use a small matrix.
    # Matrix M such that [1, z, z^2, z^3, z^4, z^5]^T = M @ [1, (z-mu), (z-mu)^2, ...]^T ?
    # Actually, it's easier to compute the coefficients of (z-mu)^k in terms of z.
    # (z-mu)^k = sum_{j=0}^k binom(k,j) z^j (-mu)^{k-j}
    
    coeff_original = np.zeros(6)
    for i in range(6):
        # Term c_u[i] * (z-mu)^i
        # Expand (z-mu)^i
        # (z-mu)^i = sum_{j=0}^i binom(i,j) z^j (-mu)^{i-j}
        for j in range(i+1):
            # Coefficient for z^j
            binom_coeff = 1.0
            # Calculate binom(i, j)
            # i is small, so we can compute directly
            if j == 0 or j == i:
                binom_val = 1.0
            else:
                # Calculate C(i, j)
                # C(i, j) = i! / (j! (i-j)!)
                # Since i <= 5, we can hardcode or use a simple loop
                num = 1
                for k in range(j):
                    num *= (i - k)
                den = 1
                for k in range(j):
                    den *= (k + 1)
                binom_val = num / den
            coeff_original[j] += c_u[i] * binom_val * ((-mu) ** (i - j))
            
    return coeff_original

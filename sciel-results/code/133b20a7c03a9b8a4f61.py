# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # --- Step 1: Nadaraya-Watson Smoothing on y given x ---
    # This denoises the target variable to capture the underlying smooth dynamics.
    
    # Robust scale estimate
    median_x = np.median(x)
    mad = np.median(np.abs(x - median_x))
    if mad < 1e-10:
        scale_x = np.std(x)
        if scale_x < 1e-10:
            scale_x = 1.0
    else:
        # Convert MAD to std equivalent (assuming normality approx)
        scale_x = mad / 0.6745

    # Grid of bandwidths to try (relative to scale_x)
    h_rels = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    
    # Fast CV for bandwidth selection
    max_cv_samples = 2000
    if n > max_cv_samples:
        idx_cv = np.random.choice(n, max_cv_samples, replace=False)
        x_cv = x[idx_cv]
        y_cv = y[idx_cv]
        n_cv = max_cv_samples
    else:
        x_cv = x
        y_cv = y
        n_cv = n

    best_h = scale_x * 0.5
    best_cv_mse = np.inf

    for h_rel in h_rels:
        h = h_rel * scale_x
        # Compute NW smoothed y
        diff = x_cv[:, None] - x_cv[None, :]
        u = diff / h
        K = np.exp(-0.5 * u * u)
        
        weights_sum = np.sum(K, axis=1)
        weights_sum[weights_sum == 0] = 1e-10
        
        y_smooth = np.sum(K * y_cv[None, :], axis=1) / weights_sum
        
        mse = np.mean((y_smooth - y_cv) ** 2)
        
        if mse < best_cv_mse:
            best_cv_mse = mse
            best_h = h

    # Apply the best bandwidth to the FULL dataset to get smoothed y
    y_smoothed = np.zeros_like(y)
    chunk_size = 10000
    
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        x_chunk = x[i:j]
        diff = x_chunk[:, None] - x[None, :]
        u = diff / best_h
        K = np.exp(-0.5 * u * u)
        
        weights_sum = np.sum(K, axis=1)
        weights_sum[weights_sum == 0] = 1e-10
        
        y_smoothed[i:j] = np.sum(K * y[None, :], axis=1) / weights_sum

    # Check time
    if time.time() > deadline - 10:
        # Fallback: return simple mean or zero if we ran out of time
        # Better to return a simple LS fit on raw data if time is tight
        try:
            x_mean = np.mean(x)
            x_std = np.std(x)
            if x_std < 1e-10: x_std = 1.0
            x_norm = (x - x_mean) / x_std
            V = np.vander(x_norm, N=6, increasing=True)
            coeff = np.linalg.lstsq(V, y, rcond=None)[0]
            coeff_orig = _transform_coeffs(coeff, x_mean, x_std)
            return coeff_orig
        except:
            return np.zeros(6)

    # --- Step 2: Fit Degree-5 Polynomial to Smoothed y using Legendre Basis ---
    # We map x to a scaled range [-1, 1] where Legendre polynomials are defined.
    # This orthogonal basis eliminates multicollinearity, making the normal equations
    # well-conditioned and the fit more stable against noise.
    
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
        x_center = x_min
    else:
        x_center = (x_min + x_max) / 2.0
    
    # Map x to z in [-1, 1]
    # z = (x - x_center) / (x_range / 2)
    # x = z * (x_range / 2) + x_center
    z = (x - x_center) / (x_range / 2.0)
    
    # Construct Legendre design matrix P (n x 6)
    # P[i, k] = P_k(z_i)
    P = np.zeros((n, 6))
    P[:, 0] = 1.0
    P[:, 1] = z
    # Recurrence: (k+1)P_{k+1}(z) = (2k+1)z P_k(z) - k P_{k-1}(z)
    for k in range(1, 5):
        P[:, k+1] = ((2*k + 1) * z * P[:, k] - k * P[:, k-1]) / (k + 1)
        
    # Fit coefficients in Legendre basis
    # We use Ridge regression with a small fixed lambda for stability,
    # but since Legendre is orthogonal, LS is usually very stable.
    # Let's do a quick lambda selection on a holdout set to be safe.
    n_holdout = max(10, min(1000, n // 10))
    indices = np.random.permutation(n)
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]
    
    P_train = P[train_idx]
    y_train = y_smoothed[train_idx]
    P_hold = P[holdout_idx]
    y_hold = y_smoothed[holdout_idx]
    
    PtP = P_train.T @ P_train
    Pty = P_train.T @ y_train
    
    # Small range of lambdas
    max_lambda = np.trace(PtP) / 6.0
    if max_lambda == 0:
        max_lambda = 1.0
    
    lambdas = np.logspace(-8, 2, 50) * max_lambda
    
    best_mse = np.inf
    best_lambda = 0.0
    
    for lam in lambdas:
        A = PtP + lam * np.eye(6)
        try:
            coeff_leg = np.linalg.solve(A, Pty)
        except np.linalg.LinAlgError:
            continue
        
        pred_hold = P_hold @ coeff_leg
        mse = np.mean((pred_hold - y_hold) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam
            
        if time.time() > deadline - 5:
            break

    # Refit on full data with best lambda
    PtP_full = P.T @ P
    Pty_full = P.T @ y_smoothed
    A_full = PtP_full + best_lambda * np.eye(6)
    
    try:
        coeff_leg_full = np.linalg.solve(A_full, Pty_full)
        if not np.all(np.isfinite(coeff_leg_full)):
            coeff_leg_full = np.linalg.lstsq(P, y_smoothed, rcond=None)[0]
    except:
        coeff_leg_full = np.linalg.lstsq(P, y_smoothed, rcond=None)[0]

    # Convert Legendre coefficients to Monomial coefficients
    # P(x) = sum_{k=0}^5 c_k P_k(z(x))
    # z(x) = a*x + b, where a = 2/x_range, b = -2*x_center/x_range
    # We need to express P_k(a*x + b) as a polynomial in x.
    # Then sum c_k * (polynomial in x).
    
    a = 2.0 / x_range
    b = -2.0 * x_center / x_range
    
    # coeff_monom[j] is the coefficient of x^j
    coeff_monom = np.zeros(6)
    
    # For each degree k Legendre polynomial, expand P_k(a*x + b) in powers of x
    # P_k(z) is a polynomial of degree k.
    # We can compute the expansion for each k up to 5.
    
    # P_0(z) = 1
    # P_1(z) = z
    # P_2(z) = (3z^2 - 1)/2
    # P_3(z) = (5z^3 - 3z)/2
    # P_4(z) = (35z^4 - 30z^2 + 3)/8
    # P_5(z) = (63z^5 - 70z^3 + 15z)/8
    
    # Let's do this programmatically to avoid hardcoding errors and handle general k if needed (though max is 5)
    # We represent polynomials as arrays of coefficients [c0, c1, ..., ck] for x^0, x^1, ..., x^k
    
    def expand_legendre(k, a, b):
        """
        Returns coefficients of P_k(a*x + b) in powers of x.
        """
        if k == 0:
            return np.array([1.0])
        if k == 1:
            # P_1(z) = z = a*x + b
            return np.array([b, a])
            
        # Recursive expansion
        # P_k(z) = ((2k-1)z P_{k-1}(z) - (k-1)P_{k-2}(z)) / k
        # Let Q_{k-1}(x) = P_{k-1}(a*x + b) = sum c_j x^j
        # Let Q_{k-2}(x) = P_{k-2}(a*x + b) = sum d_j x^j
        # z Q_{k-1}(x) = (a*x + b) Q_{k-1}(x) = a x Q_{k-1}(x) + b Q_{k-1}(x)
        
        Q_k_minus_1 = expand_legendre(k-1, a, b)
        Q_k_minus_2 = expand_legendre(k-2, a, b)
        
        # Multiply Q_{k-1} by (a*x + b)
        # Result length is len(Q_k_minus_1) + 1
        term1 = np.zeros(len(Q_k_minus_1) + 1)
        # a * x * Q_{k-1}
        term1[1:] += a * Q_k_minus_1
        # b * Q_{k-1}
        term1[:len(Q_k_minus_1)] += b * Q_k_minus_1
        
        # Subtract (k-1) * Q_{k-2}
        # Q_{k-2} length is k-1, term1 length is k+1
        term1[:len(Q_k_minus_2)] -= (k-1) * Q_k_minus_2
        
        # Divide by k
        return term1 / k
    
    for k in range(6):
        c_k = coeff_leg_full[k]
        if c_k == 0:
            continue
        expansion = expand_legendre(k, a, b)
        # Add c_k * expansion to coeff_monom
        # expansion has length k+1
        coeff_monom[:len(expansion)] += c_k * expansion
        
    if not np.all(np.isfinite(coeff_monom)):
        coeff_monom = np.zeros(6)

    return coeff_monom

# EVOLVE-BLOCK-END

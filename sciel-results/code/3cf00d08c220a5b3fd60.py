import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    N = len(x)
    if N == 0:
        return np.zeros(6)
    
    # 1. Subsampling for efficiency and robustness
    M = min(N, 2000)
    if N > M:
        indices = np.random.choice(N, M, replace=False)
        x_sub = x[indices]
        y_sub = y[indices]
    else:
        x_sub = x
        y_sub = y
        M = N
        
    # 2. Scaling for polynomial fit stability
    mu = np.mean(x_sub)
    sigma = np.std(x_sub)
    if sigma < 1e-6:
        sigma = 1.0
    z_sub = (x_sub - mu) / sigma
    
    # 3. Determine optimal bandwidth for kernel smoothing via Cross-Validation
    # Kernel: Gaussian K(u) = exp(-0.5 * u^2)
    # Bandwidths to try: scaled by sigma
    # Silverman's rule of thumb for bandwidth: h = 1.06 * sigma * N^(-1/5)
    # We'll try a range around this
    h_base = 1.06 * sigma * (M ** (-0.2))
    # Try relative bandwidths
    h_candidates = h_base * np.array([0.2, 0.5, 1.0, 2.0, 4.0])
    
    best_h = h_base
    best_mse = np.inf
    
    # Simple 5-Fold CV on the subsample
    k_folds = 5
    indices_cv = np.arange(M)
    np.random.shuffle(indices_cv)
    fold_size = M // k_folds
    
    for h in h_candidates:
        total_mse = 0.0
        count = 0
        for i in range(k_folds):
            if time.time() > deadline:
                break
                
            val_idx = indices_cv[i*fold_size : (i+1)*fold_size]
            train_idx = np.concatenate([indices_cv[:i*fold_size], indices_cv[(i+1)*fold_size:]])
            
            x_train = x_sub[train_idx]
            y_train = y_sub[train_idx]
            x_val = x_sub[val_idx]
            y_val = y_sub[val_idx]
            
            # Compute smoothed y for validation points using training points
            # y_smooth(x_val) = sum(K((x_val - x_train)/h) * y_train) / sum(K((x_val - x_train)/h))
            # Vectorized computation
            # diff shape (len(x_val), len(x_train))
            diff = (x_val[:, None] - x_train[None, :]) / h
            weights = np.exp(-0.5 * diff**2)
            
            # Avoid division by zero
            denom = np.sum(weights, axis=1)
            denom[denom < 1e-12] = 1e-12
            numer = np.sum(weights * y_train[None, :], axis=1)
            
            y_pred = numer / denom
            
            mse = np.mean((y_val - y_pred)**2)
            total_mse += mse
            count += 1
            
        if count > 0:
            avg_mse = total_mse / count
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_h = h
                
    # 4. Smooth the entire subsample using the best h
    # We will fit the polynomial to these smoothed points
    # y_smooth(x_sub)
    diff = (x_sub[:, None] - x_sub[None, :]) / best_h
    weights = np.exp(-0.5 * diff**2)
    denom = np.sum(weights, axis=1)
    denom[denom < 1e-12] = 1e-12
    numer = np.sum(weights * y_sub[None, :], axis=1)
    y_smooth = numer / denom
    
    # 5. Fit Degree-5 Polynomial to (z_sub, y_smooth)
    # z_sub is scaled x
    # We fit y_smooth = sum(a_k * z_sub^k)
    try:
        # Use lstsq for better conditioning than polyfit sometimes, but polyfit is convenient
        # Construct Vandermonde matrix
        V = np.vander(z_sub, 6, increasing=True)
        # Solve V @ a = y_smooth
        a, _, _, _ = np.linalg.lstsq(V, y_smooth, rcond=None)
    except:
        # Fallback to simple linear fit if something goes wrong
        a = np.zeros(6)
        if M > 1:
            a[:2] = np.polyfit(z_sub, y_smooth, 1)[::-1] # polyfit returns descending, we need ascending
            # Wait, np.polyfit returns coefficients for highest power first.
            # If deg=1, returns [c1, c0]. We need [c0, c1].
            # So a[0] = c0, a[1] = c1.
            # Let's just use polyfit correctly in the main path too?
            # np.polynomial.polynomial.polyfit takes ascending order.
            pass

    # Let's use np.polynomial.polynomial.polyfit which handles ascending order directly
    # It uses least squares internally.
    try:
        a = np.polynomial.polynomial.polyfit(z_sub, y_smooth, 5)
        # Ensure a has length 6
        if len(a) < 6:
            a = np.pad(a, (0, 6-len(a)))
    except:
        a = np.zeros(6)
        # Fallback linear
        if M > 1:
            a[:2] = np.polynomial.polynomial.polyfit(z_sub, y_smooth, 1)

    # 6. Convert coefficients from scaled basis z=(x-mu)/sigma to original basis x
    # a_k are coeffs for z^k
    # We want c_j for x^j
    # z = (x-mu)/sigma
    # z^k = (1/sigma^k) * (x-mu)^k
    # (x-mu)^k = sum_{j=0}^k binom(k,j) x^j (-mu)^(k-j)
    # So term k contributes to c_j: a_k * (1/sigma^k) * binom(k,j) * (-mu)^(k-j)
    
    c = np.zeros(6)
    for k in range(6):
        if a[k] == 0:
            continue
        coeff_k = a[k] / (sigma ** k)
        for j in range(k + 1):
            # binom(k, j)
            binom_val = 1
            if j > 0:
                # Calculate binom(k,j) iteratively or using math.comb
                # k is small (<=5), so simple loop is fine
                # binom(k,j) = k! / (j! (k-j)!)
                # Using math.comb is cleaner
                import math
                binom_val = math.comb(k, j)
            
            pow_mu = (-mu) ** (k - j)
            c[j] += coeff_k * binom_val * pow_mu
            
    return c

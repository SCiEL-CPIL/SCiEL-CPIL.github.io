# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    if len(x) == 0:
        return np.zeros(6)

    # Fallback: simple degree-5 polyfit if time is tight
    if time.time() > deadline - 10:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except Exception:
            return np.zeros(6)

    # Stage 1: Kernel-weighted denoising to estimate E[Y|X]
    n = len(x)
    
    # Compute a reasonable bandwidth based on data spread
    x_range = np.ptp(x)
    if x_range < 1e-10:
        bandwidth = 1.0
    else:
        # Silverman's rule of thumb adapted for 1D
        bandwidth = 1.06 * np.std(x) * (n ** (-1.0 / 5.0))
        if bandwidth < 1e-6:
            bandwidth = x_range * 0.1
        bandwidth = max(bandwidth, x_range * 0.05)

    # Use a subset of points for efficiency if n is large
    max_knn = 2000
    if n > max_knn:
        # Randomly subsample for kernel evaluation
        idx = np.random.RandomState(42).choice(n, size=max_knn, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y

    # Evaluate kernel weights at each training point using subsampled points
    # For efficiency, we'll do this in chunks
    chunk_size = 1000
    y_smoothed = np.zeros(n)
    weights_confidence = np.zeros(n) # Store unnormalized weight sums for final fit weighting

    for i in range(0, n, chunk_size):
        if time.time() > deadline - 15:
            # Fallback to simple polyfit
            try:
                coeff = np.polynomial.polynomial.polyfit(x, y, 5)
                return coeff
            except Exception:
                return np.zeros(6)
        
        i_end = min(i + chunk_size, n)
        x_chunk = x[i:i_end]
        
        # Compute kernel weights: K(x_chunk, x_sub) / sum
        # Reshape for broadcasting: (chunk_size, 1) and (1, len(x_sub))
        diff = (x_chunk[:, np.newaxis] - x_sub[np.newaxis, :]) / bandwidth
        raw_weights = np.exp(-0.5 * diff ** 2)
        
        # Normalize weights for averaging
        weight_sums = np.sum(raw_weights, axis=1, keepdims=True)
        weight_sums_safe = np.where(weight_sums < 1e-12, 1.0, weight_sums)
        normalized_weights = raw_weights / weight_sums_safe
        
        # Compute weighted average
        y_smoothed[i:i_end] = normalized_weights @ y_sub
        
        # Store confidence (sum of raw weights) for weighted polynomial fit
        weights_confidence[i:i_end] = np.sum(raw_weights, axis=1)

    # Stage 2: Fit degree-5 polynomial to smoothed pseudo-labels
    # Use Weighted Least Squares where weights are based on data density (confidence)
    try:
        # Build Vandermonde matrix for degree 5
        # A = [1, x, x^2, x^3, x^4, x^5]
        # To avoid numerical issues with large x, we might need to scale x, 
        # but polyfit does this internally. For manual WLS, we should be careful.
        # Let's stick to standard polyfit if WLS is too complex/risky, 
        # but WLS is better.
        
        # Simple WLS implementation
        # c = (A^T W A)^{-1} A^T W y
        # A is (n, 6)
        # W is (n, n) diagonal
        
        # To save memory and time, we can compute the Gram matrix directly.
        # G = sum_i w_i * v_i * v_i^T
        # b = sum_i w_i * v_i * y_i
        # where v_i is the row of A.
        
        # This is O(n * 6^2) which is fine.
        
        # However, if x is large, x^5 is huge.
        # np.polynomial.polynomial.polyfit handles scaling.
        # Let's try to use polyfit on a subset if WLS is too slow?
        # No, n is likely < 100k. 6x6 solve is trivial.
        
        # Let's use a centered/scaled x for the WLS to improve conditioning.
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-10:
            x_scaled = x
        else:
            x_scaled = (x - x_mean) / x_std
            
        # Construct A for scaled x
        # A[:, j] = x_scaled^j
        A = np.vander(x_scaled, 6, increasing=True)
        
        # Weights
        w = weights_confidence
        # Normalize weights to prevent overflow/underflow in accumulation
        w_sum = np.sum(w)
        if w_sum < 1e-10:
            w = np.ones(n)
        else:
            w = w / w_sum # Average weight is 1/n? No, sum is 1.
            # Actually, for WLS, scaling of w doesn't matter for the solution.
            
        # Compute G = A^T W A
        # A^T W = A.T * w[:, None]
        # G = (A.T * w[:, None]) @ A
        G = (A.T * w[:, None]) @ A
        
        # Compute b = A^T W y_smoothed
        b = (A.T * w[:, None]) @ y_smoothed
        
        # Solve G c = b
        try:
            c_scaled = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            # Fallback to polyfit
            coeff = np.polynomial.polynomial.polyfit(x, y_smoothed, 5)
            return coeff

        # Convert coefficients from scaled x to original x
        # P(x) = sum c_j ((x - mean)/std)^j
        # We need coefficients for x^k.
        # This requires polynomial expansion.
        # It's easier to just return the scaled coefficients? 
        # NO, the interface requires coefficients in ASCENDING powers of LOCAL SLOW STATE x.
        # So we must transform.
        
        # Alternatively, just use polyfit on the raw x and y_smoothed, 
        # but ignore the weighting? 
        # The weighting is the key improvement.
        
        # Let's transform the polynomial.
        # y = c0 + c1*z + c2*z^2 + c3*z^3 + c4*z^4 + c5*z^5
        # z = (x - mu)/sigma = x/sigma - mu/sigma
        # Let a = 1/sigma, b = -mu/sigma. z = a*x + b.
        # This is a polynomial composition.
        # We can use numpy.polynomial.polynomial.polymul or similar.
        
        # z = a*x + b
        # z^k = sum_{j=0}^k binomial(k, j) a^j b^{k-j} x^j
        
        coeff_orig = np.zeros(6)
        a = 1.0 / x_std if x_std > 1e-10 else 1.0
        b = -x_mean / x_std if x_std > 1e-10 else 0.0
        
        for j in range(6):
            cj = c_scaled[j]
            # Expand (a*x + b)^j
            # Coeff of x^i in (a*x+b)^j is binom(j, i) a^i b^(j-i)
            for i in range(j + 1):
                coeff_orig[i] += cj * (np.math.comb(j, i) * (a ** i) * (b ** (j - i)))
                
        return coeff_orig

    except Exception:
        # Fallback to direct fit on smoothed data
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y_smoothed, 5)
        except Exception:
            try:
                coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            except Exception:
                coeff = np.zeros(6)
                
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END

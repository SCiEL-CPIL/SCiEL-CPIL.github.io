import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import binned_statistic

def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n < 10:
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    # 1. Binning and Spline Fit
    n_knots = min(20, max(5, n // 10))
    quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
    knot_values = np.quantile(x, quantiles)
    
    x_min, x_max = np.min(x), np.max(x)
    margin = 0.05 * (x_max - x_min)
    knot_values = np.concatenate([[x_min - margin], knot_values, [x_max + margin]])
    knot_values = np.sort(np.unique(knot_values))
    
    if len(knot_values) < 4:
        coeff = np.zeros(6)
        coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
        return coeff

    # Bin data for spline
    n_bins = len(knot_values)
    bin_edges = np.concatenate([
        [knot_values[0] - (knot_values[1] - knot_values[0]) / 2],
        (knot_values[:-1] + knot_values[1:]) / 2,
        [knot_values[-1] + (knot_values[-1] - knot_values[-2]) / 2]
    ])
    
    bin_means, _, _ = binned_statistic(x, y, statistic='mean', bins=bin_edges)
    valid_mask = np.isfinite(bin_means)
    
    if np.sum(valid_mask) < 4:
        # Fallback
        coeff = np.zeros(6)
        coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
        return coeff

    x_knots = knot_values[valid_mask]
    y_knots = bin_means[valid_mask]
    
    try:
        cs = CubicSpline(x_knots, y_knots, bc_type='natural')
    except Exception:
        coeff = np.zeros(6)
        coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
        return coeff

    # 2. Evaluate Spline at Original X
    y_smooth = cs(x)
    
    # 3. Center and Scale X for Stable Polyfit
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    t = (x - x_mean) / x_std
    
    # 4. Fit Poly in Scaled Space
    try:
        a_scaled = np.polynomial.polynomial.polyfit(t, y_smooth, 5)
    except Exception:
        coeff = np.zeros(6)
        coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
        return coeff

    # 5. Convert Coefficients from Scaled (t) to Original (x) Basis
    # P(x) = sum(c_i x^i)
    # Q(t) = sum(a_j t^j) with t = (x - mu) / sigma => x = sigma * t + mu
    # Q(t) = P(sigma * t + mu)
    # We need c such that sum(a_j t^j) = sum(c_i (sigma * t + mu)^i)
    # Matrix A where A[j, i] is coeff of t^j in (sigma * t + mu)^i
    # a = A @ c => c = inv(A) @ a
    
    A = np.zeros((6, 6))
    for i in range(6):
        # Expand (sigma * t + mu)^i
        # Coeff of t^j is binom(i, j) * sigma^j * mu^(i-j)
        for j in range(i + 1):
            # Binomial coefficient
            binom = 1.0
            if i > 0:
                # Calculate C(i, j)
                # Using math.comb is safer if available, or manual
                # i is small (<=5), so manual is fine
                top = i
                bottom = j
                # C(i, j) = i! / (j! (i-j)!)
                # For small i, we can just use a helper or simple logic
                # Let's use a simple factorial based calc or precomputed
                # Actually, scipy.special.comb is available
                pass 
            # Let's just use scipy.special.comb
            from scipy.special import comb
            binom = comb(i, j, exact=False)
            A[j, i] = binom * (x_std ** j) * (x_mean ** (i - j))
            
    try:
        c_original = np.linalg.solve(A, a_scaled)
    except Exception:
        # Fallback if solve fails
        c_original = np.zeros(6)
        c_original[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
        
    if not np.all(np.isfinite(c_original)):
        c_original = np.zeros(6)
        c_original[:4] = np.polynomial.polynomial.polyfit(x, y, 3)

    return c_original

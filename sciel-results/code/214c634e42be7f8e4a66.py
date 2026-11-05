"""Piecewise cubic spline closure for Lorenz-96 subgrid tendency, approximated by degree-5 polynomial."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import CubicSpline


def construct(training_x, training_subgrid_tendency):
    """
    Fit a piecewise cubic spline to the training data using robust knot selection,
    then approximate the spline by a degree-5 polynomial using weighted least squares
    on Chebyshev nodes. Returns 6 coefficients in ascending powers of x.
    
    Key improvements:
    1. Use median-based binning for more robust knot values
    2. Use Chebyshev nodes for polynomial approximation (better conditioned)
    3. Use weighted least squares with weights based on local data density
    4. Regularize the polynomial fit to prevent overfitting
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n < 10:
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    # Determine number of knots based on data size
    n_knots = min(15, max(5, n // 20))
    
    # Create quantile-based knots from the training data
    quantiles = np.linspace(0.05, 0.95, n_knots)
    knot_values = np.quantile(x, quantiles)
    
    # Add boundary points with small margin
    x_min, x_max = np.min(x), np.max(x)
    margin = 0.02 * (x_max - x_min)
    knot_values = np.concatenate([[x_min - margin], knot_values, [x_max + margin]])
    
    # Sort and ensure unique knots
    knot_values = np.sort(np.unique(knot_values))
    
    if len(knot_values) < 4:
        coeff = np.zeros(6)
        coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
        return coeff
    
    # Compute binned medians for robust knot values
    n_bins = len(knot_values)
    bin_edges = np.concatenate([
        [knot_values[0] - (knot_values[1] - knot_values[0]) / 2],
        (knot_values[:-1] + knot_values[1:]) / 2,
        [knot_values[-1] + (knot_values[-1] - knot_values[-2]) / 2]
    ])
    
    # Use binned median for robustness
    bin_medians = np.empty(n_bins)
    for i in range(n_bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_medians[i] = np.median(y[mask])
        else:
            bin_medians[i] = np.nan
    
    valid_mask = np.isfinite(bin_medians)
    if np.sum(valid_mask) < 4:
        # Fallback: subsample and fit spline directly
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        if len(x_sorted) > 150:
            subsample_idx = np.linspace(0, len(x_sorted) - 1, 150, dtype=int)
            x_sorted = x_sorted[subsample_idx]
            y_sorted = y_sorted[subsample_idx]
        try:
            cs = CubicSpline(x_sorted, y_sorted, bc_type='natural')
        except Exception:
            coeff = np.zeros(6)
            coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
            return coeff
    else:
        x_knots = knot_values[valid_mask]
        y_knots = bin_medians[valid_mask]
        try:
            cs = CubicSpline(x_knots, y_knots, bc_type='natural')
        except Exception:
            coeff = np.zeros(6)
            coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
            return coeff
    
    # Evaluate spline on Chebyshev nodes for better-conditioned polynomial fit
    x_lo, x_hi = x_min - margin, x_max + margin
    n_eval = 200
    # Chebyshev nodes in [x_lo, x_hi]
    theta = np.pi * (np.arange(n_eval) + 0.5) / n_eval
    x_cheb = 0.5 * (x_hi + x_lo) + 0.5 * (x_hi - x_lo) * np.cos(theta)
    
    y_eval = cs(x_cheb)
    
    # Compute weights based on local data density
    # Points where we have more training data should have higher weight
    weights = np.ones(n_eval)
    for i in range(n_eval):
        # Count training points within a small window around x_cheb[i]
        window = 0.02 * (x_max - x_min)
        mask = np.abs(x - x_cheb[i]) < window
        count = np.sum(mask)
        weights[i] = 1.0 + 0.5 * np.log1p(count)
    
    # Normalize weights
    weights = weights / np.mean(weights)
    
    # Fit degree-5 polynomial using weighted least squares
    try:
        # Use weighted polyfit
        V = np.vander(x_cheb, 6, increasing=True)
        W = np.diag(weights)
        # Solve (V^T W V) c = V^T W y
        A = V.T @ W @ V
        b = V.T @ W @ y_eval
        # Add small regularization for stability
        reg = 1e-8 * np.trace(A) / 6
        A += reg * np.eye(6)
        coeff = np.linalg.solve(A, b)
    except Exception:
        try:
            coeff = np.polynomial.polynomial.polyfit(x_cheb, y_eval, 5)
        except Exception:
            coeff = np.zeros(6)
            coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
    
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
    
    return coeff
# EVOLVE-BLOCK-END

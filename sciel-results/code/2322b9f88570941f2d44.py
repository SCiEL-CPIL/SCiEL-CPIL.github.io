"""Rewritten program: piecewise cubic spline closure for Lorenz-96 subgrid tendency."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import binned_statistic


def construct(training_x, training_subgrid_tendency):
    """
    Fit a piecewise cubic spline (natural boundary conditions) to the training data,
    then approximate the resulting spline by a degree-5 polynomial via least squares
    on a dense grid. Returns 6 coefficients in ascending powers of x.
    
    The spline is defined by quantile-based knots derived from the training data,
    which better captures the non-polynomial, oscillatory structure of the
    Lorenz-96 subgrid tendency while maintaining smoothness.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n < 10:
        # Fallback to linear fit if too few points
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    # Determine number of knots based on data size
    # Use quantile-based knots for better coverage of the domain
    n_knots = min(20, max(5, n // 10))  # Between 5 and 20 knots
    
    # Create quantile-based knots from the training data
    quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]  # Exclude endpoints
    knot_values = np.quantile(x, quantiles)
    
    # Add boundary points (min and max of training data, slightly extended)
    x_min, x_max = np.min(x), np.max(x)
    margin = 0.05 * (x_max - x_min)
    knot_values = np.concatenate([[x_min - margin], knot_values, [x_max + margin]])
    
    # Sort and ensure unique knots
    knot_values = np.sort(np.unique(knot_values))
    
    # If we have too few unique knots, fall back to fewer
    if len(knot_values) < 4:
        # Use a simple cubic fit
        coeff = np.zeros(6)
        coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
        return coeff
    
    # Fit a cubic spline with natural boundary conditions
    # Use the knot values as x-coordinates with binned means as y-coordinates
    # This reduces noise and provides a smoother spline
    
    # Bin the data around the knots
    n_bins = len(knot_values)
    bin_edges = np.concatenate([
        [knot_values[0] - (knot_values[1] - knot_values[0]) / 2],
        (knot_values[:-1] + knot_values[1:]) / 2,
        [knot_values[-1] + (knot_values[-1] - knot_values[-2]) / 2]
    ])
    
    # Compute binned means
    bin_means, bin_edges_final, _ = binned_statistic(
        x, y, statistic='mean', bins=bin_edges
    )
    
    # Filter out bins with no data
    valid_mask = np.isfinite(bin_means)
    if np.sum(valid_mask) < 4:
        # Fallback: use direct spline interpolation on sorted data
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        # Subsample if too many points
        if len(x_sorted) > 100:
            subsample_idx = np.linspace(0, len(x_sorted) - 1, 100, dtype=int)
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
        y_knots = bin_means[valid_mask]
        try:
            cs = CubicSpline(x_knots, y_knots, bc_type='natural')
        except Exception:
            coeff = np.zeros(6)
            coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
            return coeff
    
    # Now approximate the spline by a degree-5 polynomial
    # Evaluate the spline on a dense grid within the training range
    n_eval = 1000
    x_eval = np.linspace(x_min - margin, x_max + margin, n_eval)
    y_eval = cs(x_eval)
    
    # Fit a degree-5 polynomial to the spline values
    try:
        coeff = np.polynomial.polynomial.polyfit(x_eval, y_eval, 5)
    except Exception:
        # Fallback to lower degree if fit fails
        try:
            coeff = np.zeros(6)
            coeff[:4] = np.polynomial.polynomial.polyfit(x_eval, y_eval, 3)
        except Exception:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
    
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
    
    return coeff
# EVOLVE-BLOCK-END

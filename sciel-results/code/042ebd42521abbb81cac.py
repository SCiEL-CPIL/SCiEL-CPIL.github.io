"""Strategy: Fit a piecewise cubic spline to capture local structure, then project to a degree-5 polynomial."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import UnivariateSpline

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    if len(x) < 6:
        return np.zeros(6)
        
    # Ensure unique x values for spline fitting, or handle duplicates by averaging
    # UnivariateSpline requires strictly increasing x
    unique_x, indices = np.unique(x, return_index=True)
    # If there are duplicates, we might lose some data. 
    # A better approach for duplicates: bin them.
    if len(unique_x) < len(x):
        # Simple binning: sum y for each x, divide by count
        bin_y = np.zeros_like(unique_x)
        counts = np.zeros_like(unique_x)
        for xi, yi in zip(x, y):
            idx = np.searchsorted(unique_x, xi)
            if idx < len(unique_x) and unique_x[idx] == xi:
                bin_y[idx] += yi
                counts[idx] += 1
        # Handle any zeros in counts (shouldn't happen if unique_x came from x)
        counts[counts == 0] = 1
        y_bin = bin_y / counts
        x_fit = unique_x
        y_fit = y_bin
    else:
        x_fit = x
        y_fit = y

    # Sort x for spline
    sort_idx = np.argsort(x_fit)
    x_sorted = x_fit[sort_idx]
    y_sorted = y_fit[sort_idx]
    
    # We need at least 4 points for cubic spline
    if len(x_sorted) < 4:
        # Fallback to linear if too few points
        try:
            coeff = np.polynomial.polynomial.polyfit(x_sorted, y_sorted, 1)
            out = np.zeros(6)
            out[:len(coeff)] = coeff
            return out
        except Exception:
            return np.zeros(6)

    # Fit a UnivariateSpline
    # s parameter controls smoothing. We want it to be reasonably smooth but capture structure.
    # Default s=0 is interpolation. We can try a few values or use a heuristic.
    # Given the "unresolved tendency" might be noisy, some smoothing is good.
    # Let's try s = len(x) * (np.var(y) / 100) or similar? 
    # Actually, let's just use a moderate smoothing factor or let it interpolate if data is clean.
    # For safety, let's try to fit with s=0 first, but if it fails or is too wiggly, we might need to adjust.
    # However, the output MUST be a degree-5 polynomial.
    # So the strategy is:
    # 1. Fit spline.
    # 2. Evaluate spline on a fine grid.
    # 3. Fit a degree-5 polynomial to these (x, spline(y)) points.
    
    try:
        # Use a smoothing factor. If the data is from a model, it might be relatively smooth.
        # Let's try s = 0.1 * len(x) or something similar. 
        # Or better, let's use the "s" parameter to control the residual sum of squares.
        # For now, let's use a small s to stay close to the data but avoid overfitting noise.
        # A common heuristic is s = N * (noise_variance). We don't know noise variance.
        # Let's try s = 1.0 or s = len(x)/10.
        # Let's stick to a robust approach: fit with s=0 (interpolation) if possible, 
        # but spline interpolation can be very wiggly. 
        # Let's try a few smoothing levels and pick the one that gives the best "polynomial-ness" or just use a fixed moderate one.
        # Given the time constraint, let's pick a fixed moderate smoothing.
        s = 10.0
        
        spl = UnivariateSpline(x_sorted, y_sorted, s=s)
        
        # Create a fine grid to evaluate the spline
        # Range of x
        x_min, x_max = x_sorted[0], x_sorted[-1]
        if x_min == x_max:
            return np.zeros(6)
            
        # Number of points for projection. 100-200 is usually enough for a 5th degree poly.
        n_grid = 200
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = spl(x_grid)
        
        # Fit a degree-5 polynomial to the spline values
        # Use polyfit
        coeff = np.polynomial.polynomial.polyfit(x_grid, y_grid, 5)
        
        # Check if coefficients are finite
        if not np.all(np.isfinite(coeff)):
            # Fallback to simpler fit
            coeff_simple = np.polynomial.polynomial.polyfit(x_sorted, y_sorted, 3)
            out = np.zeros(6)
            out[:len(coeff_simple)] = coeff_simple
            return out
            
        return coeff
        
    except Exception:
        # Fallback to global polynomial fit if spline fails
        try:
            coeff = np.polynomial.polynomial.polyfit(x_sorted, y_sorted, 5)
            if not np.all(np.isfinite(coeff)):
                 coeff = np.polynomial.polynomial.polyfit(x_sorted, y_sorted, 3)
            out = np.zeros(6)
            out[:len(coeff)] = coeff
            return out
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END

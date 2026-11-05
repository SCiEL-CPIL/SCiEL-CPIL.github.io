"""
Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
Combines spline-informed feature extraction with cross-validated polynomial regression.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Strategy:
    1. Normalize x for numerical stability.
    2. Fit a piecewise cubic spline (natural BC) to capture local oscillatory structure.
    3. Use the spline fit as a 'surrogate' target or feature to guide polynomial selection?
       No, the task requires returning polynomial coefficients.
       Instead, we use the spline to understand the structure, but since we must return a global polynomial,
       we focus on optimizing the polynomial fit using insights from the spline (e.g., better regularization).
       
       Actually, the instruction says: "Replace the global monomial polynomial with a piecewise cubic spline... 
       This representation avoids... allowing...". But the output MUST be 6 polynomial coefficients.
       
       Re-reading the prompt: "Fit a shared degree-five polynomial approximation...".
       The inspiration says: "spline-to-polynomial projection retains smoothness while fitting the local features".
       
       So the strategy is:
       1. Fit a flexible spline to the data (x, y).
       2. Project this spline function onto the degree-5 polynomial basis to get the coefficients.
       3. Alternatively, use the spline to generate dense, smooth data points and fit a polynomial to those?
          No, that's just smoothing.
       
       Let's look at the "strongest ideas":
       Inspiration 1: Ridge regression with cross-validation for degree/lambda selection.
       Inspiration 2: Spline-informed polynomial fit.
       
       Combined Strategy:
       1. Normalize x.
       2. Fit a cubic spline to (x_norm, y) with quantile knots.
       3. Evaluate the spline on a dense grid of x_norm values.
       4. Fit a degree-5 polynomial to these (grid_x, grid_y) points using least squares.
          This effectively projects the spline onto the polynomial space, weighted by the grid.
       5. Use cross-validation on the original data to select the number of spline knots (smoothness)
          and potentially the polynomial degree (though we are stuck with 5 output slots, we can fit lower).
       
       Wait, if we fit a polynomial to the spline's output, we are just finding the best L2 approximation 
       of the spline by a degree-5 polynomial. This is equivalent to fitting the polynomial to the original data
       if the spline is an interpolant, but the spline adds a bias (smoothing) that might help generalization.
       
       Let's try this:
       1. Normalize x.
       2. For a set of candidate smoothness parameters (knot counts) for the spline:
          a. Fit spline to (x_norm, y).
          b. Evaluate spline on a fine grid covering the range of x_norm.
          c. Fit degree-5 polynomial to (grid_x, grid_y).
          d. Calculate CV error of the resulting polynomial on the original data.
       3. Select the spline smoothness that yields the best CV error for the projected polynomial.
       4. Return the coefficients of the best polynomial.
    """
    start_time = time.time()
    DEADLINE = 105.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 6:
        if n >= 2:
            try:
                c_lin = np.polyfit(x, y, 1)
                coeff = np.zeros(6)
                coeff[0] = c_lin[1]
                coeff[1] = c_lin[0]
                return coeff
            except:
                pass
        return np.zeros(6)

    # 1. Normalize x
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # 2. Helper to build Vandermonde
    def vandermonde(xn, degree=5):
        V = np.ones((len(xn), degree + 1))
        for i in range(1, degree + 1):
            V[:, i] = V[:, i-1] * xn
        return V

    # 3. Helper to project a function (defined by spline) onto polynomial basis
    def project_spline_to_poly(spline, x_grid, degree=5):
        y_grid = spline(x_grid)
        V_grid = vandermonde(x_grid, degree)
        # Least squares fit
        coeffs, _, _, _ = np.linalg.lstsq(V_grid, y_grid, rcond=None)
        return coeffs

    # 4. Cross-validation to select spline smoothness (knot count)
    # We will test different numbers of knots for the spline.
    # The more knots, the closer the spline is to the data (interpolating).
    # Fewer knots = smoother = more bias, less variance.
    # We project the spline to a poly and check CV error of that poly.
    
    rng = np.random.RandomState(42)
    n_folds = 5
    fold_size = n // n_folds
    indices = rng.permutation(n)
    
    # Candidate knot counts (interior knots)
    # If n is small, fewer knots.
    max_knots = min(n - 4, 50)
    knot_counts = [2, 5, 10, 20, 40]
    knot_counts = [k for k in knot_counts if k <= max_knots]
    if not knot_counts:
        knot_counts = [2]
    
    best_cv_mse = np.inf
    best_knot_count = knot_counts[0]
    best_coeff_norm = None
    
    # Pre-compute grid for projection
    # The grid should cover the range of x_norm with sufficient density
    x_min, x_max = np.min(x_norm), np.max(x_norm)
    grid_size = 200
    x_grid = np.linspace(x_min, x_max, grid_size)
    
    for n_knots in knot_counts:
        if time.time() - start_time > DEADLINE - 15:
            break
            
        cv_mses = []
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            x_train = x_norm[train_idx]
            y_train = y[train_idx]
            x_val = x_norm[val_idx]
            y_val = y[val_idx]
            
            try:
                # Fit spline to training data
                # s=0 means interpolating, s>0 means smoothing
                # We want to control the "effective" smoothness via knots.
                # Using UnivariateSpline with k=3 (cubic).
                # To use quantile knots, we might need to use a different approach or 
                # just let scipy choose knots? UnivariateSpline uses fixed number of knots 
                # internally or based on s.
                # Actually, scipy.interpolate.UnivariateSpline(x, y, k=3, s=...) 
                # The number of knots is determined by s.
                # Alternatively, we can use make_interp_spline with specific knots.
                
                # Let's use make_interp_spline for explicit knot control.
                from scipy.interpolate import make_interp_spline
                
                # Place knots at quantiles of x_train
                if n_knots == 0:
                    # No interior knots, just a cubic
                    # make_interp_spline requires at least 4 points
                    if len(x_train) >= 4:
                        # Sort data
                        sort_idx = np.argsort(x_train)
                        x_tr_sorted = x_train[sort_idx]
                        y_tr_sorted = y_train[sort_idx]
                        spl = make_interp_spline(x_tr_sorted, y_tr_sorted, k=3)
                    else:
                        # Fallback to polyfit
                        c = np.polyfit(x_train, y_train, 3)
                        # Convert to ascending
                        c_asc = c[::-1]
                        coeff_norm = np.zeros(6)
                        coeff_norm[:4] = c_asc
                        y_pred = np.polyval(c, x_val)
                        mse = np.mean((y_pred - y_val) ** 2)
                        if np.isfinite(mse):
                            cv_mses.append(mse)
                        continue
                else:
                    quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
                    knots = np.quantile(x_train, quantiles)
                    # make_interp_spline needs sorted data
                    sort_idx = np.argsort(x_train)
                    x_tr_sorted = x_train[sort_idx]
                    y_tr_sorted = y_train[sort_idx]
                    # Ensure knots are within range and unique
                    knots = np.unique(knots)
                    if len(knots) == 0:
                        continue
                    spl = make_interp_spline(x_tr_sorted, y_tr_sorted, k=3, t=knots)
                
                # Project spline to polynomial using the global grid
                coeff_norm_candidate = project_spline_to_poly(spl, x_grid, degree=5)
                
                if not np.all(np.isfinite(coeff_norm_candidate)):
                    continue
                
                # Evaluate polynomial on validation set
                V_val = vandermonde(x_val, 5)
                y_pred = V_val @ coeff_norm_candidate
                mse = np.mean((y_pred - y_val) ** 2)
                
                if np.isfinite(mse):
                    cv_mses.append(mse)
                    
            except Exception:
                continue
        
        if cv_mses:
            mean_cv_mse = np.mean(cv_mses)
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_knot_count = n_knots
                
                # Store the coefficients from the best fold? No, we need to refit on all data.
                # We just record the best knot count.

    # 5. Final fit with best knot count on all data
    try:
        from scipy.interpolate import make_interp_spline
        
        sort_idx = np.argsort(x_norm)
        x_sorted = x_norm[sort_idx]
        y_sorted = y[sort_idx]
        
        if best_knot_count == 0:
            if n >= 4:
                spl = make_interp_spline(x_sorted, y_sorted, k=3)
            else:
                # Fallback
                c = np.polyfit(x_norm, y, min(5, n-1))
                coeffs_norm = np.zeros(6)
                coeffs_norm[:len(c)] = c[::-1]
        else:
            quantiles = np.linspace(0, 1, best_knot_count + 2)[1:-1]
            knots = np.quantile(x_sorted, quantiles)
            knots = np.unique(knots)
            if len(knots) == 0:
                spl = make_interp_spline(x_sorted, y_sorted, k=3)
            else:
                spl = make_interp_spline(x_sorted, y_sorted, k=3, t=knots)
        
        # Project to polynomial
        coeffs_norm = project_spline_to_poly(spl, x_grid, degree=5)
        
    except Exception:
        # Fallback to standard Ridge/OLS
        V = vandermonde(x_norm, 5)
        try:
            VtV = V.T @ V
            Vty = V.T @ y
            # Small ridge for stability
            lam = 1e-8
            A = VtV + lam * np.eye(6)
            coeffs_norm = np.linalg.solve(A, Vty)
        except:
            coeffs_norm, _, _, _ = np.linalg.lstsq(V, y, rcond=None)

    if not np.all(np.isfinite(coeffs_norm)):
        return np.zeros(6)

    # 6. Transform coefficients from normalized space to original space
    coeff_orig = np.zeros(6)
    for i in range(6):
        a_i = coeffs_norm[i] / (x_std ** i)
        for k in range(i + 1):
            coeff_orig[k] += a_i * comb(i, k) * ((-x_mean) ** (i - k))
            
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
        
    return coeff_orig

# EVOLVE-BLOCK-END

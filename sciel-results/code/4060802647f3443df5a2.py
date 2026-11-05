import numpy as np
import time
from scipy.interpolate import CubicSpline

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Uses a spline-informed approach: fit a cubic spline to the data, then find the
    best L2 polynomial approximation of that spline. This captures non-polynomial
    structure better than a direct global polynomial fit.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Flatten just in case
    x = x.flatten()
    y = y.flatten()
    
    N = len(x)
    if N == 0:
        return np.zeros(6)
    
    # 1. Sort data for spline fitting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # 2. Create quantile-based knots for the spline
    # Use a moderate number of knots to capture structure without overfitting
    n_knots = min(20, max(10, N // 100))  # Adaptive number of knots
    knot_quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]  # Interior knots
    knot_positions = np.quantile(x_sorted, knot_quantiles)
    
    # 3. Fit cubic spline with natural boundary conditions
    try:
        cs = CubicSpline(x_sorted, y_sorted, bc_type='natural')
    except Exception:
        # Fallback to simple polynomial fit if spline fails
        return _fallback_poly_fit(x, y, start_time, DEADLINE)
    
    # 4. Find best L2 polynomial approximation of degree 5 to the spline
    # We want to minimize ||P(x) - cs(x)||^2 over the domain
    # This is equivalent to finding coefficients c that minimize:
    # sum_i (P(x_i) - cs(x_i))^2 for a dense set of points
    
    # Sample the spline at many points for accurate integration
    x_eval = np.linspace(x_sorted[0], x_sorted[-1], max(1000, N // 10))
    y_spline = cs(x_eval)
    
    # Build Vandermonde matrix for polynomial fit
    V = np.vander(x_eval, N=6, increasing=True)
    
    # Solve least squares: find c such that V @ c ≈ y_spline
    try:
        c, _, _, _ = np.linalg.lstsq(V, y_spline, rcond=None)
    except Exception:
        return _fallback_poly_fit(x, y, start_time, DEADLINE)
    
    # 5. Ensure finite coefficients
    if not np.all(np.isfinite(c)):
        return _fallback_poly_fit(x, y, start_time, DEADLINE)
    
    return c

def _fallback_poly_fit(x, y, start_time, DEADLINE):
    """Fallback to direct polynomial fit with regularization."""
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    N = len(x)
    
    if N == 0:
        return np.zeros(6)
    
    mu_x = np.mean(x)
    z = x - mu_x
    
    V = np.vander(z, N=6, increasing=True)
    VtV = V.T @ V
    Vty = V.T @ y
    
    diag_mean = np.mean(np.diag(VtV))
    if diag_mean < 1e-12:
        return np.array([np.mean(y), 0, 0, 0, 0, 0])
    
    lam = 1e-8 * diag_mean
    A = VtV + lam * np.eye(6)
    
    try:
        a = np.linalg.solve(A, Vty)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(V, y, rcond=None)[0]
    
    # Transform from centered to original coordinates
    c = np.zeros(6)
    neg_mu_powers = np.array([(-mu_x) ** i for i in range(6)])
    binoms = [
        [1],
        [1, 1],
        [1, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 6, 4, 1],
        [1, 5, 10, 10, 5, 1]
    ]
    
    for k in range(6):
        a_k = a[k]
        if a_k == 0:
            continue
        for j in range(k + 1):
            c[j] += a_k * binoms[k][j] * neg_mu_powers[k - j]
    
    if not np.all(np.isfinite(c)):
        V_orig = np.vander(x, N=6, increasing=True)
        try:
            c, _, _, _ = np.linalg.lstsq(V_orig, y, rcond=None)
        except:
            c = np.zeros(6)
            c[0] = np.mean(y)
    
    return c

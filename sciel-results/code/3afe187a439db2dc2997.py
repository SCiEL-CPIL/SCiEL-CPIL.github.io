"""Rewritten program for unresolved-tendency closure task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_to_monomial(cheb_coeffs, x_shift, x_scale):
    """
    Convert Chebyshev series coefficients to monomial coefficients.
    
    If P(t) = sum_{k=0}^{n} a_k * T_k(t) where t = (x - x_shift)/x_scale,
    then we want Q(x) = sum_{j=0}^{n} c_j * x^j such that Q(x) = P((x-x_shift)/x_scale).
    
    Uses exact recurrence to avoid catastrophic cancellation.
    T_0(t) = 1
    T_1(t) = t
    T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    We build each T_k as a monomial in x, then accumulate.
    """
    n = len(cheb_coeffs)
    # monomials[k] will be the coefficients of T_k expressed in powers of x
    # monomials[k][j] is the coefficient of x^j in T_k(x_shift + x_scale * t)
    
    # T_0(t) = 1 => as function of x: 1
    monomials = [np.zeros(n) for _ in range(n)]
    monomials[0][0] = 1.0
    
    if n > 1:
        # T_1(t) = t = (x - x_shift)/x_scale = (1/x_scale)*x - (x_shift/x_scale)
        monomials[1][0] = -x_shift / x_scale
        monomials[1][1] = 1.0 / x_scale
    
    for k in range(2, n):
        # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        # t * T_{k-1}(t) where T_{k-1} is expressed in x:
        # t = (x - x_shift)/x_scale
        # t * T_{k-1}(x) = (1/x_scale)*(x - x_shift) * T_{k-1}(x)
        # = (1/x_scale) * [x * T_{k-1}(x) - x_shift * T_{k-1}(x)]
        
        # x * T_{k-1}(x): shift coefficients up by one power
        x_times_prev = np.zeros(n)
        x_times_prev[1:] = monomials[k-1][:-1]
        
        # x_shift * T_{k-1}(x)
        xshift_times_prev = x_shift * monomials[k-1]
        
        # (1/x_scale) * (x * T_{k-1} - x_shift * T_{k-1})
        t_times_prev = (x_times_prev - xshift_times_prev) / x_scale
        
        # T_k = 2 * t * T_{k-1} - T_{k-2}
        monomials[k] = 2.0 * t_times_prev - monomials[k-2]
    
    # Now accumulate: Q(x) = sum_k a_k * T_k_as_monomial(x)
    result = np.zeros(n)
    for k in range(n):
        result += cheb_coeffs[k] * monomials[k]
    
    return result


def _eval_chebyshev_series(cheb_coeffs, t_values):
    """Evaluate Chebyshev series sum a_k * T_k(t) using Clenshaw's algorithm."""
    n = len(cheb_coeffs)
    if n == 0:
        return np.zeros_like(t_values)
    if n == 1:
        return np.full_like(t_values, cheb_coeffs[0])
    
    # Clenshaw's algorithm for Chebyshev series
    # b_{n+1} = b_{n+2} = 0
    # b_k = a_k + 2*t*b_{k+1} - b_{k+2}  for k = n-1, ..., 1
    # result = a_0 + t*b_1 - b_2
    
    b_kp2 = np.zeros_like(t_values)
    b_kp1 = np.zeros_like(t_values)
    
    for k in range(n - 1, 0, -1):
        b_kp2, b_kp1 = b_kp1, cheb_coeffs[k] + 2.0 * t_values * b_kp1 - b_kp2
    
    return cheb_coeffs[0] + t_values * b_kp1 - b_kp2


def _build_chebyshev_matrix(t_values):
    """Build Chebyshev basis matrix where row i is [T_0(t_i), T_1(t_i), ..., T_5(t_i)]."""
    n_pts = len(t_values)
    n_chem = 6
    C = np.zeros((n_pts, n_chem))
    C[:, 0] = 1.0
    if n_chem > 1:
        C[:, 1] = t_values
    for k in range(2, n_chem):
        C[:, k] = 2.0 * t_values * C[:, k-1] - C[:, k-2]
    return C


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses Chebyshev basis for numerical stability, then converts to monomial coefficients.
    Regularized least-squares with cross-validation for lambda selection.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        # Not enough data for degree-5 fit; fall back to zero
        return np.zeros(6)

    # Map x to [-1, 1] using the training data range
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    x_shift = (x_max + x_min) / 2.0
    x_scale = x_range / 2.0
    
    # t = (x - x_shift) / x_scale, so t in [-1, 1]
    t = (x - x_shift) / x_scale

    # Build Chebyshev basis matrix
    C = _build_chebyshev_matrix(t)

    # If we have enough data, use a held-out subset for model selection
    use_cv = n > 100 and (time.time() - start_time) < (DEADLINE - 30.0)
    
    if use_cv:
        # Split into train and validation subsets
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        val_size = max(10, n // 10)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        
        t_train = t[train_idx]
        y_train = y[train_idx]
        t_val = t[val_idx]
        y_val = y[val_idx]
        x_train_raw = x[train_idx]
        x_val_raw = x[val_idx]
        
        C_train = _build_chebyshev_matrix(t_train)
        C_val = _build_chebyshev_matrix(t_val)
    else:
        t_train = t
        y_train = y
        t_val = t
        y_val = y
        x_train_raw = x
        x_val_raw = x
        C_train = C
        C_val = C

    # Try several regularization strengths
    lambdas = np.logspace(-12, 2, 15)
    
    best_coeff = None
    best_val_mse = np.inf
    best_train_coeff = None
    best_train_mse = np.inf

    for lam in lambdas:
        if time.time() > DEADLINE:
            break

        try:
            # Regularized least squares in Chebyshev basis
            # Solve (C^T C + lambda*I) a = C^T y
            A = C_train.T @ C_train + lam * np.eye(6)
            b = C_train.T @ y_train
            cheb_coeffs = np.linalg.solve(A, b)

            # Convert Chebyshev coefficients to monomial coefficients
            # P(t) = sum a_k * T_k(t), t = (x - x_shift)/x_scale
            # We want Q(x) = sum c_j * x^j
            monomial_coeff = _chebyshev_to_monomial(cheb_coeffs, x_shift, x_scale)

            # Check if coefficients are finite
            if not np.all(np.isfinite(monomial_coeff)):
                continue

            # Compute validation MSE using the monomial coefficients on raw x
            pred_val = np.zeros_like(y_val)
            for i in range(6):
                pred_val += monomial_coeff[i] * (x_val_raw ** i)
            val_mse = np.mean((pred_val - y_val) ** 2)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_coeff = monomial_coeff.copy()

            # Also track best training fit
            pred_train = np.zeros_like(y_train)
            for i in range(6):
                pred_train += monomial_coeff[i] * (x_train_raw ** i)
            train_mse = np.mean((pred_train - y_train) ** 2)
            if train_mse < best_train_mse:
                best_train_mse = train_mse
                best_train_coeff = monomial_coeff.copy()

        except np.linalg.LinAlgError:
            continue

    if best_coeff is not None:
        return best_coeff

    # Fallback: use the best training fit
    if best_train_coeff is not None:
        return best_train_coeff

    # Final fallback: simple linear fit
    try:
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    except Exception:
        return np.zeros(6)
# EVOLVE-BLOCK-END

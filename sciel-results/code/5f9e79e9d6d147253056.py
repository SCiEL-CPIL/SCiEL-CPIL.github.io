"""Tikhonov-regularized degree-five polynomial closure with CV-selected lambda."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _legendre_poly(x, n):
    """Compute Legendre polynomial P_n(x) using three-term recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    p_nm1 = np.ones_like(x)
    p_n = x.copy()
    for k in range(2, n + 1):
        p_np1 = ((2 * k - 1) * x * p_n - (k - 1) * p_nm1) / k
        p_nm1 = p_n
        p_n = p_np1
    return p_n


def _fit_regularized(V, y, lam):
    """Solve (V^T V + lam*I) c = V^T y for polynomial coefficients."""
    n = V.shape[0]
    A = V.T @ V + lam * np.eye(V.shape[1])
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c


def _predict(V, c):
    return V @ c


def construct(training_x, training_subgrid_tendency):
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(training_x)
    if n < 10:
        # Fallback: just fit a linear polynomial
        if n < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(training_x, training_subgrid_tendency, 1)[::-1]
        return coeff

    # Determine the domain of x for scaling to [-1, 1]
    x_min = np.min(training_x)
    x_max = np.max(training_x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # x is constant, just return the constant term
        coeff = np.zeros(6)
        coeff[0] = np.mean(training_subgrid_tendency)
        return coeff

    # Scale x to [-1, 1] for Legendre polynomials
    # x_scaled = 2*(x - x_min)/(x_max - x_min) - 1
    x_scaled = 2.0 * (training_x - x_min) / x_range - 1.0

    # Build Legendre basis matrix: columns P_0(x), P_1(x), ..., P_5(x)
    L = np.zeros((n, 6))
    for k in range(6):
        L[:, k] = _legendre_poly(x_scaled, k)

    # K-fold cross-validation to select lambda
    n_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_sizes = np.array_split(indices, n_folds)

    # Candidate lambdas: log-spaced
    y_scale = np.std(training_subgrid_tendency)
    if y_scale < 1e-12:
        y_scale = 1.0
    lam_base = y_scale ** 2
    if lam_base < 1e-12:
        lam_base = 1.0

    lambda_candidates = np.concatenate([
        [0.0],
        np.logspace(-8, 6, 40) * lam_base
    ])

    best_lam = 0.0
    best_cv_score = np.inf

    for lam in lambda_candidates:
        if _time_remaining() < 2.0:
            break
        mse_sum = 0.0
        for fold in range(n_folds):
            val_idx = fold_sizes[fold]
            train_idx = np.concatenate([fold_sizes[i] for i in range(n_folds) if i != fold])

            L_train = L[train_idx]
            y_train = training_subgrid_tendency[train_idx]
            L_val = L[val_idx]
            y_val = training_subgrid_tendency[val_idx]

            c = _fit_regularized(L_train, y_train, lam)
            pred = _predict(L_val, c)
            mse_sum += np.mean((pred - y_val) ** 2)

        cv_score = mse_sum / n_folds
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lam = lam

    # Final fit on all training data with best lambda
    coeff_legendre = _fit_regularized(L, training_subgrid_tendency, best_lam)

    # Transform from Legendre basis to monomial basis
    # We need to express sum_{i=0}^{5} c_i * P_i(x_scaled) as sum_{j=0}^{5} d_j * x^j
    # where x_scaled = 2*(x - x_min)/x_range - 1
    
    # First, express each P_i(x_scaled) in terms of x
    # x_scaled = (2/x_range) * x + (1 - 2*x_min/x_range)
    # Let a = 2/x_range, b = 1 - 2*x_min/x_range
    # Then x_scaled = a*x + b
    
    # P_i(a*x + b) is a polynomial in x of degree i
    # We can compute this by evaluating P_i at multiple points and using interpolation,
    # or by direct expansion. For degree 5, direct expansion is feasible.
    
    a = 2.0 / x_range
    b = 1.0 - 2.0 * x_min / x_range
    
    # Compute P_i(a*x + b) for i=0..5 as polynomials in x
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = (3t^2 - 1)/2
    # P_3(t) = (5t^3 - 3t)/2
    # P_4(t) = (35t^4 - 30t^2 + 3)/8
    # P_5(t) = (63t^5 - 70t^3 + 15t)/8
    
    # For each P_i, substitute t = a*x + b and expand
    # We'll use a numerical approach: evaluate at 6 points and interpolate
    
    # Actually, let's do it analytically for clarity
    # P_0(a*x+b) = 1
    # P_1(a*x+b) = a*x + b
    # P_2(a*x+b) = (3*(a*x+b)^2 - 1)/2 = (3*a^2*x^2 + 6*a*b*x + 3*b^2 - 1)/2
    # P_3(a*x+b) = (5*(a*x+b)^3 - 3*(a*x+b))/2
    # P_4(a*x+b) = (35*(a*x+b)^4 - 30*(a*x+b)^2 + 3)/8
    # P_5(a*x+b) = (63*(a*x+b)^5 - 70*(a*x+b)^3 + 15*(a*x+b))/8
    
    # Let's compute coefficients by evaluating at sufficient points
    # Use 6 distinct points in x to determine the monomial coefficients
    
    # Alternative: use the fact that we can compute the polynomial coefficients
    # by working with the expanded form
    
    # Let me use a cleaner approach: build a transformation matrix
    # M[i, j] = coefficient of x^j in P_i(a*x + b)
    
    # I'll compute this by evaluating P_i at 6 points and using Vandermonde inversion
    x_eval = np.linspace(x_min, x_max, 6)
    x_scaled_eval = 2.0 * (x_eval - x_min) / x_range - 1.0
    
    # Vandermonde matrix for monomial basis
    V_mono = np.vander(x_eval, 6, increasing=True)
    
    # For each Legendre polynomial, evaluate at x_eval
    M = np.zeros((6, 6))
    for i in range(6):
        vals = _legendre_poly(x_scaled_eval, i)
        # Solve V_mono @ M[i, :] = vals
        M[i, :] = np.linalg.solve(V_mono, vals)
    
    # Now coeff_monomial[j] = sum_i coeff_legendre[i] * M[i, j]
    coeff = M.T @ coeff_legendre
    
    # Sanity check: ensure finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(training_x, training_subgrid_tendency, 1)[::-1]
        if not np.all(np.isfinite(coeff)):
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.polynomial import chebyshev as C


def construct(training_x, training_subgrid_tendency):
    """
    Fit a degree-5 polynomial using a Chebyshev basis with optimized Tikhonov
    regularization. Returns 6 coefficients in ascending powers of x.
    
    Strategy:
    1. Center and scale x using median/std for robust normalization.
    2. Build Chebyshev design matrix for degree 5.
    3. Optimize regularization lambda via internal k-fold cross-validation.
    4. Solve regularized least squares in Chebyshev basis.
    5. Convert back to power basis using exact recurrence relations.
    """
    t_start = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    n = len(x)
    
    if n < 10:
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    # Determine scaling: center at median, scale by std for robustness
    x_center = np.median(x)
    x_scale = np.std(x)
    if x_scale < 1e-10:
        x_scale = 1.0
    
    # Normalize to roughly [-1, 1] for Chebyshev
    x_norm = (x - x_center) / x_scale
    
    # Clip extreme values to avoid numerical issues
    x_norm = np.clip(x_norm, -3.0, 3.0)
    
    # Build Chebyshev design matrix for degree 5
    degree = 5
    try:
        V = C.chebvander(x_norm, degree)
    except Exception:
        V = np.vander(x_norm, N=degree+1, increasing=True)
    
    def solve_regularized(V, y, lam_diag):
        """Solve (V^T V + lam_diag) c = V^T y"""
        VtV = V.T @ V
        Vty = V.T @ y
        A = VtV + lam_diag
        try:
            c = np.linalg.solve(A, Vty)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A, Vty, rcond=None)[0]
        return c
    
    def compute_mse(c, x_norm_test, y_test):
        """Compute MSE for given coefficients on test data"""
        try:
            V_test = C.chebvander(x_norm_test, degree)
        except Exception:
            V_test = np.vander(x_norm_test, N=degree+1, increasing=True)
        pred = V_test @ c
        return np.mean((pred - y_test) ** 2)
    
    # Try a grid of regularization parameters
    lambda_candidates = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    # Internal cross-validation to pick lambda
    n_folds = 5
    n_samples = len(x_norm)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n_samples // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    
    best_lambda = 0.0
    best_cv_score = np.inf
    
    for lam in lambda_candidates:
        cv_scores = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            x_test = x_norm[val_idx]
            y_test = y[val_idx]
            
            # Use decaying regularization: higher degrees penalized more
            lam_diag = np.diag([lam * (k+1)**2 for k in range(degree+1)])
            
            c = solve_regularized(V_train, y_train, lam_diag)
            mse = compute_mse(c, x_test, y_test)
            cv_scores.append(mse)
        
        mean_cv = np.mean(cv_scores)
        if mean_cv < best_cv_score:
            best_cv_score = mean_cv
            best_lambda = lam
    
    # Final fit with best lambda
    lam_diag = np.diag([best_lambda * (k+1)**2 for k in range(degree+1)])
    c_cheb = solve_regularized(V, y, lam_diag)
    
    # Convert Chebyshev coefficients to monomial coefficients in z
    # Use exact recurrence to avoid catastrophic cancellation
    try:
        a_z = C.cheb2poly(c_cheb)
    except Exception:
        # Fallback: manual conversion using recurrence
        a_z = np.zeros(degree+1)
        for k in range(degree+1):
            # T_k(x) expressed in power basis
            if k == 0:
                a_z[0] += c_cheb[0]
            elif k == 1:
                a_z[1] += c_cheb[1]
            else:
                # T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
                # We need to accumulate the power basis coefficients
                # This is complex, so we rely on cheb2poly
                a_z = C.cheb2poly(c_cheb)
                break
    
    if len(a_z) < 6:
        a_z = np.pad(a_z, (0, 6 - len(a_z)))
    elif len(a_z) > 6:
        a_z = a_z[:6]
    
    # Transform coefficients from z to x
    # We have P(z) = sum_{i=0}^5 a_i * z^i
    # z = (x - c) / s
    # z^i = (1/s^i) * (x - c)^i
    # (x - c)^i = sum_{j=0}^i binom(i, j) * x^j * (-c)^(i-j)
    # b_j = sum_{i=j}^5 a_i * (1/s^i) * binom(i, j) * (-c)^(i-j)
    
    b = np.zeros(6)
    for i in range(6):
        if a_z[i] == 0:
            continue
        coeff_scale = a_z[i] / (x_scale ** i)
        for j in range(i + 1):
            # binom(i, j)
            binom_val = 1.0
            if j > i - j:
                j_temp = i - j
            else:
                j_temp = j
            for k in range(1, j_temp + 1):
                binom_val *= (i - j_temp + k) / k
            
            term = coeff_scale * binom_val * ((-x_center) ** (i - j))
            b[j] += term
    
    # Ensure finite
    if not np.all(np.isfinite(b)):
        b = np.zeros(6)
        try:
            b[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
    
    return b
# EVOLVE-BLOCK-END

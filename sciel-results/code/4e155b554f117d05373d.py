import numpy as np
import time

def _chebyshev_basis(x, degree):
    """Compute Chebyshev polynomial basis values for input x."""
    x_norm = 2.0 * x - 1.0  # Map to [-1, 1]
    basis = np.zeros((len(x), degree + 1))
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = x_norm
    for k in range(2, degree + 1):
        basis[:, k] = 2.0 * x_norm * basis[:, k-1] - basis[:, k-2]
    return basis

def _cheb_to_monomial(cheb_coeffs, degree):
    """Convert Chebyshev coefficients to monomial coefficients."""
    # Use the identity: T_n(x) = sum of binomial terms
    # For numerical stability, use numpy's polynomial conversion
    # T_n(x/2 + 1/2) expressed in powers of x
    monomial = np.zeros(degree + 1)
    for n in range(degree + 1):
        if abs(cheb_coeffs[n]) < 1e-15:
            continue
        # Convert T_n(2x-1) to monomial in x
        # T_n(z) where z = 2x-1
        # We need coefficients such that T_n(2x-1) = sum_{k=0}^n c_k x^k
        # Use recurrence or direct formula
        # For small degrees, use the explicit formula
        if n == 0:
            monomial[0] += cheb_coeffs[n]
        elif n == 1:
            monomial[0] += cheb_coeffs[n] * (-1)
            monomial[1] += cheb_coeffs[n] * 2
        elif n == 2:
            monomial[0] += cheb_coeffs[n] * 1
            monomial[1] += cheb_coeffs[n] * (-4)
            monomial[2] += cheb_coeffs[n] * 4
        elif n == 3:
            monomial[0] += cheb_coeffs[n] * (-2)
            monomial[1] += cheb_coeffs[n] * 12
            monomial[2] += cheb_coeffs[n] * (-24)
            monomial[3] += cheb_coeffs[n] * 16
        elif n == 4:
            monomial[0] += cheb_coeffs[n] * 3
            monomial[1] += cheb_coeffs[n] * (-24)
            monomial[2] += cheb_coeffs[n] * 72
            monomial[3] += cheb_coeffs[n] * (-96)
            monomial[4] += cheb_coeffs[n] * 16
        elif n == 5:
            monomial[0] += cheb_coeffs[n] * (-4)
            monomial[1] += cheb_coeffs[n] * 40
            monomial[2] += cheb_coeffs[n] * (-160)
            monomial[3] += cheb_coeffs[n] * 320
            monomial[4] += cheb_coeffs[n] * (-320)
            monomial[5] += cheb_coeffs[n] * 32
    return monomial

def _ridge_fit_chebyshev(x, y, degree, alpha):
    """Fit polynomial using Chebyshev basis with ridge regularization."""
    basis = _chebyshev_basis(x, degree)
    # Ridge regression: solve (A^T A + alpha * I) c = A^T y
    n = len(x)
    A = basis
    # Scale regularization by n for consistency
    reg = alpha * n * np.eye(degree + 1)
    # Don't regularize the constant term as heavily
    reg[0, 0] *= 0.1
    try:
        A_t_A = A.T @ A
        A_t_y = A.T @ y
        coeff_cheb = np.linalg.solve(A_t_A + reg, A_t_y)
    except np.linalg.LinAlgError:
        coeff_cheb = np.linalg.lstsq(A_t_A + reg, A_t_y, rcond=None)[0]
    return coeff_cheb

def construct(training_x, training_subgrid_tendency):
    """
    Fits a degree-5 polynomial using Chebyshev-basis Ridge regression
    with cross-validation to select the regularization parameter.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge cases
    if len(x) == 0:
        return np.zeros(6)
    if len(x) < 6:
        try:
            # Use simple linear fit for small data
            coeff = np.polynomial.polynomial.polyfit(x, y, min(1, len(x)-1))
            return np.pad(coeff, (0, 6 - len(coeff)))
        except Exception:
            return np.zeros(6)

    # K-fold Cross-Validation for ridge parameter selection
    n_splits = 5
    indices = np.arange(len(x))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    folds = [indices[i::n_splits] for i in range(n_splits)]
    
    # Test different regularization strengths
    alpha_candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    best_mse = np.inf
    best_alpha = 1e-4
    
    for alpha in alpha_candidates:
        fold_mses = []
        for i in range(n_splits):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
            
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]
            
            try:
                coeff_cheb = _ridge_fit_chebyshev(x_train, y_train, 5, alpha)
                # Evaluate using Chebyshev basis
                basis_val = _chebyshev_basis(x_val, 5)
                y_pred = basis_val @ coeff_cheb
                mse = np.mean((y_val - y_pred) ** 2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_alpha = alpha
            
        # Time check
        if time.time() > deadline - 10.0:
            break

    # Final fit on all data with best alpha
    try:
        coeff_cheb = _ridge_fit_chebyshev(x, y, 5, best_alpha)
        # Convert to monomial coefficients
        coeff = _cheb_to_monomial(coeff_cheb, 5)
    except Exception:
        # Fallback to simple polynomial fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 3)
            coeff = np.pad(coeff, (0, 6 - len(coeff)))
        except Exception:
            coeff = np.zeros(6)
            
    # Ensure we return exactly 6 coefficients
    coeff = np.asarray(coeff, dtype=np.float64)
    if len(coeff) < 6:
        coeff = np.pad(coeff, (0, 6 - len(coeff)))
    elif len(coeff) > 6:
        coeff = coeff[:6]
        
    # Verify finiteness
    if not np.all(np.isfinite(coeff)):
        coeff = np.where(np.isfinite(coeff), coeff, 0.0)

    return coeff

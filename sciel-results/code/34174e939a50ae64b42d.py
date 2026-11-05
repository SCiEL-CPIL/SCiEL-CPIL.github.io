import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency
    using Chebyshev basis for numerical stability, with Tikhonov regularization
    selected via cross-validation.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
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
    
    if n == 0:
        return np.zeros(6)
    if n < 6:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, n-1))
            return np.pad(coeff, (0, 6 - len(coeff)))
        except:
            return np.zeros(6)
    
    # Normalize x to [-1, 1] for Chebyshev basis
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_min, x_max = x_min - 1.0, x_min + 1.0
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    
    # Build Chebyshev basis matrix T(x) where column j = T_j(x)
    # Use recurrence: T_0=1, T_1=x, T_{k}=2x*T_{k-1} - T_{k-2}
    def chebyshev_basis(xn, degree=5):
        npts = len(xn)
        T = np.zeros((npts, degree + 1))
        T[:, 0] = 1.0
        if degree >= 1:
            T[:, 1] = xn.copy()
        for k in range(2, degree + 1):
            T[:, k] = 2.0 * xn * T[:, k-1] - T[:, k-2]
        return T
    
    T = chebyshev_basis(x_norm, 5)
    
    # K-fold cross-validation for regularization parameter
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    
    if n >= 30:
        n_folds = 5
    elif n >= 12:
        n_folds = 3
    else:
        n_folds = 2
    
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        start_idx = f * fold_size
        end_idx = start_idx + fold_size if f < n_folds - 1 else n
        folds.append(indices[start_idx:end_idx])
    
    # Structured regularization: penalize higher Chebyshev coefficients more
    # This acts as a smoothness prior in Chebyshev space
    reg_weights = np.array([1.0, 1.0, 5.0, 25.0, 125.0, 625.0])
    reg_matrix = np.diag(reg_weights)
    
    # Regularization strengths to try
    lambdas = [0.0] + list(np.logspace(-10, 3, 20))
    
    best_mse = np.inf
    best_coeffs_cheb = None
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 10:
            break
        try:
            mse_sum = 0.0
            valid = True
            for f in range(n_folds):
                val_idx = folds[f]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
                
                T_train = T[train_idx]
                y_train = y[train_idx]
                T_val = T[val_idx]
                y_val = y[val_idx]
                
                if lam == 0.0:
                    coeffs, _, _, _ = np.linalg.lstsq(T_train, y_train, rcond=None)
                else:
                    # Tikhonov: (T^T T + lam * W) c = T^T y
                    TtT = T_train.T @ T_train
                    Tty = T_train.T @ y_train
                    A = TtT + lam * reg_matrix
                    try:
                        coeffs = np.linalg.solve(A, Tty)
                    except np.linalg.LinAlgError:
                        coeffs, _, _, _ = np.linalg.lstsq(A, Tty, rcond=None)
                
                if not np.all(np.isfinite(coeffs)):
                    valid = False
                    break
                
                y_pred = T_val @ coeffs
                mse_sum += np.mean((y_pred - y_val) ** 2)
            
            if not valid:
                continue
            
            avg_mse = mse_sum / n_folds
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_coeffs_cheb = coeffs.copy()
        except:
            continue
    
    # If CV failed, fallback to OLS
    if best_coeffs_cheb is None:
        try:
            best_coeffs_cheb, _, _, _ = np.linalg.lstsq(T, y, rcond=None)
            if not np.all(np.isfinite(best_coeffs_cheb)):
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    # Convert Chebyshev coefficients to monomial coefficients
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    # T_5(x) = 16x^5 - 20x^3 + 5x
    #
    # P(x) = sum_{k=0}^5 c_k * T_k(x)
    # We need to express in powers of the ORIGINAL x, not x_norm.
    # x_norm = 2*(x - x_min)/(x_max - x_min) - 1
    # Let a = 2/(x_max - x_min), b = -1 - 2*x_min/(x_max - x_min)
    # x_norm = a*x + b
    #
    # So T_k(x_norm) = T_k(a*x + b)
    # We can compute this by evaluating the Chebyshev polynomial at (a*x + b)
    # and extracting monomial coefficients.
    
    a = 2.0 / (x_max - x_min)
    b = -1.0 - 2.0 * x_min / (x_max - x_min)
    
    # For each Chebyshev coefficient c_k, we need the monomial expansion of c_k * T_k(a*x + b)
    # T_k(t) for t = a*x + b can be expanded using the recurrence in x.
    # Let's build the monomial coefficients of T_k(a*x + b) for k=0..5
    
    # T_0(t) = 1 -> [1, 0, 0, 0, 0, 0]
    # T_1(t) = t = a*x + b -> [b, a, 0, 0, 0, 0]
    # T_2(t) = 2t^2 - 1 = 2(a*x+b)^2 - 1 = 2a^2*x^2 + 4ab*x + (2b^2 - 1)
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    
    # Compute monomial coefficients of T_k(a*x + b) for k=0..5
    # Using the explicit formulas:
    
    # T_0: [1, 0, 0, 0, 0, 0]
    # T_1: [b, a, 0, 0, 0, 0]
    
    # T_2 = 2t^2 - 1
    # t^2 = a^2 x^2 + 2ab x + b^2
    # T_2 = 2a^2 x^2 + 4ab x + (2b^2 - 1)
    
    # T_3 = 4t^3 - 3t
    # t^3 = a^3 x^3 + 3a^2b x^2 + 3ab^2 x + b^3
    # T_3 = 4a^3 x^3 + 12a^2b x^2 + (12ab^2 - 3a) x + (4b^3 - 3b)
    
    # T_4 = 8t^4 - 8t^2 + 1
    # t^4 = a^4 x^4 + 4a^3b x^3 + 6a^2b^2 x^2 + 4ab^3 x + b^4
    # T_4 = 8a^4 x^4 + 32a^3b x^3 + (48a^2b^2 - 8a^2) x^2 + (32ab^3 - 16ab) x + (8b^4 - 8b^2 + 1)
    
    # T_5 = 16t^5 - 20t^3 + 5t
    # t^5 = a^5 x^5 + 5a^4b x^4 + 10a^3b^2 x^3 + 10a^2b^3 x^2 + 5ab^4 x + b^5
    # T_5 = 16a^5 x^5 + 80a^4b x^4 + (160a^3b^2 - 20a^3) x^3 + (160a^2b^3 - 60a^2b) x^2 + (80ab^4 - 30ab) x + (16b^5 - 20b^3 + 5b)
    
    cheb_to_monomial = np.zeros((6, 6))
    
    # T_0
    cheb_to_monomial[0, 0] = 1.0
    
    # T_1
    cheb_to_monomial[1, 0] = b
    cheb_to_monomial[1, 1] = a
    
    # T_2
    cheb_to_monomial[2, 0] = 2*b**2 - 1
    cheb_to_monomial[2, 1] = 4*a*b
    cheb_to_monomial[2, 2] = 2*a**2
    
    # T_3
    cheb_to_monomial[3, 0] = 4*b**3 - 3*b
    cheb_to_monomial[3, 1] = 12*a*b**2 - 3*a
    cheb_to_monomial[3, 2] = 12*a**2*b
    cheb_to_monomial[3, 3] = 4*a**3
    
    # T_4
    cheb_to_monomial[4, 0] = 8*b**4 - 8*b**2 + 1
    cheb_to_monomial[4, 1] = 32*a*b**3 - 16*a*b
    cheb_to_monomial[4, 2] = 48*a**2*b**2 - 8*a**2
    cheb_to_monomial[4, 3] = 32*a**3*b
    cheb_to_monomial[4, 4] = 8*a**4
    
    # T_5
    cheb_to_monomial[5, 0] = 16*b**5 - 20*b**3 + 5*b
    cheb_to_monomial[5, 1] = 80*a*b**4 - 30*a*b
    cheb_to_monomial[5, 2] = 160*a**2*b**3 - 60*a**2*b
    cheb_to_monomial[5, 3] = 160*a**3*b**2 - 20*a**3
    cheb_to_monomial[5, 4] = 80*a**4*b
    cheb_to_monomial[5, 5] = 16*a**5
    
    # Final monomial coefficients: sum_k c_k * cheb_to_monomial[k, :]
    coeff_orig = np.zeros(6)
    for k in range(6):
        if np.isfinite(best_coeffs_cheb[k]):
            coeff_orig += best_coeffs_cheb[k] * cheb_to_monomial[k, :]
    
    # Verify finiteness
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback: direct OLS in monomial basis
        try:
            V = np.vander(x, 6, increasing=True)
            coeff_orig, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
            if not np.all(np.isfinite(coeff_orig)):
                coeff_orig = np.zeros(6)
        except:
            coeff_orig = np.zeros(6)
    
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.where(np.isfinite(coeff_orig), coeff_orig, 0.0)
    
    return coeff_orig

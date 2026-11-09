import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = time.time() + 115.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    if n < 6:
        deg = max(0, n - 1)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full_coeff = np.zeros(6, dtype=np.float64)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Scale x to [-1, 1] for Chebyshev fit
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.array([y.mean(), 0, 0, 0, 0, 0], dtype=np.float64)

    x_center = (x_min + x_max) / 2.0
    x_half_range = x_range / 2.0
    x_scaled = (x - x_center) / x_half_range

    # Build Chebyshev Vandermonde matrix (degree 0 to 5)
    T = np.zeros((n, 6), dtype=np.float64)
    T[:, 0] = 1.0
    T[:, 1] = x_scaled
    T[:, 2] = 2.0 * x_scaled * T[:, 1] - T[:, 0]
    T[:, 3] = 2.0 * x_scaled * T[:, 2] - T[:, 1]
    T[:, 4] = 2.0 * x_scaled * T[:, 3] - T[:, 2]
    T[:, 5] = 2.0 * x_scaled * T[:, 4] - T[:, 3]

    # Regularization search using K-Fold Cross Validation
    # This is more robust than a single train/val split
    
    # Determine number of folds based on time and data size
    # 5 folds is a good balance. If n is small, use less.
    k_folds = 5
    if n < 500:
        k_folds = 3
    
    # Lambda grid: logarithmic spacing
    # trace(T^T T) gives a scale for the regularization
    # We compute T^T T once to get the trace and to speed up solves
    TtT_full = T.T @ T
    trace_TtT = np.trace(TtT_full)
    if trace_TtT < 1e-12:
        trace_TtT = 1.0
    
    # Lambda values as fraction of trace_TtT / 6
    # We want to cover a wide range of regularization strengths
    lambda_fracs = np.logspace(-8, 2, 40)
    
    best_lambda_frac = 0.0
    best_cv_mse = np.inf
    
    # Prepare indices for K-Fold
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_size = n // k_folds
    remainder = n % k_folds
    
    fold_indices = []
    start = 0
    for i in range(k_folds):
        size = fold_size + (1 if i < remainder else 0)
        end = start + size
        fold_indices.append(indices[start:end])
        start = end

    # Pre-compute TtT and Tty for each fold's training set?
    # Actually, for each lambda, we solve (TtT_train + lam*I) c = Tty_train
    # We can pre-compute TtT_train and Tty_train for each fold to save time
    
    fold_data = []
    for i in range(k_folds):
        val_idx = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(k_folds) if j != i])
        
        T_train = T[train_idx]
        y_train = y[train_idx]
        T_val = T[val_idx]
        y_val = y[val_idx]
        
        TtT_train = T_train.T @ T_train
        Tty_train = T_train.T @ y_train
        
        fold_data.append((TtT_train, Tty_train, T_val, y_val))

    # Search for best lambda
    eye_6 = np.eye(6, dtype=np.float64)
    
    for frac in lambda_fracs:
        if time.time() > deadline - 10.0:
            break
            
        lam = frac * trace_TtT / 6.0
        total_mse = 0.0
        
        for i in range(k_folds):
            TtT_train, Tty_train, T_val, y_val = fold_data[i]
            A = TtT_train + lam * eye_6
            try:
                cheb_coeff = np.linalg.solve(A, Tty_train)
                pred_val = T_val @ cheb_coeff
                mse = np.mean((pred_val - y_val) ** 2)
                total_mse += mse
            except np.linalg.LinAlgError:
                # If singular, try lstsq or skip
                try:
                    cheb_coeff, _, _, _ = np.linalg.lstsq(A, Tty_train, rcond=None)
                    pred_val = T_val @ cheb_coeff
                    mse = np.mean((pred_val - y_val) ** 2)
                    total_mse += mse
                except Exception:
                    total_mse += np.inf
                    break
        
        avg_mse = total_mse / k_folds
        if avg_mse < best_cv_mse:
            best_cv_mse = avg_mse
            best_lambda_frac = frac

    # Refit on full dataset with best lambda
    lam = best_lambda_frac * trace_TtT / 6.0
    A = TtT_full + lam * eye_6
    Tty_full = T.T @ y
    
    try:
        best_cheb_coeff = np.linalg.solve(A, Tty_full)
    except np.linalg.LinAlgError:
        try:
            best_cheb_coeff, _, _, _ = np.linalg.lstsq(A, Tty_full, rcond=None)
        except Exception:
            return np.zeros(6, dtype=np.float64)

    if not np.all(np.isfinite(best_cheb_coeff)):
        return np.zeros(6, dtype=np.float64)

    # Convert Chebyshev coeffs to Monomial coeffs in scaled variable z
    a0, a1, a2, a3, a4, a5 = best_cheb_coeff
    
    # T0 = 1
    # T1 = z
    # T2 = 2z^2 - 1
    # T3 = 4z^3 - 3z
    # T4 = 8z^4 - 8z^2 + 1
    # T5 = 16z^5 - 20z^3 + 5z
    
    # P(z) = a0*T0 + a1*T1 + a2*T2 + a3*T3 + a4*T4 + a5*T5
    # Coefficients for z^j:
    b0 = a0 - a2 + a4
    b1 = a1 - 3*a3 + 5*a5
    b2 = 2*a2 - 8*a4
    b3 = 4*a3 - 20*a5
    b4 = 8*a4
    b5 = 16*a5
    
    b = np.array([b0, b1, b2, b3, b4, b5], dtype=np.float64)
    
    # Convert from monomial coeffs in z to monomial coeffs in x
    # z = (x - c) / s
    # P(z) = sum_{j=0}^5 b_j z^j
    # z^j = (x-c)^j / s^j
    # (x-c)^j = sum_{k=0}^j binom(j, k) x^k (-c)^(j-k)
    
    c = x_center
    s = x_half_range
    
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for j in range(6):
        bj = b[j]
        if abs(bj) < 1e-15:
            continue
        inv_s_pow = 1.0 / (s ** j)
        for k in range(j + 1):
            binom_coeff = comb(j, k)
            power_c = (-c) ** (j - k)
            coeff_orig[k] += bj * (binom_coeff * power_c) * inv_s_pow
            
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6, dtype=np.float64)

    return coeff_orig.astype(np.float64)

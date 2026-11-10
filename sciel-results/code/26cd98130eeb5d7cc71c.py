"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _chebyshev_vander(x, deg):
    """
    Return Chebyshev polynomial basis matrix V of shape (N, deg+1).
    x is assumed to be in [-1, 1].
    V[:, j] = T_j(x)
    """
    N = len(x)
    V = np.zeros((N, deg + 1))
    V[:, 0] = 1.0
    if deg >= 1:
        V[:, 1] = x
    for j in range(2, deg + 1):
        V[:, j] = 2.0 * x * V[:, j - 1] - V[:, j - 2]
    return V

def _cheb_to_power_coeffs(c_cheb, deg=5):
    """
    Convert Chebyshev coefficients (ascending in T_j) to power coefficients
    for variable x in [-1,1]: p(x) = sum_j c_j T_j(x) = sum_i b_i x^i.
    Returns b of length deg+1.
    """
    b = np.zeros(deg + 1)
    for j in range(deg + 1):
        if c_cheb[j] == 0:
            continue
        # T_j(x) expressed as sum of x^i
        # T_0 = 1
        # T_1 = x
        # T_j = 2x T_{j-1} - T_{j-2}
        t = np.zeros(deg + 1)
        t[0] = 1.0
        if deg >= 1:
            t[1] = 1.0
        for k in range(2, j + 1):
            # 2x * t_{k-1}
            new_t = np.zeros(deg + 1)
            for i in range(deg):
                new_t[i + 1] += 2.0 * t[i]
            new_t -= t_prev if k >= 2 else np.zeros(deg + 1)
            if k >= 3:
                new_t -= t_prev2
            t_prev2 = t_prev
            t_prev = t
            t = new_t
        # Actually let me redo this more carefully
        pass
    
    # Simpler recursive approach
    b = np.zeros(deg + 1)
    # T_0
    b[0] += c_cheb[0]
    if deg >= 1:
        # T_1
        b[1] += c_cheb[1]
    for j in range(2, deg + 1):
        if c_cheb[j] == 0:
            continue
        # Compute T_j in power basis
        # T_0 = 1, T_1 = x
        # T_j = 2x*T_{j-1} - T_{j-2}
        t_jm2 = np.zeros(deg + 1)
        t_jm2[0] = 1.0  # T_0
        t_jm1 = np.zeros(deg + 1)
        t_jm1[1] = 1.0  # T_1
        if j == 2:
            t_j = 2.0 * t_jm1 * np.array([0, 1, 0, 0, 0, 0, 0]) - t_jm2
            # 2x * T_1 = 2x^2, minus T_0 = 1, so T_2 = 2x^2 - 1
            t_j = np.zeros(deg + 1)
            t_j[2] = 2.0
            t_j[0] = -1.0
        else:
            t_j = np.zeros(deg + 1)
            for i in range(deg):
                t_j[i + 1] += 2.0 * t_jm1[i]
            t_j -= t_jm2
            t_jm2, t_jm1 = t_jm1, t_j
            if j > 2:
                # Need to iterate properly
                pass
        
        # Let me just do a clean iterative computation
        break
    
    # Clean approach: compute all T_j in power basis iteratively
    T_power = [None] * (deg + 1)
    T_power[0] = np.zeros(deg + 1)
    T_power[0][0] = 1.0
    if deg >= 1:
        T_power[1] = np.zeros(deg + 1)
        T_power[1][1] = 1.0
    for j in range(2, deg + 1):
        T_power[j] = np.zeros(deg + 1)
        for i in range(deg):
            T_power[j][i + 1] += 2.0 * T_power[j - 1][i]
        T_power[j] -= T_power[j - 2]
    
    for j in range(deg + 1):
        b += c_cheb[j] * T_power[j]
    
    return b

def _convert_cheb_to_original(c_cheb, mu, sigma, deg=5):
    """
    c_cheb: Chebyshev coefficients for z in [-1,1]
    z = (x - mu) / sigma
    Returns power coefficients b for x.
    """
    # First convert Chebyshev to power basis in z
    b_z = _cheb_to_power_coeffs(c_cheb, deg)
    
    # Now expand sum_i b_z[i] * z^i where z = (x-mu)/sigma
    # = sum_i b_z[i] / sigma^i * (x-mu)^i
    # = sum_i b_z[i] / sigma^i * sum_k C(i,k) x^k (-mu)^(i-k)
    b_x = np.zeros(deg + 1)
    for i in range(deg + 1):
        if b_z[i] == 0:
            continue
        coeff = b_z[i] / (sigma ** i)
        for k in range(i + 1):
            # C(i, k) * (-mu)^(i-k)
            binom = 1
            for m in range(k):
                binom = binom * (i - m) // (m + 1)
            b_x[k] += coeff * binom * ((-mu) ** (i - k))
    return b_x

def _solve_ridge(V, y, lam):
    """Solve ridge regression with given design matrix and regularization."""
    N, P = V.shape
    A = V.T @ V + lam * np.eye(P)
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
        if np.all(np.isfinite(c)):
            return c
    except:
        pass
    try:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.all(np.isfinite(c)):
            return c
    except:
        pass
    return np.zeros(P)

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    n = len(x)
    if n < 10:
        return np.array([np.mean(y)] + [0.0] * 5)

    # Map x to [-1, 1] for Chebyshev
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    mu = (x_max + x_min) / 2.0
    sigma = x_range / 2.0
    z = (x - mu) / sigma  # Now in [-1, 1]

    deg = 5
    V_cheb = _chebyshev_vander(z, deg)

    # Cross-validation to select ridge parameter
    n_folds = 5
    if n < n_folds * 2:
        n_folds = 2
        if n < 4:
            return np.array([np.mean(y)] + [0.0] * 5)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((val_idx, train_idx))

    # Ridge lambda search on Chebyshev basis
    ridge_lambdas = np.logspace(-8, 2, 25)
    best_cv_mse = np.inf
    best_lambda = ridge_lambdas[0]

    for lam in ridge_lambdas:
        if time.time() > deadline - 20.0:
            break

        cv_mse_sum = 0.0
        valid_folds = 0

        for val_idx, train_idx in folds:
            if len(train_idx) < deg + 1:
                continue

            V_tr = V_cheb[train_idx]
            y_tr = y[train_idx]
            V_val = V_cheb[val_idx]
            y_val = y[val_idx]

            try:
                c = _solve_ridge(V_tr, y_tr, lam)
                if not np.all(np.isfinite(c)):
                    continue

                pred = V_val @ c
                mse = np.mean((pred - y_val) ** 2)
                cv_mse_sum += mse
                valid_folds += 1
            except Exception:
                continue

        if valid_folds > 0:
            avg_mse = cv_mse_sum / valid_folds
            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                best_lambda = lam

    # Refit on all data with best lambda
    try:
        c_cheb = _solve_ridge(V_cheb, y, best_lambda)
        if np.all(np.isfinite(c_cheb)):
            b_orig = _convert_cheb_to_original(c_cheb, mu, sigma, deg)
            if np.all(np.isfinite(b_orig)):
                # Final polish: check if we can improve with a tighter solve
                if time.time() < deadline - 5.0:
                    # Try a slightly different lambda for potential improvement
                    for lam_try in [best_lambda * 0.5, best_lambda * 2.0]:
                        if time.time() > deadline - 5.0:
                            break
                        c_try = _solve_ridge(V_cheb, y, lam_try)
                        if np.all(np.isfinite(c_try)):
                            # Evaluate on training data as proxy
                            pred_try = V_cheb @ c_try
                            mse_try = np.mean((pred_try - y) ** 2)
                            pred_best = V_cheb @ c_cheb
                            mse_best = np.mean((pred_best - y) ** 2)
                            if mse_try < mse_best:
                                c_cheb = c_try
                                b_orig = _convert_cheb_to_original(c_cheb, mu, sigma, deg)
                                if np.all(np.isfinite(b_orig)):
                                    break
                return b_orig
    except Exception:
        pass

    # Fallback: unregularized least squares
    try:
        c_cheb = np.linalg.lstsq(V_cheb, y, rcond=None)[0]
        if np.all(np.isfinite(c_cheb)):
            b_orig = _convert_cheb_to_original(c_cheb, mu, sigma, deg)
            if np.all(np.isfinite(b_orig)):
                return b_orig
    except Exception:
        pass

    return np.array([np.mean(y)] + [0.0] * 5)

# EVOLVE-BLOCK-END

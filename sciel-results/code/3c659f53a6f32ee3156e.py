"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _nCr(n, r):
    if r < 0 or r > n:
        return 0
    if r == 0:
        return 1
    r = min(r, n - r)
    res = 1
    for i in range(r):
        res = res * (n - i) // (i + 1)
    return res

def _convert_coeffs_to_original(a, mu, sigma, deg=5):
    """
    Convert polynomial coefficients 'a' for the variable z = (x-mu)/sigma
    to coefficients 'b' for the original variable x.
    y = sum(a_j * z^j) = sum(b_i * x^i)
    """
    b = np.zeros(deg + 1)
    for j in range(deg + 1):
        if a[j] == 0:
            continue
        coeff_a = a[j] / (sigma ** j)
        for k in range(j + 1):
            binom_val = _nCr(j, k)
            term_coeff = coeff_a * binom_val * ((-mu) ** (j - k))
            b[k] += term_coeff
    return b

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 10:
        return np.array([np.mean(y)] + [0.0]*5)

    # Standardize for conditioning, but map to Chebyshev node domain [-1, 1]
    # This gives a well-conditioned orthogonal basis on the data range.
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_min = x - 0.5
        x_max = x + 0.5
    z = 2.0 * (x - x_min) / (x_max - x_min) - 1.0  # in [-1, 1]

    deg = 5
    n = len(z)

    # Build Chebyshev basis matrix (orthogonal on [-1,1] with proper weighting)
    # T_k(cos(theta)) = cos(k*theta). Use recurrence for stability.
    T = np.zeros((n, deg + 1))
    T[:, 0] = 1.0
    if deg >= 1:
        T[:, 1] = z
    for k in range(2, deg + 1):
        T[:, k] = 2.0 * z * T[:, k-1] - T[:, k-2]

    # Weights for Chebyshev quadrature to approximate L2 norm
    # Use Gauss-Chebyshev-like weights: w_k = pi/2 * (1 for k=0, 2 for k=deg, 1 otherwise)
    # But since we're doing weighted least squares on data, we use empirical weights
    # Actually, for ridge regression, we can just use the Chebyshev basis directly
    # as it's well-conditioned. The orthogonality helps with the conditioning of V^T V.

    # Precompute for CV
    n_folds = 5
    if n < n_folds * 2:
        n_folds = max(2, n // 2)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        folds.append((indices[start:end], np.concatenate([indices[:start], indices[end:]])))

    best_cv_mse = np.inf
    best_lambda = 1.0

    # Precompute V^T V and V^T y for each fold
    fold_data = []
    for val_idx, train_idx in folds:
        if len(train_idx) < deg + 1:
            continue
        V_tr = T[train_idx]
        y_tr = y[train_idx]
        V_val = T[val_idx]
        y_val = y[val_idx]
        VtV = V_tr.T @ V_tr
        Vty = V_tr.T @ y_tr
        fold_data.append((VtV, Vty, V_val, y_val))

    I = np.eye(deg + 1)

    # Stage 1: Coarse search over a wide range
    lambdas_coarse = np.logspace(-8, 8, 80)

    for lam in lambdas_coarse:
        if time.time() > deadline - 30.0:
            break
        cv_mse_sum = 0.0
        valid_folds = 0
        for VtV, Vty, V_val, y_val in fold_data:
            try:
                A = VtV + lam * I
                c = np.linalg.solve(A, Vty)
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

    # Stage 2: Fine search around best_lambda
    if time.time() < deadline - 25.0:
        ratio = 1.5
        lam_lo = max(best_lambda / ratio, 1e-12)
        lam_hi = best_lambda * ratio
        lambdas_fine = np.logspace(np.log10(lam_lo), np.log10(lam_hi), 50)
        lambdas_fine = np.unique(lambdas_fine)

        for lam in lambdas_fine:
            if time.time() > deadline - 15.0:
                break
            cv_mse_sum = 0.0
            valid_folds = 0
            for VtV, Vty, V_val, y_val in fold_data:
                try:
                    A = VtV + lam * I
                    c = np.linalg.solve(A, Vty)
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

    # Stage 3: Very fine search around best_lambda
    if time.time() < deadline - 10.0:
        ratio = 1.15
        lam_lo = max(best_lambda / ratio, 1e-12)
        lam_hi = best_lambda * ratio
        lambdas_vfine = np.logspace(np.log10(lam_lo), np.log10(lam_hi), 40)
        lambdas_vfine = np.unique(lambdas_vfine)

        for lam in lambdas_vfine:
            if time.time() > deadline - 5.0:
                break
            cv_mse_sum = 0.0
            valid_folds = 0
            for VtV, Vty, V_val, y_val in fold_data:
                try:
                    A = VtV + lam * I
                    c = np.linalg.solve(A, Vty)
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

    # Refit on all data with best_lambda
    VtV_full = T.T @ T
    Vty_full = T.T @ y

    if best_lambda < 1e-10:
        # Use lstsq for very small lambda
        try:
            c_cheb, _, _, _ = np.linalg.lstsq(T, y, rcond=None)
            if np.all(np.isfinite(c_cheb)):
                # Convert Chebyshev coefficients to standard power basis on z,
                # then to original x
                # First, convert Chebyshev to power basis on z
                # Use recurrence: T_0=1, T_1=z, T_k=2z*T_{k-1}-T_{k-2}
                # We need to express T_k as a polynomial in z
                # Build the conversion matrix
                conv = np.zeros((deg+1, deg+1))
                conv[0, 0] = 1.0
                if deg >= 1:
                    conv[1, 1] = 1.0
                for k in range(2, deg+1):
                    # T_k = 2z*T_{k-1} - T_{k-2}
                    conv[k, 1:k+1] += 2.0 * conv[k-1, 0:k]
                    conv[k, 0:k] -= conv[k-2, 0:k]
                # Now c_cheb @ conv gives coefficients in power basis of z
                c_z = c_cheb @ conv
                # Convert z = (x - x_min)/(x_max - x_min) * 2 - 1
                # z = (2/(x_max-x_min))*x - (2*x_min/(x_max-x_min)) - 1
                # z = a*x + b where a = 2/(x_max-x_min), b = -(2*x_min/(x_max-x_min)) - 1
                a_z = 2.0 / (x_max - x_min)
                b_z = -(2.0 * x_min / (x_max - x_min)) - 1.0
                # Now convert from power basis in z to power basis in x
                # z = a*x + b
                # z^j = (a*x + b)^j = sum_{k=0}^j C(j,k) * a^k * b^{j-k} * x^k
                c_x = np.zeros(deg+1)
                for j in range(deg+1):
                    if c_z[j] == 0:
                        continue
                    for k in range(j+1):
                        c_x[k] += c_z[j] * _nCr(j, k) * (a_z ** k) * (b_z ** (j-k))
                if np.all(np.isfinite(c_x)):
                    return c_x
        except Exception:
            pass
    else:
        A = VtV_full + best_lambda * I
        try:
            c_cheb = np.linalg.solve(A, Vty_full)
            if np.all(np.isfinite(c_cheb)):
                # Convert Chebyshev to power basis on z
                conv = np.zeros((deg+1, deg+1))
                conv[0, 0] = 1.0
                if deg >= 1:
                    conv[1, 1] = 1.0
                for k in range(2, deg+1):
                    conv[k, 1:k+1] += 2.0 * conv[k-1, 0:k]
                    conv[k, 0:k] -= conv[k-2, 0:k]
                c_z = c_cheb @ conv
                a_z = 2.0 / (x_max - x_min)
                b_z = -(2.0 * x_min / (x_max - x_min)) - 1.0
                c_x = np.zeros(deg+1)
                for j in range(deg+1):
                    if c_z[j] == 0:
                        continue
                    for k in range(j+1):
                        c_x[k] += c_z[j] * _nCr(j, k) * (a_z ** k) * (b_z ** (j-k))
                if np.all(np.isfinite(c_x)):
                    return c_x
        except Exception:
            pass

    # Fallback: Unregularized fit in Chebyshev basis
    try:
        c_cheb, _, _, _ = np.linalg.lstsq(T, y, rcond=None)
        if np.all(np.isfinite(c_cheb)):
            conv = np.zeros((deg+1, deg+1))
            conv[0, 0] = 1.0
            if deg >= 1:
                conv[1, 1] = 1.0
            for k in range(2, deg+1):
                conv[k, 1:k+1] += 2.0 * conv[k-1, 0:k]
                conv[k, 0:k] -= conv[k-2, 0:k]
            c_z = c_cheb @ conv
            a_z = 2.0 / (x_max - x_min)
            b_z = -(2.0 * x_min / (x_max - x_min)) - 1.0
            c_x = np.zeros(deg+1)
            for j in range(deg+1):
                if c_z[j] == 0:
                    continue
                for k in range(j+1):
                    c_x[k] += c_z[j] * _nCr(j, k) * (a_z ** k) * (b_z ** (j-k))
            if np.all(np.isfinite(c_x)):
                return c_x
    except Exception:
        pass

    return np.array([np.mean(y)] + [0.0]*5)

# EVOLVE-BLOCK-END

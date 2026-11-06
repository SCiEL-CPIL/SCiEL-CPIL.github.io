"""Tikhonov-regularized degree-5 polynomial fit with efficient CV and coefficient transformation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import comb


def construct(training_x, training_subgrid_tendency):
    """Fit a shared degree-5 polynomial to the unresolved tendency.
    
    Uses Tikhonov (L2) regularization with the strength selected by internal
    k-fold cross-validation to balance training fit and generalization to
    unseen forcing regimes (e.g., F=10).
    """
    t0 = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)

    if n < 6:
        return np.zeros(6)

    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        c = np.zeros(6)
        c[0] = y.mean()
        return c

    # Scale x to [-1, 1]
    x_scaled = 2.0 * (x - x_min) / x_range - 1.0
    V = np.vander(x_scaled, N=6, increasing=True)

    # Precompute transformation matrix T such that c_orig = T @ c_scaled
    a = 2.0 / x_range
    b = -(2.0 * x_min / x_range + 1.0)
    
    T = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            T[j, i] = comb(i, j) * (a ** j) * (b ** (i - j))

    # Cross-validation setup
    n_folds = 5
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    # Precompute V^T V and V^T y for each training fold
    fold_data = []
    for f in range(n_folds):
        val_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        
        V_train = V[train_idx]
        y_train = y[train_idx]
        V_val = V[val_idx]
        y_val = y[val_idx]
        
        # Precompute Gram matrix and cross-product
        VtV = V_train.T @ V_train  # 6x6
        Vty = V_train.T @ y_train  # 6x1
        
        fold_data.append((VtV, Vty, V_val, y_val))

    def solve_ridge_from_precomputed(VtV, Vty, lam):
        """Solve ridge regression using precomputed V^T V and V^T y."""
        if lam == 0.0:
            try:
                return np.linalg.solve(VtV, Vty)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(VtV, Vty, rcond=None)[0]
        else:
            A = VtV + lam * np.eye(6)
            try:
                return np.linalg.solve(A, Vty)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(A, Vty, rcond=None)[0]

    def compute_cv_score(lam):
        """Compute mean CV MSE for a given lambda using precomputed fold data."""
        cv_errors = []
        for VtV, Vty, V_val, y_val in fold_data:
            c_train = solve_ridge_from_precomputed(VtV, Vty, lam)
            pred_val = V_val @ c_train
            cv_errors.append(np.mean((pred_val - y_val) ** 2))
        if cv_errors:
            return np.mean(cv_errors)
        return np.inf

    # Coarse lambda search
    lambdas_coarse = np.concatenate([
        [0.0],
        np.logspace(-10, 6, 40)
    ])

    best_lambda = 0.0
    best_cv_score = np.inf

    for lam in lambdas_coarse:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_score = compute_cv_score(lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Fine lambda search around best_lambda
    if best_lambda > 0:
        lower = max(best_lambda * 0.5, 1e-15)
        upper = best_lambda * 2.0
        lambdas_fine = np.logspace(np.log10(lower), np.log10(upper), 20)
    else:
        lambdas_fine = np.logspace(-10, -2, 20)

    for lam in lambdas_fine:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_score = compute_cv_score(lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Final fit on all data
    VtV_full = V.T @ V
    Vty_full = V.T @ y
    coeff_scaled = solve_ridge_from_precomputed(VtV_full, Vty_full, best_lambda)

    # Transform coefficients
    coeff = T @ coeff_scaled

    if not np.all(np.isfinite(coeff)):
        coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)

    return coeff
# EVOLVE-BLOCK-END

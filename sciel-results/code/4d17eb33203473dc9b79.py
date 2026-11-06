"""Rewritten program: Tikhonov-regularized degree-5 polynomial fit with internal CV and SVD solver."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import comb


def construct(training_x, training_subgrid_tendency):
    """Fit a shared degree-5 polynomial to the unresolved tendency.
    
    Uses Tikhonov (L2) regularization with the strength selected by internal
    k-fold cross-validation to balance training fit and generalization to
    unseen forcing regimes (e.g., F=10).
    Improvements:
    - SVD-based ridge regression for numerical stability.
    - Precomputed coefficient transformation matrix.
    - Two-stage lambda search (coarse then fine).
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
    # c_scaled[i] is coeff for (ax+b)^i
    # c_orig[j] is coeff for x^j
    # (ax+b)^i = sum_{j=0}^i comb(i, j) a^j b^(i-j) x^j
    a = 2.0 / x_range
    b = -(2.0 * x_min / x_range + 1.0)
    
    T = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            T[j, i] = comb(i, j) * (a ** j) * (b ** (i - j))

    # SVD of V for stable ridge regression
    # V = U @ diag(S) @ Vt
    try:
        U, S, Vt = np.linalg.svd(V, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if SVD fails (unlikely)
        U, S, Vt = None, None, None

    def solve_ridge(lam):
        """Solve ridge regression using SVD components."""
        if U is None:
            # Fallback
            if lam == 0.0:
                c, *_ = np.linalg.lstsq(V, y, rcond=None)
            else:
                A = V.T @ V + lam * np.eye(6)
                b_vec = V.T @ y
                c = np.linalg.solve(A, b_vec)
            return c

        # c = Vt.T @ (S^2 + lam I)^-1 @ S @ U.T @ y
        # Note: S is 1D array of singular values.
        # Let S_mat = diag(S).
        # (S_mat^T S_mat + lam I)^-1 S_mat^T = diag(S) (S^2 + lam I)^-1
        # So c = Vt.T @ diag( S / (S^2 + lam) ) @ (U.T @ y)
        
        if lam == 0.0:
            # Avoid division by zero for very small S
            # Use pseudo-inverse logic
            inv_S = np.zeros_like(S)
            mask = S > 1e-12
            inv_S[mask] = 1.0 / S[mask]
            # c = Vt.T @ inv_S @ (U.T @ y)
            Uy = U.T @ y
            c = Vt.T @ (inv_S * Uy)
        else:
            factor = S / (S**2 + lam)
            Uy = U.T @ y
            c = Vt.T @ (factor * Uy)
            
        return c

    # Cross-validation setup
    n_folds = 5
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

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
        cv_errors = []
        for f in range(n_folds):
            val_idx = folds[f]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])

            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]

            # Solve ridge on train fold
            # Reuse SVD? SVD is for full V. For folds, V is different.
            # SVD per fold is expensive but n is likely not huge.
            # If n is large, this might be slow.
            # Let's check time.
            try:
                U_f, S_f, Vt_f = np.linalg.svd(V_train, full_matrices=False)
            except np.linalg.LinAlgError:
                U_f, S_f, Vt_f = None, None, None

            if U_f is not None:
                if lam == 0.0:
                    inv_S_f = np.zeros_like(S_f)
                    mask_f = S_f > 1e-12
                    inv_S_f[mask_f] = 1.0 / S_f[mask_f]
                    Uy_f = U_f.T @ y_train
                    c_train = Vt_f.T @ (inv_S_f * Uy_f)
                else:
                    factor_f = S_f / (S_f**2 + lam)
                    Uy_f = U_f.T @ y_train
                    c_train = Vt_f.T @ (factor_f * Uy_f)
            else:
                # Fallback
                if lam == 0.0:
                    c_train, *_ = np.linalg.lstsq(V_train, y_train, rcond=None)
                else:
                    A = V_train.T @ V_train + lam * np.eye(6)
                    b_vec = V_train.T @ y_train
                    c_train = np.linalg.solve(A, b_vec)

            pred_val = V_val @ c_train
            cv_errors.append(np.mean((pred_val - y_val) ** 2))

        if cv_errors:
            mean_cv = np.mean(cv_errors)
            if mean_cv < best_cv_score:
                best_cv_score = mean_cv
                best_lambda = lam

    # Fine lambda search around best_lambda
    if best_lambda > 0:
        # Search in a small range around best_lambda
        lower = best_lambda * 0.5
        upper = best_lambda * 2.0
        lambdas_fine = np.logspace(np.log10(lower), np.log10(upper), 20)
    else:
        # If best was 0, search small positive values
        lambdas_fine = np.logspace(-10, -2, 20)

    for lam in lambdas_fine:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_errors = []
        for f in range(n_folds):
            val_idx = folds[f]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])

            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]

            try:
                U_f, S_f, Vt_f = np.linalg.svd(V_train, full_matrices=False)
            except np.linalg.LinAlgError:
                U_f, S_f, Vt_f = None, None, None

            if U_f is not None:
                if lam == 0.0:
                    inv_S_f = np.zeros_like(S_f)
                    mask_f = S_f > 1e-12
                    inv_S_f[mask_f] = 1.0 / S_f[mask_f]
                    Uy_f = U_f.T @ y_train
                    c_train = Vt_f.T @ (inv_S_f * Uy_f)
                else:
                    factor_f = S_f / (S_f**2 + lam)
                    Uy_f = U_f.T @ y_train
                    c_train = Vt_f.T @ (factor_f * Uy_f)
            else:
                if lam == 0.0:
                    c_train, *_ = np.linalg.lstsq(V_train, y_train, rcond=None)
                else:
                    A = V_train.T @ V_train + lam * np.eye(6)
                    b_vec = V_train.T @ y_train
                    c_train = np.linalg.solve(A, b_vec)

            pred_val = V_val @ c_train
            cv_errors.append(np.mean((pred_val - y_val) ** 2))

        if cv_errors:
            mean_cv = np.mean(cv_errors)
            if mean_cv < best_cv_score:
                best_cv_score = mean_cv
                best_lambda = lam

    # Final fit on all data
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        # Return best found so far? We don't store it. 
        # Just do a quick unregularized or regularized fit with best_lambda
        pass

    coeff_scaled = solve_ridge(best_lambda)
    
    # Transform coefficients
    coeff = T @ coeff_scaled

    if not np.all(np.isfinite(coeff)):
        coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)

    return coeff
# EVOLVE-BLOCK-END

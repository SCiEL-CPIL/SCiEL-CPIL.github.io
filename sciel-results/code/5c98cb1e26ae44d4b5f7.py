"""Rewritten program: odd-symmetric degree-5 polynomial with IRLS + ridge + CV lambda selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 145.0

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = x.size

    if n < 6:
        return np.zeros(6)

    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    # Odd-symmetric basis: x, x^3, x^5 in scaled coordinates
    # This exploits the antisymmetry of the subgrid tendency in X
    V_odd = np.column_stack([x_scaled, x_scaled**3, x_scaled**5])
    p = 3  # number of odd-power coefficients

    # Full Vandermonde for final rescaling (degree 5, all powers)
    V_full = np.vander(x_scaled, N=6, increasing=True)

    # Regularization: penalize all odd-power coefficients
    P_odd = np.ones(p)

    def solve_ridge(V_mat, y_vec, lam):
        """Solve (V^T V + lam * I) c = V^T y."""
        A = V_mat.T @ V_mat + lam * np.eye(p)
        b = V_mat.T @ y_vec
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    # K-fold cross-validation for lambda selection
    y_var = np.var(y)
    if y_var < 1e-12:
        y_var = 1.0
    lam_grid = np.logspace(-10, 4, 41) * y_var / n

    def cv_score(lam, n_folds=5, seed=42):
        """K-fold CV mean MSE for given lambda."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        fold_size = n // n_folds
        cv_mses = []
        for k in range(n_folds):
            val_start = k * fold_size
            val_end = val_start + fold_size if k < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            trn_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            V_trn = V_odd[trn_idx]
            y_trn = y[trn_idx]
            V_val = V_odd[val_idx]
            y_val = y[val_idx]

            c = solve_ridge(V_trn, y_trn, lam)
            pred = V_val @ c
            cv_mses.append(np.mean((y_val - pred) ** 2))
        return np.mean(cv_mses)

    # Select best lambda via CV
    best_lam = lam_grid[0]
    best_cv = np.inf
    for lam in lam_grid:
        if time.time() > deadline - 60.0:
            break
        cv = cv_score(lam)
        if cv < best_cv:
            best_cv = cv
            best_lam = lam

    # Fine-tune lambda around the CV optimum
    if time.time() < deadline - 55.0:
        lam_fine = np.linspace(best_lam * 0.5, best_lam * 2.0, 11)
        for lam in lam_fine:
            cv = cv_score(lam)
            if cv < best_cv:
                best_cv = cv
                best_lam = lam

    # Robust IRLS with Tukey's biweight, using selected lambda
    # Initial OLS fit
    try:
        coeff_odd, _, _, _ = np.linalg.lstsq(V_odd, y, rcond=None)
    except np.linalg.LinAlgError:
        coeff_odd = np.zeros(p)

    pred = V_odd @ coeff_odd
    resid = y - pred
    scale = np.median(np.abs(resid)) / 0.6745
    if scale < 1e-12:
        scale = np.std(resid) + 1e-12

    def tukey_weights(r, c=4.685):
        ar = np.abs(r) / scale
        w = np.zeros_like(ar)
        mask = ar <= c
        w[mask] = (1.0 - (ar[mask] / c) ** 2) ** 2
        return w

    best_coeff_odd = coeff_odd.copy()
    best_mse = np.mean(resid ** 2)
    prev_coeff = best_coeff_odd.copy()

    # IRLS iterations
    for it in range(20):
        if time.time() > deadline - 20.0:
            break
        w = tukey_weights(resid)
        if w.sum() < n * 0.1:
            w = np.ones(n)
        sqrt_w = np.sqrt(w)
        Vw = V_odd * sqrt_w[:, None]
        yw = y * sqrt_w

        # Try lambda near the CV-selected value
        lam_candidates = [best_lam]
        if best_lam > 0:
            lam_candidates.extend([best_lam * 0.3, best_lam * 0.5, best_lam * 1.5, best_lam * 3.0])

        for lam in lam_candidates:
            try:
                c_reg = solve_ridge(Vw, yw, lam)
                pred_reg = V_odd @ c_reg
                mse_reg = np.mean((y - pred_reg) ** 2)
                if mse_reg < best_mse:
                    best_mse = mse_reg
                    best_coeff_odd = c_reg
            except np.linalg.LinAlgError:
                continue

        # Update residuals
        pred = V_odd @ best_coeff_odd
        resid = y - pred
        new_scale = np.median(np.abs(resid)) / 0.6745
        if new_scale < 1e-12:
            new_scale = np.std(resid) + 1e-12
        scale = new_scale

        # Convergence check
        if it > 0 and np.max(np.abs(best_coeff_odd - prev_coeff)) < 1e-12:
            break
        prev_coeff = best_coeff_odd.copy()

    # Final polish: try a broader range of lambdas with the final IRLS weights
    if time.time() < deadline - 10.0:
        w_final = tukey_weights(resid)
        if w_final.sum() < n * 0.1:
            w_final = np.ones(n)
        sqrt_wf = np.sqrt(w_final)
        Vw_f = V_odd * sqrt_wf[:, None]
        yw_f = y * sqrt_wf

        lam_broad = np.logspace(-6, 2, 21) * y_var / n
        for lam in lam_broad:
            if time.time() > deadline - 10.0:
                break
            try:
                c_f = solve_ridge(Vw_f, yw_f, lam)
                pred_f = V_odd @ c_f
                mse_f = np.mean((y - pred_f) ** 2)
                if mse_f < best_mse:
                    best_mse = mse_f
                    best_coeff_odd = c_f
            except np.linalg.LinAlgError:
                continue

    # Convert odd-symmetric scaled coefficients to full 6-coefficient vector in original x basis
    # p(x_scaled) = c0*x + c1*x^3 + c2*x^5 where x_scaled = (x - m)/s
    # p((x-m)/s) = c0*(x-m)/s + c1*((x-m)/s)^3 + c2*((x-m)/s)^5
    # = c0/s * (x-m) + c1/s^3 * (x-m)^3 + c2/s^5 * (x-m)^5
    # Expand each (x-m)^k using binomial theorem
    d = np.zeros(6)
    for k, ck in zip([1, 3, 5], best_coeff_odd):
        ck_scaled = ck / (x_std ** k)
        # (x - m)^k = sum_{j=0}^{k} C(k,j) * x^j * (-m)^{k-j}
        for j in range(k + 1):
            d[j] += ck_scaled * comb(k, j) * ((-x_mean) ** (k - j))

    # Ensure finite
    d = np.where(np.isfinite(d), d, 0.0)
    return d
# EVOLVE-BLOCK-END

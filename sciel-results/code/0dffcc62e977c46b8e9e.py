"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _ridge_fit_scaled(x, y, deg, lam, col_scales=None):
    """Fit polynomial with scale-aware ridge regularization.
    
    col_scales: per-column std of the Vandermonde matrix, used to normalize
    the regularization so each coefficient is penalized relative to its natural scale.
    Returns 6-coeff array or None.
    """
    if lam <= 0.0:
        try:
            c = np.polynomial.polynomial.polyfit(x, y, deg)
            if not np.all(np.isfinite(c)):
                return None
            out = np.zeros(6)
            out[:deg + 1] = c
            return out
        except Exception:
            return None

    V = np.polynomial.polynomial.polyvander(x, deg)
    n_coeffs = deg + 1

    if col_scales is None:
        col_scales = np.std(V, axis=0)
        col_scales[col_scales < 1e-12] = 1.0

    # Scale-aware regularization: penalty = lam * (col_scale * c_i)^2
    # This is equivalent to penalizing c_i with weight lam * col_scale_i^2
    reg_diag = (lam * col_scales ** 2)
    reg_diag[0] = 0.0  # Don't penalize the intercept

    try:
        A = V.T @ V + np.diag(reg_diag)
        b = V.T @ y
        c = np.linalg.solve(A, b)
        if not np.all(np.isfinite(c)):
            return None
        out = np.zeros(6)
        out[:n_coeffs] = c
        return out
    except Exception:
        return None


def _cv_mse_scaled(x, y, deg, lam, n_folds=5, seed=42):
    """K-fold CV MSE with scale-aware ridge."""
    n = len(x)
    if n < 2 * n_folds:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        c = _ridge_fit_scaled(x, y, deg, lam)
        if c is None:
            return np.inf
        pred = np.polynomial.polynomial.polyval(x, c)
        return np.mean((pred - y) ** 2)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    total_mse = 0.0
    valid_folds = 0

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(train_idx) < deg + 1:
            continue

        c = _ridge_fit_scaled(x[train_idx], y[train_idx], deg, lam)
        if c is None:
            continue

        pred = np.polynomial.polynomial.polyval(x[val_idx], c)
        total_mse += np.mean((pred - y[val_idx]) ** 2)
        valid_folds += 1

    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 6:
        return np.zeros(6)

    best_coeff = None
    best_score = np.inf

    # Phase 1: Coarse search over (degree, lambda)
    # Use log-spaced lambdas for finer resolution
    coarse_lambdas = [0.0] + list(np.logspace(-6, 2, 17))
    degrees = [3, 4, 5]

    candidates = []
    for deg in degrees:
        for lam in coarse_lambdas:
            candidates.append((deg, lam))

    # Sort candidates: prefer higher degree with moderate regularization
    # Actually, just iterate all
    for deg, lam in candidates:
        if time.time() > deadline - 10.0:
            break

        score = _cv_mse_scaled(x, y, deg, lam, n_folds=5, seed=42)

        if score < best_score:
            best_score = score
            c = _ridge_fit_scaled(x, y, deg, lam)
            if c is not None:
                best_coeff = c
                best_deg = deg
                best_lam = lam

    if best_coeff is None:
        # Fallback
        try:
            c = np.polynomial.polynomial.polyfit(x, y, 5)
            if np.all(np.isfinite(c)):
                out = np.zeros(6)
                out[:6] = c
                return out
        except Exception:
            pass
        return np.zeros(6)

    # Phase 2: Fine-tune lambda around the best value
    if time.time() < deadline - 15.0:
        fine_lambdas = []
        if best_lam > 0:
            fine_lambdas = list(np.logspace(np.log10(best_lam) - 0.5, np.log10(best_lam) + 0.5, 11))
        else:
            fine_lambdas = [0.0] + list(np.logspace(-8, -3, 11))

        for lam in fine_lambdas:
            if time.time() > deadline - 10.0:
                break
            score = _cv_mse_scaled(x, y, best_deg, lam, n_folds=5, seed=42)
            if score < best_score:
                best_score = score
                c = _ridge_fit_scaled(x, y, best_deg, lam)
                if c is not None:
                    best_coeff = c
                    best_lam = lam

    # Phase 3: Try a few more degrees near the best with the best lambda
    if time.time() < deadline - 10.0:
        for deg in [max(1, best_deg - 1), best_deg, min(5, best_deg + 1)]:
            if deg == best_deg:
                continue
            if time.time() > deadline - 5.0:
                break
            score = _cv_mse_scaled(x, y, deg, best_lam, n_folds=5, seed=42)
            if score < best_score:
                best_score = score
                c = _ridge_fit_scaled(x, y, deg, best_lam)
                if c is not None:
                    best_coeff = c

    # Phase 4: Final polish — re-fit with all data using the best (deg, lam)
    # Use a slightly tighter solve
    if time.time() < deadline - 2.0:
        V = np.polynomial.polynomial.polyvander(x, best_deg)
        col_scales = np.std(V, axis=0)
        col_scales[col_scales < 1e-12] = 1.0
        n_coeffs = best_deg + 1

        if best_lam <= 0.0:
            try:
                c = np.polynomial.polynomial.polyfit(x, y, best_deg)
                if np.all(np.isfinite(c)):
                    out = np.zeros(6)
                    out[:n_coeffs] = c
                    return out
            except Exception:
                pass
        else:
            reg_diag = (best_lam * col_scales ** 2)
            reg_diag[0] = 0.0
            try:
                A = V.T @ V + np.diag(reg_diag)
                b = V.T @ y
                # Use lstsq for numerical robustness as a polish
                c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                if np.all(np.isfinite(c)):
                    out = np.zeros(6)
                    out[:n_coeffs] = c
                    return out
            except Exception:
                pass

    if best_coeff is not None and np.all(np.isfinite(best_coeff)):
        return best_coeff

    # Ultimate fallback
    try:
        c = np.polynomial.polynomial.polyfit(x, y, 5)
        if np.all(np.isfinite(c)):
            out = np.zeros(6)
            out[:6] = c
            return out
    except Exception:
        pass
    return np.zeros(6)
# EVOLVE-BLOCK-END

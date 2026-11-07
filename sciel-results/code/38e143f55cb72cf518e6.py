"""
Structural change: Replace fixed-degree polynomial regression with an adaptive
ensemble that searches over multiple model families (polynomial degrees, Fourier
series, RBF kernels, splines) using cross-validation on the training data.

Key design choices:
1. Generate candidate feature matrices for several basis families.
2. For each candidate, fit via ridge regression (Tikhonov regularization) to
   handle potential overfitting and ill-conditioning.
3. Select the best candidate via 5-fold cross-validation on the training set.
4. Time guard ensures we finish well before the 300s limit.

This is a structural change from the seed's fixed degree-2 polynomial to an
adaptive model selection across multiple functional forms.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def _make_polynomial_features(X, degree):
    """Polynomial features up to given degree for d-dimensional input."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for idx in itertools.combinations_with_replacement(range(d), deg):
            feat = np.ones(n)
            for j in idx:
                feat = feat * X[:, j]
            cols.append(feat)
    return np.column_stack(cols)


def _make_fourier_features(X, n_freq):
    """Fourier features: sin/cos of k*x for k=1..n_freq, for each dimension."""
    n, d = X.shape
    cols = [np.ones(n)]
    for dim in range(d):
        x = X[:, dim]
        for k in range(1, n_freq + 1):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _make_rbf_features(X, centers, bandwidths):
    """RBF features: exp(-||x - c_j||^2 / (2 * bw_j^2)) for given centers/bandwidths."""
    n, d = X.shape
    cols = [np.ones(n)]
    for c, bw in zip(centers, bandwidths):
        diff = X - c
        dist_sq = np.sum(diff ** 2, axis=1)
        cols.append(np.exp(-dist_sq / (2.0 * bw ** 2)))
    return np.column_stack(cols)


def _make_spline_features(X, n_knots):
    """Cubic B-spline-like features using piecewise polynomials with knots."""
    n, d = X.shape
    cols = [np.ones(n)]
    for dim in range(d):
        x = X[:, dim]
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-12:
            cols.append(x.copy())
            continue
        knots = np.linspace(x_min, x_max, n_knots + 1)
        # Use truncated power basis for cubic splines
        for k in range(1, n_knots):
            for power in [1, 2, 3]:
                cols.append(np.maximum(x - knots[k], 0) ** power)
        # Also include linear term
        cols.append(x)
    return np.column_stack(cols)


def _ridge_fit(A, y, alpha):
    """Solve ridge regression: (A^T A + alpha*I) w = A^T y."""
    n_features = A.shape[1]
    # Don't regularize the intercept (first column)
    reg = np.eye(n_features) * alpha
    reg[0, 0] = 0.0
    ATA = A.T @ A + reg
    ATy = A.T @ y
    try:
        w = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def _cross_validate(A, y, alpha, n_folds=5):
    """Simple k-fold CV mean squared error."""
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    mses = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        A_tr, y_tr = A[train_idx], y[train_idx]
        A_val, y_val = A[val_idx], y[val_idx]
        
        w = _ridge_fit(A_tr, y_tr, alpha)
        pred = A_val @ w
        mse = np.mean((pred - y_val) ** 2)
        mses.append(mse)
    return np.mean(mses)


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    n, d = X_train.shape
    
    # Normalize features for better conditioning
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    x_std[x_std < 1e-10] = 1.0
    
    def _normalize(X):
        return (X - x_mean) / x_std
    
    X_norm = _normalize(X_train)
    
    # Generate candidate feature sets
    candidates = []  # list of (feature_matrix, name)
    
    # Polynomial features: degrees 1, 2, 3, 4
    for deg in [1, 2, 3, 4]:
        try:
            A = _make_polynomial_features(X_norm, deg)
            candidates.append((A, f"poly_deg{deg}"))
        except Exception:
            pass
    
    # Fourier features: 3, 5, 8 frequencies
    for nf in [3, 5, 8]:
        try:
            A = _make_fourier_features(X_norm, nf)
            candidates.append((A, f"fourier_{nf}"))
        except Exception:
            pass
    
    # RBF features with different numbers of centers and bandwidths
    # Use quantile-based centers for robustness
    n_centers_options = [5, 10, 15]
    bandwidth_options = [0.3, 0.5, 1.0]
    for nc in n_centers_options:
        for bw in bandwidth_options:
            try:
                # Select centers as quantiles
                if d == 1:
                    centers = np.quantile(X_norm, np.linspace(0.1, 0.9, nc)).reshape(-1, 1)
                else:
                    # For multi-dim, use grid of quantiles
                    centers = []
                    for dim in range(d):
                        q = np.quantile(X_norm[:, dim], np.linspace(0.1, 0.9, nc))
                        for qi in range(nc):
                            c = np.zeros(d)
                            c[dim] = q[qi]
                            centers.append(c)
                    centers = np.array(centers[:nc * d])
                    # Subsample to nc centers
                    if len(centers) > nc:
                        idx = np.random.choice(len(centers), nc, replace=False)
                        centers = centers[idx]
                bandwidths = np.full(len(centers), bw)
                A = _make_rbf_features(X_norm, centers, bandwidths)
                candidates.append((A, f"rbf_c{nc}_bw{bw}"))
            except Exception:
                pass
    
    # Spline features
    for nk in [4, 6, 8]:
        try:
            A = _make_spline_features(X_norm, nk)
            candidates.append((A, f"spline_{nk}"))
        except Exception:
            pass
    
    if not candidates:
        # Fallback: simple linear
        A = np.column_stack([np.ones(n), X_norm])
        candidates.append((A, "linear"))
    
    # Cross-validate each candidate with a few regularization strengths
    best_cv_mse = np.inf
    best_A = None
    best_name = None
    best_alpha = None
    
    alphas = [1e-6, 1e-4, 1e-2, 1.0]
    
    for A, name in candidates:
        if time.time() > deadline - 30:
            break
        
        # Quick check: if too many features relative to samples, skip
        if A.shape[1] > n * 2:
            continue
        
        for alpha in alphas:
            if time.time() > deadline - 30:
                break
            try:
                cv_mse = _cross_validate(A, y_train, alpha, n_folds=5)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_A = A
                    best_name = name
                    best_alpha = alpha
            except Exception:
                continue
    
    # If no candidate worked, fall back to linear
    if best_A is None:
        A = np.column_stack([np.ones(n), X_norm])
        best_A = A
        best_alpha = 1e-6
    
    # Final fit on all data with best parameters
    try:
        w = _ridge_fit(best_A, y_train, best_alpha)
    except Exception:
        # Ultimate fallback: least squares on linear features
        A = np.column_stack([np.ones(n), X_norm])
        w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        best_A = A
    
    final_A = best_A
    
    def predict(X):
        X_norm_test = _normalize(X)
        # Reconstruct features in the same way as best_A was built
        # We need to identify which feature type was used
        # Store the feature type info
        return _predict_with_features(X_norm_test, best_name, best_alpha, w, X_train, X_norm)
    
    # Actually, we need to store the feature generation parameters
    # Let's refactor: store the feature matrix builder
    feature_type = best_name
    
    def predict(X):
        X_norm_test = _normalize(X)
        n_test = X_norm_test.shape[0]
        
        if feature_type.startswith("poly"):
            deg = int(feature_type.split("deg")[1])
            A_test = _make_polynomial_features(X_norm_test, deg)
        elif feature_type.startswith("fourier"):
            nf = int(feature_type.split("_")[1])
            A_test = _make_fourier_features(X_norm_test, nf)
        elif feature_type.startswith("rbf"):
            # Parse parameters
            parts = feature_type.split("_")
            nc = int(parts[1][1:])  # c5 -> 5
            bw = float(parts[2][2:])  # bw0.3 -> 0.3
            if d == 1:
                centers = np.quantile(X_norm, np.linspace(0.1, 0.9, nc)).reshape(-1, 1)
            else:
                centers = []
                for dim in range(d):
                    q = np.quantile(X_norm[:, dim], np.linspace(0.1, 0.9, nc))
                    for qi in range(nc):
                        c = np.zeros(d)
                        c[dim] = q[qi]
                        centers.append(c)
                centers = np.array(centers[:nc * d])
                if len(centers) > nc:
                    idx = np.random.choice(len(centers), nc, replace=False)
                    centers = centers[idx]
            bandwidths = np.full(len(centers), bw)
            A_test = _make_rbf_features(X_norm_test, centers, bandwidths)
        elif feature_type.startswith("spline"):
            nk = int(feature_type.split("_")[1])
            A_test = _make_spline_features(X_norm_test, nk)
        else:  # linear
            A_test = np.column_stack([np.ones(n_test), X_norm_test])
        
        pred = A_test @ w
        # Ensure finite values
        pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred
    
    return predict
# EVOLVE-BLOCK-END

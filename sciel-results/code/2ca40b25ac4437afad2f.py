"""Symbolic regression via RBF-ridge with adaptive bandwidth search."""
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
DEADLINE = 330.0  # seconds


def _rbf_features(X, centers, gamma):
    """Compute RBF kernel matrix K(X, centers) with given gamma."""
    # X: (n, d), centers: (m, d)
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
    x2 = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    c2 = np.sum(centers ** 2, axis=1, keepdims=True).T  # (1, m)
    dist2 = x2 - 2.0 * (X @ centers.T) + c2  # (n, m)
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-gamma * dist2)


def _ridge_solve(K, y, lam):
    """Solve (K + lam*I) w = y."""
    n = K.shape[0]
    A = K + lam * np.eye(n)
    try:
        w = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, y, rcond=None)[0]
    return w


def _predict_rbf(X, centers, gamma, w):
    K = _rbf_features(X, centers, gamma)
    return K @ w


def _median_dist_sq(X):
    """Compute median squared distance between all pairs (sampled for efficiency)."""
    n = X.shape[0]
    if n <= 50:
        idx = np.random.default_rng(42).permutation(n)
        sample = X[idx[:min(50, n)]]
    else:
        sample = X
    m = sample.shape[0]
    # Compute pairwise distances for a subset to keep it fast
    if m > 40:
        rng = np.random.default_rng(123)
        idx = rng.permutation(m)
        sample = sample[idx[:40]]
        m = 40
    dists = np.array([np.sum((sample[i] - sample[j]) ** 2) for i in range(m) for j in range(i + 1, m)])
    return np.median(dists) if len(dists) > 0 else 1.0


def fit(X_train, y_train):
    global _START_TIME
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).ravel()
    n, d = X.shape
    
    # Time budget per equation: ~30 seconds (10 equations * 30s = 300s)
    eq_deadline = _START_TIME + 30.0
    
    # --- Step 1: Try polynomial regression (fast baseline) ---
    def poly_features(X, deg=3):
        cols = [np.ones(len(X))]
        # Linear
        for j in range(d):
            cols.append(X[:, j])
        # Interactions and powers up to deg
        for i in range(d):
            for j in range(i, d):
                for k in range(d):
                    # i+j+k <= deg and i<=j<=k
                    if i + j + k <= deg:
                        term = X[:, i] * X[:, j] * X[:, k]
                        cols.append(term)
        return np.column_stack(cols)
    
    best_poly_score = np.inf
    best_poly_w = None
    best_poly_deg = 0
    
    for deg in [2, 3, 4]:
        if time.time() > eq_deadline - 5:
            break
        try:
            F = poly_features(X, deg)
            # Ridge regularization to handle collinearity
            lam = 1e-4
            w, *_ = np.linalg.lstsq(F + lam * np.eye(F.shape[1]), y, rcond=None)
            pred = F @ w
            score = np.mean((pred - y) ** 2)
            if score < best_poly_score:
                best_poly_score = score
                best_poly_w = w
                best_poly_deg = deg
        except Exception:
            continue
    
    # --- Step 2: RBF regression with adaptive gamma search ---
    # Use training data as centers (or a subset if n is large)
    if n <= 200:
        centers = X
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=200, replace=False)
        centers = X[idx]
    
    m = centers.shape[0]
    K_train = _rbf_features(centers, centers, 1.0)  # placeholder, we'll recompute
    
    # Compute heuristic gamma
    med_d2 = _median_dist_sq(X)
    if med_d2 < 1e-12:
        med_d2 = 1.0
    gamma_heuristic = 1.0 / (2.0 * med_d2)
    
    # Gamma grid centered around heuristic
    gamma_grid = np.geomspace(gamma_heuristic * 0.01, gamma_heuristic * 100.0, 30)
    
    # Regularization grid
    lam_grid = np.logspace(-6, 2, 15)
    
    best_rbf_score = np.inf
    best_rbf_gamma = gamma_heuristic
    best_rbf_lam = 1e-4
    best_rbf_w = None
    
    # Precompute K for each gamma (expensive, so do it smartly)
    # Actually, for each gamma we need K_train. Let's iterate.
    
    for gamma in gamma_grid:
        if time.time() > eq_deadline - 10:
            break
        try:
            K = _rbf_features(centers, centers, gamma)
        except Exception:
            continue
        
        for lam in lam_grid:
            if time.time() > eq_deadline - 10:
                break
            try:
                w = _ridge_solve(K, y, lam)
                pred = K @ w
                score = np.mean((pred - y) ** 2)
                if score < best_rbf_score:
                    best_rbf_score = score
                    best_rbf_gamma = gamma
                    best_rbf_lam = lam
                    best_rbf_w = w
            except Exception:
                continue
    
    # --- Step 3: Choose best model ---
    # Compare training scores (since we don't have validation, use training fit as proxy)
    # Actually, for noiseless data, a good model should fit training well.
    # But overfitting is a concern. Use a simple heuristic: prefer RBF if it fits better,
    # but also consider that polynomial might be more robust.
    
    # Let's use a cross-validation-like approach: hold out 20% for validation
    rng_cv = np.random.default_rng(7)
    idx_cv = rng_cv.permutation(n)
    n_val = max(1, n // 5)
    val_idx = idx_cv[:n_val]
    tr_idx = idx_cv[n_val:]
    
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    
    # Evaluate polynomial on validation
    poly_val_score = np.inf
    if best_poly_w is not None:
        try:
            F_val = poly_features(X_val, best_poly_deg)
            pred_val = F_val @ best_poly_w
            poly_val_score = np.mean((pred_val - y_val) ** 2)
        except Exception:
            pass
    
    # Evaluate RBF on validation
    rbf_val_score = np.inf
    if best_rbf_w is not None:
        try:
            # Rebuild RBF with training subset centers
            if n_tr := len(tr_idx) <= 200:
                centers_cv = X_tr
            else:
                idx_cv2 = rng_cv.choice(len(tr_idx), size=200, replace=False)
                centers_cv = X_tr[idx_cv2]
            K_cv = _rbf_features(centers_cv, centers_cv, best_rbf_gamma)
            w_cv = _ridge_solve(K_cv, y_tr, best_rbf_lam)
            K_val = _rbf_features(X_val, centers_cv, best_rbf_gamma)
            pred_val = K_val @ w_cv
            rbf_val_score = np.mean((pred_val - y_val) ** 2)
        except Exception:
            pass
    
    # Choose model with better validation score
    use_rbf = (rbf_val_score < poly_val_score) and (rbf_val_score < np.inf)
    
    if use_rbf:
        # Refit RBF on full training set with best gamma and lam
        centers_full = X if n <= 200 else X[rng_cv.choice(n, size=200, replace=False)]
        K_full = _rbf_features(centers_full, centers_full, best_rbf_gamma)
        w_full = _ridge_solve(K_full, y, best_rbf_lam)
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            K_test = _rbf_features(X_test, centers_full, best_rbf_gamma)
            return K_test @ w_full
    else:
        # Use polynomial
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            F_test = poly_features(X_test, best_poly_deg)
            return F_test @ best_poly_w
    
    return predict

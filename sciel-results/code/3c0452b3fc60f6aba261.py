# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 300.0  # safety margin before 330s kill
    
    X = X_train.reshape(-1, 1).ravel()
    y = y_train.ravel()
    n = len(X)
    
    # Sort by X for interpolation-based methods
    idx = np.argsort(X)
    Xs = X[idx]
    ys = y[idx]
    
    # Cross-validation: 5-fold
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    
    def cv_score(model_factory):
        """Return mean CV MSE."""
        mses = []
        for fold in range(n_folds):
            if time.time() > deadline - 30:
                break
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            tr_idx = np.setdiff1d(indices, val_idx)
            
            X_tr = Xs[tr_idx]
            y_tr = ys[tr_idx]
            X_val = Xs[val_idx]
            y_val = ys[val_idx]
            
            try:
                pred = model_factory(X_tr, y_tr, X_val)
                if not np.all(np.isfinite(pred)):
                    return np.inf
                mse = np.mean((pred - y_val) ** 2)
                mses.append(mse)
            except Exception:
                return np.inf
        
        if not mses:
            return np.inf
        return np.mean(mses)
    
    # Candidate models
    candidates = []
    
    # 1. PCHIP interpolation
    def pchip_factory(X_tr, y_tr, X_val):
        interp = PchipInterpolator(X_tr, y_tr)
        return interp(X_val)
    candidates.append(("pchip", pchip_factory))
    
    # 2. RBF kernel regression with different lengthscales
    def make_rbf_factory(lgamma):
        def factory(X_tr, y_tr, X_val):
            # Build kernel matrix
            K = np.exp(-lgamma * (X_tr[:, None] - X_tr[None, :]) ** 2)
            K[np.diag_indices_from(K)] += 1e-6  # regularization
            w = np.linalg.solve(K, y_tr)
            K_val = np.exp(-lgamma * (X_val[:, None] - X_tr[None, :]) ** 2)
            return K_val @ w
        return factory
    
    for lgamma in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        candidates.append((f"rbf_{lgamma}", make_rbf_factory(lgamma)))
    
    # 3. Chebyshev polynomial fit (adaptive degree)
    def make_cheb_factory(max_degree):
        def factory(X_tr, y_tr, X_val):
            # Normalize to [-1, 1]
            x_min, x_max = X_tr.min(), X_tr.max()
            if x_max - x_min < 1e-10:
                x_min, x_max = -1, 1
            scale = (x_max - x_min) / 2
            offset = (x_max + x_min) / 2
            
            Xn_tr = (X_tr - offset) / scale
            Xn_val = (X_val - offset) / scale
            
            # Build Chebyshev basis
            basis_tr = np.column_stack([np.polynomial.chebyshev.chebvander(Xn_tr, d) for d in range(max_degree + 1)])
            # Actually chebvander gives all at once
            basis_tr = np.polynomial.chebyshev.chebvander(Xn_tr, max_degree)
            basis_val = np.polynomial.chebyshev.chebvander(Xn_val, max_degree)
            
            # Ridge regression for stability
            w, *_ = np.linalg.lstsq(basis_tr, y_tr, rcond=None)
            return basis_val @ w
        return factory
    
    for degree in [3, 5, 7, 10, 15, 20]:
        candidates.append((f"cheb_{degree}", make_cheb_factory(degree)))
    
    # 4. Spline interpolation
    from scipy.interpolate import CubicSpline
    def spline_factory(X_tr, y_tr, X_val):
        try:
            cs = CubicSpline(X_tr, y_tr)
            return cs(X_val)
        except Exception:
            return np.full_like(X_val, np.mean(y_tr))
    candidates.append(("spline", spline_factory))
    
    # 5. Local polynomial (moving window)
    def make_local_poly_factory(window_size, poly_degree):
        def factory(X_tr, y_tr, X_val):
            preds = np.zeros(len(X_val))
            for i, xv in enumerate(X_val):
                # Find nearest points
                dists = np.abs(X_tr - xv)
                nearest = np.argsort(dists)[:window_size]
                Xw = X_tr[nearest]
                yw = y_tr[nearest]
                if len(Xw) < poly_degree + 1:
                    preds[i] = np.mean(yw)
                else:
                    # Weighted local polynomial
                    weights = np.exp(-dists[nearest] ** 2 / (2 * np.median(dists[nearest]) ** 2 + 1e-10))
                    # Solve weighted least squares
                    X_design = np.column_stack([Xw ** d for d in range(poly_degree + 1)])
                    W = np.diag(weights)
                    try:
                        XtWX = X_design.T @ W @ X_design
                        XtWy = X_design.T @ W @ yw
                        w_local = np.linalg.solve(XtWX + 1e-8 * np.eye(poly_degree + 1), XtWy)
                        preds[i] = np.array([1.0] + [xv ** d for d in range(1, poly_degree + 1)]) @ w_local
                    except Exception:
                        preds[i] = np.average(yw, weights=weights)
            return preds
        return factory
    
    for ws in [10, 20, 40]:
        for pd in [2, 3]:
            candidates.append((f"local_{ws}_{pd}", make_local_poly_factory(ws, pd)))
    
    # Evaluate candidates with CV
    cv_scores = {}
    for name, factory in candidates:
        if time.time() > deadline - 60:
            break
        score = cv_score(factory)
        cv_scores[name] = score
    
    # Select best models (top 3 for ensemble)
    sorted_candidates = sorted(cv_scores.items(), key=lambda x: x[0] if x[0] > 0 else x[0])
    # Filter out inf scores
    valid_candidates = [(name, score) for name, score in cv_scores.items() if np.isfinite(score)]
    
    if not valid_candidates:
        # Fallback: mean prediction
        mean_y = np.mean(y)
        def predict_fallback(X):
            return np.full(len(X), mean_y)
        return predict_fallback
    
    # Sort by CV score
    valid_candidates.sort(key=lambda x: x[1])
    
    # Use top 3 for ensemble
    top_k = min(3, len(valid_candidates))
    top_models = valid_candidates[:top_k]
    
    # Build ensemble weights inversely proportional to CV score
    weights = np.array([1.0 / max(score, 1e-10) for _, score in top_models])
    weights = weights / weights.sum()
    
    # Build final ensemble predictor
    model_factories = []
    for name, _ in top_models:
        for cname, cfactory in candidates:
            if cname == name:
                model_factories.append(cfactory)
                break
    
    def predict(X_test):
        if time.time() > deadline:
            # Emergency: return mean
            return np.full(len(X_test), np.mean(y))
        
        X_test = np.asarray(X_test).reshape(-1, 1).ravel()
        
        # Sort for consistency
        # Use the training data sorted
        preds = np.zeros(len(X_test))
        for i, (factory, w) in enumerate(zip(model_factories, weights)):
            try:
                p = factory(Xs, ys, X_test)
                if np.all(np.isfinite(p)):
                    preds += w * p
            except Exception:
                pass
        
        # If no valid predictions, fallback to mean
        if not np.all(np.isfinite(preds)) or np.allclose(preds, 0):
            preds = np.full(len(X_test), np.mean(y))
        
        return preds
    
    return predict
# EVOLVE-BLOCK-END

"""Adaptive piecewise polynomial interpolation with RANSAC outlier rejection for noiseless 1D symbolic regression."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import make_interp_spline, PchipInterpolator

def _nrmse(y, yhat):
    """Compute normalized MSE."""
    resid = y - yhat
    mse = np.mean(resid * resid)
    vy = np.var(y)
    if vy <= 1e-12:
        return mse
    return mse / vy

def _ransac_piecewise_fit(x, y, max_iter=100, inlier_thresh=1e-3, min_inliers=20):
    """
    RANSAC-style robust fitting using piecewise cubic splines.
    Returns (inlier_mask, best_spline, best_score) or (None, None, inf) if failed.
    """
    n = len(x)
    if n < 10:
        return None, None, np.inf
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicates
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if len(unique_x) < len(x_sorted):
        y_unique = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])
        x_sorted = unique_x
        y_sorted = y_unique
        n = len(x_sorted)
    
    if n < 10:
        return None, None, np.inf
    
    best_inlier_mask = None
    best_spline = None
    best_score = np.inf
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        # Sample 3-5 random points as basis
        k = min(n, rng.randint(3, min(6, n)))
        idx = rng.choice(n, size=k, replace=False)
        idx = np.sort(idx)
        
        x_sample = x_sorted[idx]
        y_sample = y_sorted[idx]
        
        # Fit a PCHIP (monotone cubic) or make_interp_spline on the sample
        try:
            if k >= 4:
                spline = make_interp_spline(x_sample, y_sample, k=3)
            else:
                spline = PchipInterpolator(x_sample, y_sample)
            
            # Evaluate on all points
            y_pred = spline(x_sorted)
            if not np.all(np.isfinite(y_pred)):
                continue
            
            # Compute residuals
            resid = np.abs(y_sorted - y_pred)
            
            # Determine inliers
            # Use robust scale estimate
            med_resid = np.median(resid)
            if med_resid < 1e-10:
                # Very good fit, all points are inliers
                inlier_mask = np.ones(n, dtype=bool)
            else:
                threshold = max(inlier_thresh * med_resid, 1e-6)
                inlier_mask = resid <= threshold
            
            n_inliers = np.sum(inlier_mask)
            if n_inliers < min_inliers:
                continue
            
            # Refit on inliers only
            x_in = x_sorted[inlier_mask]
            y_in = y_sorted[inlier_mask]
            
            if len(x_in) >= 4:
                try:
                    spline_refined = make_interp_spline(x_in, y_in, k=3)
                    y_pred_refined = spline_refined(x_sorted)
                    if np.all(np.isfinite(y_pred_refined)):
                        score = _nrmse(y_sorted, y_pred_refined)
                        if score < best_score:
                            best_score = score
                            best_spline = spline_refined
                            best_inlier_mask = inlier_mask
                except Exception:
                    pass
            else:
                try:
                    spline_refined = PchipInterpolator(x_in, y_in)
                    y_pred_refined = spline_refined(x_sorted)
                    if np.all(np.isfinite(y_pred_refined)):
                        score = _nrmse(y_sorted, y_pred_refined)
                        if score < best_score:
                            best_score = score
                            best_spline = spline_refined
                            best_inlier_mask = inlier_mask
                except Exception:
                    pass
                    
        except Exception:
            continue
    
    return best_inlier_mask, best_spline, best_score

def _adaptive_knot_spline(x, y, max_knots=50, min_knots=10):
    """
    Adaptive piecewise polynomial interpolation with iterative knot refinement.
    Starts with a global spline and refines in regions of high residual.
    """
    n = len(x)
    if n < 10:
        return None, np.inf
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicates
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if len(unique_x) < len(x_sorted):
        y_unique = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])
        x_sorted = unique_x
        y_sorted = y_unique
        n = len(x_sorted)
    
    if n < 10:
        return None, np.inf
    
    # Start with a global cubic spline
    try:
        current_spline = make_interp_spline(x_sorted, y_sorted, k=3)
        y_pred = current_spline(x_sorted)
        score = _nrmse(y_sorted, y_pred)
        
        if score < 1e-8:
            return current_spline, score
        
        # Iteratively add knots at high-residual regions
        current_knots = set([x_sorted[0], x_sorted[-1]])
        
        for refinement_iter in range(30):
            if time.time() > 270:  # Safety deadline
                break
                
            y_pred = current_spline(x_sorted)
            if not np.all(np.isfinite(y_pred)):
                break
                
            resid = np.abs(y_sorted - y_pred)
            score = _nrmse(y_sorted, y_pred)
            
            if score < 1e-8:
                break
            
            # Find the point with maximum residual
            max_resid_idx = np.argmax(resid)
            new_knot = x_sorted[max_resid_idx]
            
            if new_knot in current_knots:
                # Try second-worst
                sorted_resid = np.argsort(resid)[::-1]
                found_new = False
                for idx in sorted_resid[:10]:
                    if x_sorted[idx] not in current_knots:
                        new_knot = x_sorted[idx]
                        found_new = True
                        break
                if not found_new:
                    break
            
            current_knots.add(new_knot)
            knots = np.array(sorted(current_knots))
            
            if len(knots) >= max_knots:
                break
            
            # Refit with new knots
            try:
                # Use Pchip for stability, or make_interp_spline with knots
                if len(knots) >= 4:
                    # Interpolate at knot locations
                    y_at_knots = current_spline(knots)
                    new_spline = make_interp_spline(knots, y_at_knots, k=3)
                    y_pred_new = new_spline(x_sorted)
                    if np.all(np.isfinite(y_pred_new)):
                        score_new = _nrmse(y_sorted, y_pred_new)
                        if score_new < score:
                            current_spline = new_spline
                            score = score_new
                        else:
                            # Revert
                            current_knots.discard(new_knot)
                else:
                    break
            except Exception:
                current_knots.discard(new_knot)
                break
        
        return current_spline, score
    except Exception:
        return None, np.inf

def _build_simple_features(X, max_p=8):
    """Build a moderate feature matrix as fallback."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones(n, dtype=np.float64)]
    
    for j in range(d):
        x = X[:, j].astype(np.float64)
        ax = np.abs(x)
        
        for p in range(1, max_p + 1):
            cols.append(x ** p)
        
        cols.append(ax)
        cols.append(np.sqrt(ax))
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        
        xc = np.clip(x, -50, 50)
        cols.append(np.exp(xc))
        cols.append(np.exp(-xc))
        
        cols.append(np.where(x > 1e-9, np.log(x), 0.0))
        cols.append(np.log1p(ax))
    
    return np.column_stack(cols)

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 285.0
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    y_mean = float(np.mean(y_train))
    
    best_predict = None
    best_score = np.inf
    
    if d == 1:
        x_train = X_train[:, 0]
        
        # Method 1: RANSAC piecewise spline
        inlier_mask, spline_ransac, score_ransac = _ransac_piecewise_fit(
            x_train, y_train, max_iter=150, inlier_thresh=1e-3, min_inliers=15
        )
        if spline_ransac is not None and np.isfinite(score_ransac):
            best_score = score_ransac
            spline_obj = spline_ransac
            x_min = float(np.min(x_train))
            x_max = float(np.max(x_train))
            
            def ransac_predict(X, spline=spline_obj, y_mean=y_mean, xmin=x_min, xmax=x_max):
                x_test = X[:, 0]
                # Clip to training range for extrapolation safety
                x_clip = np.clip(x_test, xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin))
                y_pred = spline(x_clip)
                return np.where(np.isfinite(y_pred), y_pred, y_mean).astype(np.float64)
            
            best_predict = ransac_predict
        
        # Method 2: Adaptive knot spline
        if best_score > 1e-6 and time.time() < DEADLINE - 20:
            spline_adapt, score_adapt = _adaptive_knot_spline(x_train, y_train)
            if spline_adapt is not None and np.isfinite(score_adapt) and score_adapt < best_score:
                best_score = score_adapt
                spline_obj = spline_adapt
                x_min = float(np.min(x_train))
                x_max = float(np.max(x_train))
                
                def adapt_predict(X, spline=spline_obj, y_mean=y_mean, xmin=x_min, xmax=x_max):
                    x_test = X[:, 0]
                    x_clip = np.clip(x_test, xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin))
                    y_pred = spline(x_clip)
                    return np.where(np.isfinite(y_pred), y_pred, y_mean).astype(np.float64)
                
                best_predict = adapt_predict
        
        # Method 3: Simple global spline as fallback
        if best_score > 1e-4 and time.time() < DEADLINE - 10:
            try:
                sort_idx = np.argsort(x_train)
                x_sorted = x_train[sort_idx]
                y_sorted = y_train[sort_idx]
                unique_x, inverse = np.unique(x_sorted, return_inverse=True)
                if len(unique_x) < len(x_sorted):
                    y_unique = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])
                else:
                    unique_x = x_sorted
                    y_unique = y_sorted
                
                spline_global = make_interp_spline(unique_x, y_unique, k=3)
                yhat = spline_global(x_train)
                if np.all(np.isfinite(yhat)):
                    score_global = _nrmse(y_train, yhat)
                    if score_global < best_score:
                        best_score = score_global
                        spline_obj = spline_global
                        x_min = float(np.min(x_train))
                        x_max = float(np.max(x_train))
                        
                        def global_predict(X, spline=spline_obj, y_mean=y_mean, xmin=x_min, xmax=x_max):
                            x_test = X[:, 0]
                            x_clip = np.clip(x_test, xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin))
                            y_pred = spline(x_clip)
                            return np.where(np.isfinite(y_pred), y_pred, y_mean).astype(np.float64)
                        
                        best_predict = global_predict
            except Exception:
                pass
    
    # Fallback: simple feature regression
    if best_predict is None or best_score > 1e-3:
        try:
            Phi = _build_simple_features(X_train, max_p=8)
            # Remove non-finite columns
            finite_cols = np.all(np.isfinite(Phi), axis=0)
            Phi_clean = Phi[:, finite_cols]
            
            if Phi_clean.shape[1] > 0:
                # Use ridge with a reasonable lambda
                lam = 1e-4
                A = Phi_clean.T @ Phi_clean + lam * np.eye(Phi_clean.shape[1])
                b = Phi_clean.T @ y_train
                try:
                    w = np.linalg.solve(A, b)
                    yhat = Phi_clean @ w
                    if np.all(np.isfinite(yhat)):
                        score = _nrmse(y_train, yhat)
                        if score < best_score:
                            w_final = np.where(np.isfinite(w), w, 0.0)
                            
                            def ridge_predict(X, w=w_final, y_mean=y_mean):
                                P = _build_simple_features(X, max_p=8)
                                P_clean = P[:, finite_cols]
                                yhat = P_clean @ w
                                return np.where(np.isfinite(yhat), yhat, y_mean).astype(np.float64)
                            
                            best_predict = ridge_predict
                            best_score = score
                except np.linalg.LinAlgError:
                    pass
        except Exception:
            pass
    
    # Final fallback: mean
    if best_predict is None:
        return lambda X: np.full(X.shape[0], y_mean, dtype=np.float64)
    
    return best_predict

# EVOLVE-BLOCK-END

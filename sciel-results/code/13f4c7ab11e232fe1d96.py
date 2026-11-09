import numpy as np
import time
from scipy.interpolate import UnivariateSpline

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 280
    
    X = X_train
    y = y_train
    n, d = X.shape
    x = X[:, 0]
    
    # Sort by x for spline
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Cross-validation for Smoothing Spline
    # Use 5-fold CV
    k = 5
    fold_size = n // k
    # If n is small, use leave-one-out? n=200, 5-fold is fine.
    
    # We need to be careful with the time.
    # Fitting a spline is fast.
    
    # Candidate smoothing factors
    # s is the sum of squared residuals allowed.
    # s=0 is interpolation.
    # s=inf is a straight line (or constant?).
    # Let's try relative to total variance.
    var_y = np.var(y)
    if var_y == 0:
        return lambda X: np.zeros(len(X))
    
    # Try a log-spaced range of s
    # s values: 0, 1e-4*var, 1e-3*var, 1e-2*var, 1e-1*var, 1*var, 10*var
    s_candidates = [0, 1e-4*var_y, 1e-3*var_y, 1e-2*var_y, 1e-1*var_y, 1.0*var_y, 10*var_y]
    
    best_s = 0
    best_cv_error = np.inf
    
    # Simple CV: Split into 5 folds
    # For each candidate s, compute mean CV error
    indices = np.arange(n)
    # Shuffle? The data might be sorted. Random shuffling is better for CV.
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    
    for s in s_candidates:
        cv_errors = []
        for i in range(k):
            val_idx = indices[i*fold_size:(i+1)*fold_size]
            tr_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
            
            x_tr = x_sorted[tr_idx] # Wait, x_sorted is sorted. tr_idx refers to original indices?
            # My x_sorted is sorted by x. The indices `tr_idx` are into the original array.
            # I need to map back.
            # Let's just use the original x, y for CV, and sort inside the fit.
            
            x_cv_tr = x[tr_idx]
            y_cv_tr = y[tr_idx]
            x_cv_val = x[val_idx]
            y_cv_val = y[val_idx]
            
            # Fit spline on train
            try:
                sp = UnivariateSpline(x_cv_tr, y_cv_tr, k=3, s=s)
                y_pred = sp(x_cv_val)
                err = np.mean((y_pred - y_cv_val)**2)
                cv_errors.append(err)
            except Exception:
                cv_errors.append(np.inf)
                
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_error:
            best_cv_error = mean_cv_error
            best_s = s
            
    # Final fit with best_s
    # Use all data
    try:
        final_spline = UnivariateSpline(x, y, k=3, s=best_s)
    except Exception:
        # Fallback to OLS on rich basis if spline fails
        def make_features_rich(x):
            p1 = np.ones_like(x)
            p2 = x
            p3 = x**2
            p4 = x**3
            p5 = x**4
            p6 = x**5
            p7 = x**6
            t1 = np.sin(x)
            t2 = np.cos(x)
            t3 = np.sin(2*x)
            t4 = np.cos(2*x)
            t5 = np.sin(3*x)
            t6 = np.cos(3*x)
            t7 = np.sin(4*x)
            t8 = np.cos(4*x)
            t9 = np.sin(5*x)
            t10 = np.cos(5*x)
            e1 = np.exp(x)
            e2 = np.exp(-x)
            return np.column_stack([p1, p2, p3, p4, p5, p6, p7, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, e1, e2])
            
        F_train = make_features_rich(x)
        w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
        
        def predict(X):
            x = X[:, 0]
            F = make_features_rich(x)
            return F @ w
        return predict

    def predict(X):
        x = X[:, 0]
        return final_spline(x)
        
    return predict

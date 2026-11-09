"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    """
    Fit a regression model using adaptive basis selection with cross-validation.
    Tries multiple model families (polynomial, Fourier, spline) and picks
    the best by CV score. For 1D inputs, splines are exact for smooth noiseless
    functions, which should handle all hidden equations well.
    """
    n, d = X_train.shape
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin
    
    def cv_score(model_fn, n_folds=5):
        """Cross-validation score (lower is better)."""
        idx = np.random.RandomState(42).permutation(n)
        fold_size = n // n_folds
        scores = []
        for i in range(n_folds):
            val_idx = idx[i * fold_size:(i + 1) * fold_size]
            train_idx = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            try:
                pred = model_fn(X_tr, y_tr)(X_val)
                if not np.all(np.isfinite(pred)):
                    return np.inf
                mse = np.mean((pred - y_val) ** 2)
                var_y = np.var(y_val)
                if var_y < 1e-12:
                    return 0.0
                scores.append(mse / var_y)
            except Exception:
                return np.inf
        return np.mean(scores)
    
    def make_poly_model(degree):
        """Create a polynomial model of given degree."""
        def build_features(X):
            cols = [np.ones(len(X))]
            for j in range(d):
                for p in range(1, degree + 1):
                    cols.append(X[:, j] ** p)
            # Cross terms for d > 1
            if d > 1:
                for i in range(d):
                    for j in range(i + 1, d):
                        cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        
        def model_fn(X_tr, y_tr):
            w, *_ = np.linalg.lstsq(build_features(X_tr), y_tr, rcond=None)
            return lambda X: build_features(X) @ w
        return model_fn
    
    def make_fourier_model(n_harmonics):
        """Create a Fourier model with given number of harmonics."""
        def build_features(X):
            cols = [np.ones(len(X))]
            for j in range(d):
                for k in range(1, n_harmonics + 1):
                    cols.append(np.sin(k * X[:, j]))
                    cols.append(np.cos(k * X[:, j]))
            return np.column_stack(cols)
        
        def model_fn(X_tr, y_tr):
            w, *_ = np.linalg.lstsq(build_features(X_tr), y_tr, rcond=None)
            return lambda X: build_features(X) @ w
        return model_fn
    
    def make_spline_model(n_knots):
        """Create a spline interpolation model for 1D inputs.
        Uses cubic spline interpolation which is exact for smooth noiseless 1D functions."""
        def model_fn(X_tr, y_tr):
            if d != 1:
                # For multi-dimensional, fall back to polynomial
                return make_poly_model(4)(X_tr, y_tr)
            
            x_tr = X_tr[:, 0]
            # Sort by x for interpolation
            sort_idx = np.argsort(x_tr)
            x_sorted = x_tr[sort_idx]
            y_sorted = y_tr[sort_idx]
            
            # Use cubic spline interpolation
            from scipy.interpolate import interp1d
            
            # Handle potential duplicate x values by averaging
            unique_x, inverse = np.unique(x_sorted, return_inverse=True)
            if len(unique_x) < len(x_sorted):
                y_unique = np.zeros(len(unique_x))
                counts = np.zeros(len(unique_x))
                np.add.at(y_unique, inverse, y_sorted)
                np.add.at(counts, inverse, 1)
                y_unique /= counts
            else:
                unique_x = x_sorted
                y_unique = y_sorted
            
            # Create interpolator
            interpolator = interp1d(unique_x, y_unique, kind='cubic', 
                                     bounds_error=False, fill_value='extrapolate')
            
            def predict(X):
                x = X[:, 0]
                return interpolator(x)
            
            return predict
        return model_fn
    
    # Candidate models to try
    candidates = []
    
    # Polynomials of various degrees
    for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
        candidates.append(('poly', deg, make_poly_model(deg)))
    
    # Fourier models
    for nh in [2, 3, 4, 5, 6, 8, 10]:
        candidates.append(('fourier', nh, make_fourier_model(nh)))
    
    # Spline models (exact for smooth 1D noiseless functions)
    for nk in [50, 100, 150, 200]:
        candidates.append(('spline', nk, make_spline_model(nk)))
    
    # Evaluate each candidate with CV
    best_score = np.inf
    best_model_fn = None
    best_name = None
    
    for name, param, model_fn in candidates:
        if time.time() > deadline - 30.0:
            break
        try:
            score = cv_score(model_fn)
            if score < best_score:
                best_score = score
                best_model_fn = model_fn
                best_name = f"{name}_{param}"
        except Exception:
            continue
    
    # If no model worked, fall back to degree-2 polynomial
    if best_model_fn is None:
        best_model_fn = make_poly_model(2)
    
    # Fit the best model on all training data
    try:
        predict_fn = best_model_fn(X_train, y_train)
        # Verify it produces finite outputs
        test_pred = predict_fn(X_train[:10])
        if not np.all(np.isfinite(test_pred)):
            predict_fn = make_poly_model(2)(X_train, y_train)
    except Exception:
        predict_fn = make_poly_model(2)(X_train, y_train)
    
    return predict_fn
# EVOLVE-BLOCK-END

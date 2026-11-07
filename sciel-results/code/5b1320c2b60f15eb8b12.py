"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


def fit(X_train, y_train):
    """
    Fit a regression model to the training data.
    Uses a combination of:
    1. Polynomial regression (degree up to 10)
    2. PCHIP interpolation for 1D data
    3. Gaussian Process-like smoothing
    4. Local polynomial regression (LOESS-like)
    
    Returns the best model based on cross-validation on training data.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n_samples, n_features = X.shape
    
    # Check if it's 1D data
    is_1d = (n_features == 1)
    
    # Sort by input for 1D data
    if is_1d:
        sort_idx = np.argsort(X[:, 0])
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
    else:
        X_sorted = X
        y_sorted = y
    
    # Candidate models
    models = []
    model_names = []
    
    # 1. Polynomial regression of various degrees
    max_degree = 12
    for degree in range(1, max_degree + 1):
        if time.time() > deadline - 10:
            break
        
        # Create polynomial features
        if is_1d:
            x = X_sorted[:, 0]
            V = np.vander(x, N=degree + 1, increasing=True)
        else:
            # For multi-dim, use tensor product (but our data is 1D)
            x = X_sorted[:, 0]
            V = np.vander(x, N=degree + 1, increasing=True)
        
        try:
            # Regularized least squares
            alpha = 1e-8 * degree
            A = V.T @ V + alpha * np.eye(V.shape[1])
            b = V.T @ y_sorted
            w = np.linalg.solve(A, b)
            
            def make_poly_predict(w, degree):
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    x = X_pred[:, 0]
                    V_pred = np.vander(x, N=degree + 1, increasing=True)
                    return V_pred @ w
                return predict
            
            models.append(make_poly_predict(w, degree))
            model_names.append(f"poly_{degree}")
        except Exception:
            continue
    
    # 2. PCHIP interpolation (1D only)
    if is_1d:
        if time.time() > deadline - 10:
            pass
        else:
            try:
                x_unique, y_unique = [], []
                x_sorted = X_sorted[:, 0]
                # Handle duplicate x values by averaging
                for i in range(len(x_sorted)):
                    if i == 0 or x_sorted[i] != x_sorted[i - 1]:
                        x_unique.append(x_sorted[i])
                        y_unique.append(y_sorted[i])
                    else:
                        y_unique[-1] += y_sorted[i]
                        # Will need to average later
                # More robust: use np.unique
                x_unique, unique_indices = np.unique(x_sorted, return_index=True)
                y_unique = y_sorted[unique_indices]
                
                # Check for strict monotonicity in x (should be true after unique)
                if len(x_unique) >= 2 and np.all(np.diff(x_unique) > 0):
                    pchip = PchipInterpolator(x_unique, y_unique)
                    
                    def make_pchip_predict(pchip_interp):
                        def predict(X_pred):
                            X_pred = np.asarray(X_pred, dtype=np.float64)
                            if X_pred.ndim == 1:
                                X_pred = X_pred.reshape(-1, 1)
                            x = X_pred[:, 0]
                            return pchip_interp(x)
                        return predict
                    
                    models.append(make_pchip_predict(pchip))
                    model_names.append("pchip")
            except Exception:
                continue
    
    # 3. Cubic spline interpolation (1D only)
    if is_1d:
        if time.time() > deadline - 10:
            pass
        else:
            try:
                from scipy.interpolate import CubicSpline
                x_unique, unique_indices = np.unique(X_sorted[:, 0], return_index=True)
                y_unique = y_sorted[unique_indices]
                
                if len(x_unique) >= 4 and np.all(np.diff(x_unique) > 0):
                    cs = CubicSpline(x_unique, y_unique, bc_type="natural")
                    
                    def make_cs_predict(cs_interp):
                        def predict(X_pred):
                            X_pred = np.asarray(X_pred, dtype=np.float64)
                            if X_pred.ndim == 1:
                                X_pred = X_pred.reshape(-1, 1)
                            x = X_pred[:, 0]
                            return cs_interp(x)
                        return predict
                    
                    models.append(make_cs_predict(cs))
                    model_names.append("cubic_spline")
            except Exception:
                continue
    
    # 4. Local polynomial regression (LOESS-like)
    if is_1d:
        if time.time() > deadline - 5:
            pass
        else:
            try:
                x_sorted = X_sorted[:, 0]
                y_sorted = y_sorted
                
                # Try different bandwidths
                for frac in [0.3, 0.5, 0.7, 1.0]:
                    if time.time() > deadline - 5:
                        break
                    
                    def make_loess_predict(x_train, y_train, frac, degree=2):
                        def predict(X_pred):
                            X_pred = np.asarray(X_pred, dtype=np.float64)
                            if X_pred.ndim == 1:
                                X_pred = X_pred.reshape(-1, 1)
                            x_pred = X_pred[:, 0]
                            n = len(x_train)
                            k = max(degree + 1, int(frac * n))
                            preds = np.zeros(len(x_pred))
                            for i, xp in enumerate(x_pred):
                                # Find k nearest neighbors
                                dists = np.abs(x_train - xp)
                                idx = np.argpartition(dists, k)[:k]
                                # Weights based on distance
                                w = 1.0 / (dists[idx] + 1e-10)
                                w = w / w.sum()
                                # Weighted polynomial fit
                                x_local = x_train[idx]
                                y_local = y_train[idx]
                                V = np.vander(x_local, N=degree + 1, increasing=True)
                                W = np.diag(w)
                                try:
                                    A = V.T @ W @ V
                                    b = V.T @ W @ y_local
                                    coef = np.linalg.solve(A, b)
                                    preds[i] = np.vander([xp], N=degree + 1, increasing=True) @ coef
                                except Exception:
                                    preds[i] = np.average(y_local, weights=w)
                            return preds
                        return predict
                    
                    models.append(make_loess_predict(x_sorted, y_sorted, frac, 2))
                    model_names.append(f"loess_{frac}")
            except Exception:
                continue
    
    # 5. Sine/cosine basis (Fourier-like)
    if is_1d:
        if time.time() > deadline - 10:
            pass
        else:
            try:
                x_sorted = X_sorted[:, 0]
                # Estimate frequency range
                x_range = x_sorted.max() - x_sorted.min()
                if x_range > 1e-10:
                    # Try different number of harmonics
                    for n_harm in [3, 5, 8, 12]:
                        if time.time() > deadline - 5:
                            break
                        
                        # Build Fourier features
                        freqs = np.arange(1, n_harm + 1)
                        V = np.column_stack([
                            np.ones_like(x_sorted),
                            *[np.sin(2 * np.pi * f * x_sorted / x_range) for f in freqs],
                            *[np.cos(2 * np.pi * f * x_sorted / x_range) for f in freqs]
                        ])
                        
                        alpha = 1e-6
                        A = V.T @ V + alpha * np.eye(V.shape[1])
                        b = V.T @ y_sorted
                        w = np.linalg.solve(A, b)
                        
                        def make_fourier_predict(w, n_harm, x_range):
                            def predict(X_pred):
                                X_pred = np.asarray(X_pred, dtype=np.float64)
                                if X_pred.ndim == 1:
                                    X_pred = X_pred.reshape(-1, 1)
                                x = X_pred[:, 0]
                                freqs = np.arange(1, n_harm + 1)
                                V_pred = np.column_stack([
                                    np.ones_like(x),
                                    *[np.sin(2 * np.pi * f * x / x_range) for f in freqs],
                                    *[np.cos(2 * np.pi * f * x / x_range) for f in freqs]
                                ])
                                return V_pred @ w
                            return predict
                        
                        models.append(make_fourier_predict(w, n_harm, x_range))
                        model_names.append(f"fourier_{n_harm}")
            except Exception:
                continue
    
    # If no models were created, fall back to simple mean
    if not models:
        y_mean = np.mean(y)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.full(X.shape[0], y_mean)
        return predict
    
    # Evaluate models using leave-one-out or k-fold CV on training data
    # Use 5-fold CV for efficiency
    n_folds = 5
    fold_sizes = np.array_split(np.arange(n_samples), n_folds)
    
    best_model = None
    best_cv_score = np.inf
    
    for model_idx, model in enumerate(models):
        if time.time() > deadline - 5:
            break
        
        cv_errors = []
        for fold in range(n_folds):
            val_idx = fold_sizes[fold]
            train_idx = np.concatenate([fold_sizes[i] for i in range(n_folds) if i != fold])
            
            # Refit on training subset
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            try:
                # Re-fit the model on the fold's training data
                if is_1d:
                    sort_idx = np.argsort(X_train_fold[:, 0])
                    X_fold_sorted = X_train_fold[sort_idx]
                    y_fold_sorted = y_train_fold[sort_idx]
                    x_fold = X_fold_sorted[:, 0]
                else:
                    X_fold_sorted = X_train_fold
                    y_fold_sorted = y_train_fold
                    x_fold = X_train_fold[:, 0]
                
                # Determine model type and refit
                name = model_names[model_idx]
                
                if name.startswith("poly_"):
                    degree = int(name.split("_")[1])
                    V = np.vander(x_fold, N=degree + 1, increasing=True)
                    alpha = 1e-8 * degree
                    A = V.T @ V + alpha * np.eye(V.shape[1])
                    b = V.T @ y_fold_sorted
                    w_fold = np.linalg.solve(A, b)
                    V_val = np.vander(X_val_fold[:, 0], N=degree + 1, increasing=True)
                    preds = V_val @ w_fold
                    
                elif name == "pchip":
                    x_unique, unique_indices = np.unique(x_fold, return_index=True)
                    y_unique = y_fold_sorted[unique_indices]
                    if len(x_unique) >= 2 and np.all(np.diff(x_unique) > 0):
                        pchip_fold = PchipInterpolator(x_unique, y_unique)
                        preds = pchip_fold(X_val_fold[:, 0])
                    else:
                        preds = np.full(len(X_val_fold), np.mean(y_fold_sorted))
                        
                elif name == "cubic_spline":
                    from scipy.interpolate import CubicSpline
                    x_unique, unique_indices = np.unique(x_fold, return_index=True)
                    y_unique = y_fold_sorted[unique_indices]
                    if len(x_unique) >= 4 and np.all(np.diff(x_unique) > 0):
                        cs_fold = CubicSpline(x_unique, y_unique, bc_type="natural")
                        preds = cs_fold(X_val_fold[:, 0])
                    else:
                        preds = np.full(len(X_val_fold), np.mean(y_fold_sorted))
                        
                elif name.startswith("loess_"):
                    frac = float(name.split("_")[1])
                    degree = 2
                    preds = np.zeros(len(X_val_fold))
                    for i, xp in enumerate(X_val_fold[:, 0]):
                        dists = np.abs(x_fold - xp)
                        k = max(degree + 1, int(frac * len(x_fold)))
                        idx = np.argpartition(dists, k)[:k]
                        w = 1.0 / (dists[idx] + 1e-10)
                        w = w / w.sum()
                        x_local = x_fold[idx]
                        y_local = y_fold_sorted[idx]
                        V_local = np.vander(x_local, N=degree + 1, increasing=True)
                        W = np.diag(w)
                        try:
                            A = V_local.T @ W @ V_local
                            b = V_local.T @ W @ y_local
                            coef = np.linalg.solve(A, b)
                            preds[i] = np.vander([xp], N=degree + 1, increasing=True) @ coef
                        except Exception:
                            preds[i] = np.average(y_local, weights=w)
                            
                elif name.startswith("fourier_"):
                    n_harm = int(name.split("_")[1])
                    x_range = x_fold.max() - x_fold.min()
                    if x_range < 1e-10:
                        preds = np.full(len(X_val_fold), np.mean(y_fold_sorted))
                    else:
                        freqs = np.arange(1, n_harm + 1)
                        V = np.column_stack([
                            np.ones_like(x_fold),
                            *[np.sin(2 * np.pi * f * x_fold / x_range) for f in freqs],
                            *[np.cos(2 * np.pi * f * x_fold / x_range) for f in freqs]
                        ])
                        alpha = 1e-6
                        A = V.T @ V + alpha * np.eye(V.shape[1])
                        b = V.T @ y_fold_sorted
                        w_fold = np.linalg.solve(A, b)
                        V_val = np.column_stack([
                            np.ones_like(X_val_fold[:, 0]),
                            *[np.sin(2 * np.pi * f * X_val_fold[:, 0] / x_range) for f in freqs],
                            *[np.cos(2 * np.pi * f * X_val_fold[:, 0] / x_range) for f in freqs]
                        ])
                        preds = V_val @ w_fold
                        
                else:
                    continue
                
                mse = np.mean((preds - y_val_fold) ** 2)
                cv_errors.append(mse)
                
            except Exception:
                cv_errors.append(np.inf)
                break
        
        if len(cv_errors) == n_folds:
            avg_mse = np.mean(cv_errors)
            if avg_mse < best_cv_score:
                best_cv_score = avg_mse
                best_model = model
                best_model_name = model_names[model_idx]
    
    # If CV failed, use the first model
    if best_model is None:
        best_model = models[0]
    
    return best_model

# EVOLVE-BLOCK-END

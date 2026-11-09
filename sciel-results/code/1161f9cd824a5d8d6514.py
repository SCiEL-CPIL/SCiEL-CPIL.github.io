"""Symbolic regression via ensemble of kernel ridge regression + polynomial features."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

def fit(X_train, y_train):
    t_start = time.time()
    deadline = 320.0  # seconds, safety margin before 330s kill
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize features for kernel computations (but remember original for prediction)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    def time_left():
        return deadline - (time.time() - t_start)
    
    # Candidate models to try
    best_predict = None
    best_train_rmse = np.inf
    
    def make_rbf_predict(alpha, K_train, X_norm):
        """Create a predict function for RBF kernel ridge regression."""
        def predict(X):
            Xn = (np.asarray(X, dtype=np.float64) - X_mean) / X_std
            K_test = np.exp(-alpha * ((Xn[:, np.newaxis, :] - X_norm[np.newaxis, :, :])**2).sum(axis=2))
            y_pred = K_test @ (np.linalg.solve(K_train + alpha * np.eye(n), y_norm))
            return y_pred * y_std + y_mean
        return predict
    
    def make_poly_predict(w, degree):
        """Create a predict function for polynomial features."""
        def make_features(X):
            Xn = (np.asarray(X, dtype=np.float64) - X_mean) / X_std
            cols = [np.ones(len(Xn))]
            # Linear terms
            for j in range(d):
                cols.append(Xn[:, j])
            # Quadratic terms
            if degree >= 2:
                for i in range(d):
                    for j in range(i, d):
                        cols.append(Xn[:, i] * Xn[:, j])
            # Cubic terms
            if degree >= 3:
                for i in range(d):
                    for j in range(i, d):
                        for k in range(j, d):
                            cols.append(Xn[:, i] * Xn[:, j] * Xn[:, k])
            return np.column_stack(cols)
        
        def predict(X):
            Xn = (np.asarray(X, dtype=np.float64) - X_mean) / X_std
            cols = [np.ones(len(Xn))]
            for j in range(d):
                cols.append(Xn[:, j])
            if degree >= 2:
                for i in range(d):
                    for j in range(i, d):
                        cols.append(Xn[:, i] * Xn[:, j])
            if degree >= 3:
                for i in range(d):
                    for j in range(i, d):
                        for k in range(j, d):
                            cols.append(Xn[:, i] * Xn[:, j] * Xn[:, k])
            F = np.column_stack(cols)
            return (F @ w) * y_std + y_mean
        return predict
    
    # Model 1: Polynomial degree 2 with ridge
    try:
        if time_left() > 2.0:
            F_train = []
            F_train.append(np.ones(n))
            for j in range(d):
                F_train.append(X_norm[:, j])
            for i in range(d):
                for j in range(i, d):
                    F_train.append(X_norm[:, i] * X_norm[:, j])
            F_train = np.column_stack(F_train)
            
            # Try several regularization strengths
            for lam in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
                w = np.linalg.solve(F_train.T @ F_train + lam * np.eye(F_train.shape[1]), F_train.T @ y_norm)
                pred_train = F_train @ w
                rmse = np.sqrt(np.mean((pred_train - y_norm)**2))
                if rmse < best_train_rmse:
                    best_train_rmse = rmse
                    best_predict = make_poly_predict(w, 2)
    except:
        pass
    
    # Model 2: Polynomial degree 3 with ridge
    try:
        if time_left() > 2.0 and d <= 3:
            F_train = []
            F_train.append(np.ones(n))
            for j in range(d):
                F_train.append(X_norm[:, j])
            for i in range(d):
                for j in range(i, d):
                    F_train.append(X_norm[:, i] * X_norm[:, j])
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        F_train.append(X_norm[:, i] * X_norm[:, j] * X_norm[:, k])
            F_train = np.column_stack(F_train)
            
            for lam in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
                w = np.linalg.solve(F_train.T @ F_train + lam * np.eye(F_train.shape[1]), F_train.T @ y_norm)
                pred_train = F_train @ w
                rmse = np.sqrt(np.mean((pred_train - y_norm)**2))
                if rmse < best_train_rmse:
                    best_train_rmse = rmse
                    best_predict = make_poly_predict(w, 3)
    except:
        pass
    
    # Model 3: RBF kernel ridge regression with various length scales
    try:
        if time_left() > 5.0:
            # Compute pairwise distances
            diff = X_norm[:, np.newaxis, :] - X_norm[np.newaxis, :, :]
            sq_dist = (diff**2).sum(axis=2)
            
            # Try various alpha values (inverse length scale squared)
            alpha_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
            
            for alpha in alpha_candidates:
                if time_left() < 1.0:
                    break
                K = np.exp(-alpha * sq_dist)
                # Add regularization
                reg = 1e-6
                try:
                    K_inv_y = np.linalg.solve(K + reg * np.eye(n), y_norm)
                    pred_train = K @ K_inv_y
                    rmse = np.sqrt(np.mean((pred_train - y_norm)**2))
                    if rmse < best_train_rmse:
                        best_train_rmse = rmse
                        # Store the solution for efficient prediction
                        K_train = K.copy()
                        K_inv_y_copy = K_inv_y.copy()
                        alpha_copy = alpha
                        best_predict = (lambda K_t, Kiy, a: lambda X: (
                            np.exp(-a * (((np.asarray(X, dtype=np.float64) - X_mean) / X_std)[:, np.newaxis, :] - X_norm[np.newaxis, :, :])**2).sum(axis=2)
                        ) @ Kiy * y_std + y_mean)(K_train, K_inv_y_copy, alpha_copy)
                except np.linalg.LinAlgError:
                    continue
    except:
        pass
    
    # Model 4: Composite kernel (RBF + polynomial) 
    try:
        if time_left() > 10.0:
            # Use a moderate RBF + linear kernel
            for alpha in [0.1, 0.5, 1.0, 2.0]:
                if time_left() < 2.0:
                    break
                K_rbf = np.exp(-alpha * sq_dist)
                K_lin = X_norm @ X_norm.T + 1.0  # linear kernel with bias
                K = K_rbf + K_lin
                reg = 1e-6
                try:
                    K_inv_y = np.linalg.solve(K + reg * np.eye(n), y_norm)
                    pred_train = K @ K_inv_y
                    rmse = np.sqrt(np.mean((pred_train - y_norm)**2))
                    if rmse < best_train_rmse:
                        best_train_rmse = rmse
                        K_train = K.copy()
                        K_inv_y_copy = K_inv_y.copy()
                        alpha_copy = alpha
                        def make_composite_predict(K_t, Kiy, a):
                            def predict(X):
                                Xn = (np.asarray(X, dtype=np.float64) - X_mean) / X_std
                                K_rbf_test = np.exp(-a * ((Xn[:, np.newaxis, :] - X_norm[np.newaxis, :, :])**2).sum(axis=2))
                                K_lin_test = Xn @ X_norm.T + 1.0
                                K_test = K_rbf_test + K_lin_test
                                return (K_test @ Kiy) * y_std + y_mean
                            return predict
                        best_predict = make_composite_predict(K_train, K_inv_y_copy, alpha_copy)
                except np.linalg.LinAlgError:
                    continue
    except:
        pass
    
    # Model 5: Neural network-like approach using random Fourier features
    try:
        if time_left() > 5.0:
            # Random Fourier features for RBF approximation
            n_features = 200
            rng = np.random.RandomState(42)
            # Scale based on data range
            freq_scale = 1.0 / X_std
            Omega = rng.randn(n, n_features) * freq_scale
            b = rng.rand(n) * 2 * np.pi
            
            F_rff = np.column_stack([
                np.sqrt(2.0 / n) * np.cos(X_norm @ Omega + b),
                np.sqrt(2.0 / n) * np.sin(X_norm @ Omega + b)
            ])
            
            for lam in [1e-6, 1e-4, 1e-3, 0.01]:
                w_rff = np.linalg.solve(F_rff.T @ F_rff + lam * np.eye(F_rff.shape[1]), F_rff.T @ y_norm)
                pred_train = F_rff @ w_rff
                rmse = np.sqrt(np.mean((pred_train - y_norm)**2))
                if rmse < best_train_rmse:
                    best_train_rmse = rmse
                    Omega_copy = Omega.copy()
                    b_copy = b.copy()
                    w_rff_copy = w_rff.copy()
                    def make_rff_predict(Om, b_r, w_r):
                        def predict(X):
                            Xn = (np.asarray(X, dtype=np.float64) - X_mean) / X_std
                            F_test = np.column_stack([
                                np.sqrt(2.0 / n) * np.cos(Xn @ Om + b_r),
                                np.sqrt(2.0 / n) * np.sin(Xn @ Om + b_r)
                            ])
                            return (F_test @ w_r) * y_std + y_mean
                        return predict
                    best_predict = make_rff_predict(Omega_copy, b_copy, w_rff_copy)
    except:
        pass
    
    # Fallback: if nothing worked, use simple linear regression
    if best_predict is None:
        try:
            F = np.column_stack([np.ones(n)] + [X_norm[:, j] for j in range(d)])
            w = np.linalg.lstsq(F, y_norm, rcond=None)[0]
            def predict(X):
                Xn = (np.asarray(X, dtype=np.float64) - X_mean) / X_std
                F_test = np.column_stack([np.ones(len(Xn))] + [Xn[:, j] for j in range(d)])
                return (F_test @ w) * y_std + y_mean
            best_predict = predict
        except:
            # Absolute fallback: return mean
            best_predict = lambda X: np.full(len(np.asarray(X)), y_mean)
    
    return best_predict
# EVOLVE-BLOCK-END

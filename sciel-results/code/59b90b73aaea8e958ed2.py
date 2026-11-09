import numpy as np
import time
from sklearn.model_selection import KFold

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # 280s limit to be safe within 300s/330s
    
    X = X_train
    y = y_train
    n_samples = X.shape[0]
    
    # 1. Normalize X to [0, 1] for better numerical conditioning
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min
    # Avoid division by zero
    x_range[x_range == 0] = 1.0
    
    def normalize(X_in):
        return (X_in - x_min) / x_range

    X_norm = normalize(X)
    
    # Define feature generation functions
    def get_poly_features(X_n, degree):
        # X_n is (N, 1)
        cols = [np.ones(len(X_n))]
        for i in range(1, degree + 1):
            cols.append(X_n[:, 0] ** i)
        return np.column_stack(cols)
    
    def get_rbf_features(X_n, n_centers, gamma):
        # X_n is (N, 1)
        # Centers evenly spaced in [0, 1]
        centers = np.linspace(0, 1, n_centers).reshape(-1, 1)
        diff = X_n - centers.T
        sq_dist = diff ** 2
        return np.exp(-gamma * sq_dist)

    # Define candidate models as (name, feature_generator, ridge_lambda)
    # We use Ridge regression (L2 regularization) for stability
    candidates = [
        ("Poly_1", lambda X_n: get_poly_features(X_n, 1), 1e-8),
        ("Poly_2", lambda X_n: get_poly_features(X_n, 2), 1e-8),
        ("Poly_3", lambda X_n: get_poly_features(X_n, 3), 1e-8),
        ("Poly_5", lambda X_n: get_poly_features(X_n, 5), 1e-6),
        ("Poly_8", lambda X_n: get_poly_features(X_n, 8), 1e-4),
        ("RBF_10_g10", lambda X_n: get_rbf_features(X_n, 10, 10), 1e-4),
        ("RBF_10_g50", lambda X_n: get_rbf_features(X_n, 10, 50), 1e-4),
        ("RBF_20_g20", lambda X_n: get_rbf_features(X_n, 20, 20), 1e-4),
        ("RBF_20_g100", lambda X_n: get_rbf_features(X_n, 20, 100), 1e-3),
    ]
    
    # 2. Evaluate models using Cross-Validation
    # We use 5-fold CV to estimate generalization error
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_cv_mse = np.inf
    best_model_name = None
    best_feature_func = None
    best_lambda = None
    
    for name, feat_func, lam in candidates:
        if time.time() > deadline:
            break
            
        try:
            cv_errors = []
            for train_idx, val_idx in kf.split(X_norm):
                X_tr_n = X_norm[train_idx]
                y_tr = y[train_idx]
                X_val_n = X_norm[val_idx]
                y_val = y[val_idx]
                
                # Build features
                Phi_tr = feat_func(X_tr_n)
                Phi_val = feat_func(X_val_n)
                
                # Ridge Regression
                # Solve (Phi^T Phi + lambda I) w = Phi^T y
                n_features = Phi_tr.shape[1]
                A = Phi_tr.T @ Phi_tr + lam * np.eye(n_features)
                b = Phi_tr.T @ y_tr
                
                try:
                    w = np.linalg.solve(A, b)
                    pred_val = Phi_val @ w
                    mse = np.mean((pred_val - y_val) ** 2)
                    cv_errors.append(mse)
                except np.linalg.LinAlgError:
                    # If solve fails, use least squares with higher regularization
                    w = np.linalg.lstsq(Phi_tr + 1e-8 * np.random.randn(*Phi_tr.shape), y_tr, rcond=None)[0]
                    pred_val = Phi_val @ w
                    mse = np.mean((pred_val - y_val) ** 2)
                    cv_errors.append(mse)
            
            mean_cv_mse = np.mean(cv_errors)
            
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_model_name = name
                best_feature_func = feat_func
                best_lambda = lam
                
        except Exception:
            continue

    # 3. Fallback to Poly_2 if CV failed or no model was selected
    if best_feature_func is None:
        best_feature_func = lambda X_n: get_poly_features(X_n, 2)
        best_lambda = 1e-8
        best_model_name = "Poly_2_Fallback"

    # 4. Refit the best model on the ENTIRE training set
    Phi_full = best_feature_func(X_norm)
    n_features = Phi_full.shape[1]
    A_full = Phi_full.T @ Phi_full + best_lambda * np.eye(n_features)
    b_full = Phi_full.T @ y
    
    try:
        w_full = np.linalg.solve(A_full, b_full)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        w_full = np.linalg.lstsq(Phi_full, y, rcond=None)[0]

    # 5. Define predict function
    def predict(X):
        X_n = normalize(X)
        Phi_test = best_feature_func(X_n)
        preds = Phi_test @ w_full
        # Ensure finite values
        preds[~np.isfinite(preds)] = 0.0
        return preds

    return predict

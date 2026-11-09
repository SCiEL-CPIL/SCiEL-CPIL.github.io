"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 290.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Build polynomial features up to degree 4
    def build_poly_features(X, max_deg=4):
        cols = [np.ones(len(X))]
        for degree in range(1, max_deg + 1):
            for combo in combinations_with_replacement(range(d), degree):
                term = np.ones(len(X))
                for idx in combo:
                    term = term * X[:, idx]
                cols.append(term)
        return np.column_stack(cols)
    
    # Build extended features: polynomials + non-linear transforms
    def build_extended_features(X, max_deg=4):
        cols = [np.ones(len(X))]
        # Polynomial terms
        for degree in range(1, max_deg + 1):
            for combo in combinations_with_replacement(range(d), degree):
                term = np.ones(len(X))
                for idx in combo:
                    term = term * X[:, idx]
                cols.append(term)
        # Non-linear features per input
        for j in range(d):
            xj = X[:, j]
            cols.append(np.log(xj + 1e-10))
            cols.append(np.sqrt(xj + 1e-10))
            cols.append(1.0 / (xj + 1e-10))
            cols.append(xj * np.log(xj + 1e-10))
            cols.append(np.log(xj + 1e-10) ** 2)
            cols.append(np.sqrt(xj + 1e-10) ** 3)
        # Pairwise non-linear products
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.log(X[:, i] + 1e-10) * np.log(X[:, j] + 1e-10))
                cols.append(np.sqrt(X[:, i] + 1e-10) * np.sqrt(X[:, j] + 1e-10))
                cols.append((1.0 / (X[:, i] + 1e-10)) * (1.0 / (X[:, j] + 1e-10)))
        return np.column_stack(cols)
    
    # Simple least squares fallback
    def simple_fit(X, y, feature_builder, max_deg=4):
        F = feature_builder(X, max_deg)
        try:
            w, *_ = np.linalg.lstsq(F, y, rcond=None)
        except:
            w = np.zeros(F.shape[1])
        
        def predict(X_pred):
            F_pred = feature_builder(X_pred, max_deg)
            return F_pred @ w
        return predict
    
    # Ridge regression with given lambda
    def ridge_fit(X, y, feature_builder, max_deg=4, lam=1e-6):
        F = feature_builder(X, max_deg)
        n_features = F.shape[1]
        try:
            A = F.T @ F + lam * np.eye(n_features)
            b = F.T @ y
            w = np.linalg.solve(A, b)
        except:
            try:
                w, *_ = np.linalg.lstsq(F, y, rcond=None)
            except:
                w = np.zeros(n_features)
        
        def predict(X_pred):
            F_pred = feature_builder(X_pred, max_deg)
            return F_pred @ w
        return predict
    
    # 5-fold cross-validation to estimate generalization error
    def cross_validate(X, y, feature_builder, max_deg=4, lam=0.0, n_folds=5):
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        errors = []
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = X[val_idx]
            y_val_cv = y[val_idx]
            
            F_train = feature_builder(X_train_cv, max_deg)
            n_features = F_train.shape[1]
            
            try:
                if lam == 0.0:
                    w, *_ = np.linalg.lstsq(F_train, y_train_cv, rcond=None)
                else:
                    A = F_train.T @ F_train + lam * np.eye(n_features)
                    b = F_train.T @ y_train_cv
                    w = np.linalg.solve(A, b)
            except:
                continue
            
            F_val = feature_builder(X_val_cv, max_deg)
            pred = F_val @ w
            mse = np.mean((pred - y_val_cv) ** 2)
            errors.append(mse)
        
        if len(errors) == 0:
            return np.inf
        return np.mean(errors)
    
    # Try polynomial features with different degrees
    best_predictor = None
    best_cv_score = np.inf
    
    # Try degree 3 and 4 polynomials with different regularization
    for max_deg in [3, 4]:
        for lam in [0.0, 1e-8, 1e-6, 1e-4, 1e-2]:
            if time.time() > deadline - 20:
                break
            cv_score = cross_validate(X, y, build_poly_features, max_deg, lam)
            if cv_score < best_cv_score:
                best_cv_score = cv_score
                best_predictor = ridge_fit(X, y, build_poly_features, max_deg, lam)
    
    # Try extended features if time allows
    if time.time() < deadline - 30:
        for max_deg in [3, 4]:
            for lam in [0.0, 1e-6, 1e-4, 1e-2]:
                if time.time() > deadline - 15:
                    break
                cv_score = cross_validate(X, y, build_extended_features, max_deg, lam)
                if cv_score < best_cv_score:
                    best_cv_score = cv_score
                    best_predictor = ridge_fit(X, y, build_extended_features, max_deg, lam)
    
    # Fallback if nothing worked
    if best_predictor is None:
        best_predictor = simple_fit(X, y, build_poly_features, 3)
    
    return best_predictor
# EVOLVE-BLOCK-END

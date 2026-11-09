import numpy as np
import time
import sys
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before 330s limit
    
    N, D = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    def time_ok():
        return time.time() < deadline
    
    # --- Feature Generation Functions ---
    
    def build_poly_features(X, degree):
        """Build polynomial features up to given degree."""
        cols = [np.ones(len(X))]
        for deg in range(1, degree + 1):
            for idx in product(range(D), repeat=deg):
                # Only include sorted indices to avoid duplicates
                sorted_idx = tuple(sorted(idx))
                # Check if this combination was already added
                term = np.ones(len(X))
                for k in range(deg):
                    term = term * X[:, sorted_idx[k]]
                cols.append(term)
        return np.column_stack(cols)
    
    def build_trig_features(X, max_freq=3):
        """Build trigonometric features."""
        cols = [np.ones(len(X))]
        for j in range(D):
            for k in range(1, max_freq + 1):
                cols.append(np.sin(k * X[:, j]))
                cols.append(np.cos(k * X[:, j]))
        return np.column_stack(cols)
    
    def build_mixed_features(X):
        """Build mixed transcendental features: x_i * sin(x_j), x_i * cos(x_j), etc."""
        cols = [np.ones(len(X))]
        for i in range(D):
            for j in range(D):
                if i != j:
                    cols.append(X[:, i] * np.sin(X[:, j]))
                    cols.append(X[:, i] * np.cos(X[:, j]))
                    cols.append(np.sin(X[:, i]) * X[:, j])
                    cols.append(np.cos(X[:, i]) * X[:, j])
        # Also add some single-variable transcendentals
        for j in range(D):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
            cols.append(np.exp(-X[:, j]))
            cols.append(np.log(X[:, j]))
            cols.append(1.0 / X[:, j])
            cols.append(np.sqrt(X[:, j]))
        return np.column_stack(cols)
    
    def build_rbf_features(X, centers, bandwidths):
        """Build RBF features."""
        cols = [np.ones(len(X))]
        for c in centers:
            for b in bandwidths:
                dist_sq = np.sum((X - c) ** 2, axis=1)
                cols.append(np.exp(-b * dist_sq))
        return np.column_stack(cols)
    
    def fit_and_score(X, y, features):
        """Fit linear model and return training MSE."""
        try:
            # Use ridge regression for numerical stability
            # Small ridge penalty to handle collinearity
            lambda_val = 1e-8
            n_features = features.shape[1]
            A = features.T @ features + lambda_val * np.eye(n_features)
            b = features.T @ y
            w = np.linalg.solve(A, b)
            pred = features @ w
            mse = np.mean((pred - y) ** 2)
            return mse, w
        except:
            return np.inf, None
    
    # --- Strategy: Try multiple feature sets, pick best by training MSE ---
    # Since data is noiseless, training MSE should be very small for the right model
    
    best_mse = np.inf
    best_predictor = None
    
    # 1. Try polynomial features of various degrees
    for degree in [2, 3, 4]:
        if not time_ok():
            break
        try:
            feat = build_poly_features(X_train, degree)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_poly_features, dg=degree, ww=w: f(X, dg) @ ww
        except:
            pass
    
    # 2. Try trigonometric features
    if time_ok():
        try:
            feat = build_trig_features(X_train, max_freq=3)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_trig_features, ww=w: f(X, 3) @ ww
        except:
            pass
    
    # 3. Try mixed features
    if time_ok():
        try:
            feat = build_mixed_features(X_train)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_mixed_features, ww=w: f(X) @ ww
        except:
            pass
    
    # 4. Try RBF features with a few centers
    if time_ok():
        try:
            # Use some training points as centers
            n_centers = min(15, N)
            # Spread centers across the domain
            indices = np.linspace(0, N - 1, n_centers, dtype=int)
            centers = X_train[indices]
            bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]
            feat = build_rbf_features(X_train, centers, bandwidths)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_rbf_features, cs=centers, bs=bandwidths, ww=w: f(X, cs, bs) @ ww
        except:
            pass
    
    # 5. Try combined feature sets
    if time_ok():
        try:
            feat1 = build_poly_features(X_train, 3)
            feat2 = build_trig_features(X_train, max_freq=2)
            feat = np.column_stack([feat1, feat2])
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f1=build_poly_features, f2=build_trig_features, ww=w: np.column_stack([f1(X, 3), f2(X, 2)]) @ ww
        except:
            pass
    
    # 6. If still not good, try a more aggressive RBF with more centers
    if time_ok() and best_mse > 1e-6:
        try:
            n_centers = min(30, N)
            indices = np.linspace(0, N - 1, n_centers, dtype=int)
            centers = X_train[indices]
            bandwidths = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            feat = build_rbf_features(X_train, centers, bandwidths)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_rbf_features, cs=centers, bs=bandwidths, ww=w: f(X, cs, bs) @ ww
        except:
            pass
    
    # 7. Fallback: simple linear
    if best_predictor is None:
        try:
            feat = np.column_stack([np.ones(N), X_train])
            mse, w = fit_and_score(X_train, y_train, feat)
            best_predictor = lambda X: np.column_stack([np.ones(X.shape[0]), X]) @ w
        except:
            best_predictor = lambda X: np.full(X.shape[0], y_mean)
    
    return best_predictor

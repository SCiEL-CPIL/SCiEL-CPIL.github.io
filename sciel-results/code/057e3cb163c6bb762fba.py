"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    def time_left():
        return 270.0 - (time.time() - start_time)
    
    # Generate polynomial features up to degree max_deg
    def poly_features(X, max_deg=3):
        cols = [np.ones(len(X))]
        for deg in range(1, max_deg + 1):
            for idx in product(range(d), repeat=deg):
                if sorted(idx) != idx:
                    continue
                feat = np.ones(len(X))
                for i in idx:
                    feat *= X[:, i]
                cols.append(feat)
        return np.column_stack(cols)
    
    # Generate trig features
    def trig_features(X, freqs=1.0):
        cols = [np.ones(len(X))]
        for j in range(d):
            for f in freqs:
                cols.append(np.sin(f * X[:, j]))
                cols.append(np.cos(f * X[:, j]))
        # Cross terms for 2D
        if d >= 2:
            for f in freqs[:2]:
                cols.append(np.sin(f * X[:, 0]) * np.cos(f * X[:, 1]))
                cols.append(np.cos(f * X[:, 0]) * np.sin(f * X[:, 1]))
        return np.column_stack(cols)
    
    # RBF features
    def rbf_features(X, centers, gammas):
        cols = []
        for c, g in zip(centers, gammas):
            dist = np.sum((X - c) ** 2, axis=1)
            cols.append(np.exp(-g * dist))
        return np.column_stack(cols)
    
    # Ridge regression
    def ridge_solve(F, y, alpha=1e-6):
        FtF = F.T @ F + alpha * np.eye(F.shape[1])
        Fty = F.T @ y
        w = np.linalg.solve(FtF, Fty)
        return w
    
    # Predict with given weights
    def predict_with_weights(F, w):
        return F @ w
    
    # Try multiple models and pick best by validation
    np.random.seed(42)
    indices = np.random.permutation(n)
    n_val = max(1, n // 5)
    val_idx = indices[:n_val]
    tr_idx = indices[n_val:]
    
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    best_model = None
    best_val_mse = np.inf
    best_predict = lambda X: np.full(len(X), y_mean)
    
    models = []
    
    # Polynomial models of various degrees
    for deg in [2, 3, 4, 5]:
        if time_left() < 10:
            break
        F_tr = poly_features(X_tr, deg)
        F_val = poly_features(X_val, deg)
        for alpha in [1e-8, 1e-6, 1e-4, 1e-2]:
            w = ridge_solve(F_tr, y_tr, alpha)
            pred_val = F_val @ w
            mse = np.mean((pred_val - y_val) ** 2)
            if mse < best_val_mse:
                best_val_mse = mse
                best_model = ('poly', deg, alpha)
                # Refit on all data
                F_all = poly_features(X_train, deg)
                w_all = ridge_solve(F_all, y_train, alpha)
                best_predict = lambda X, F_fn=lambda X_: poly_features(X_, deg), w_=w_all: F_fn(X) @ w_
    
    # Trig models
    for freqs in [[1.0], [0.5, 1.0], [0.5, 1.0, 1.5, 2.0]]:
        if time_left() < 10:
            break
        F_tr = trig_features(X_tr, freqs)
        F_val = trig_features(X_val, freqs)
        for alpha in [1e-8, 1e-6, 1e-4]:
            w = ridge_solve(F_tr, y_tr, alpha)
            pred_val = F_val @ w
            mse = np.mean((pred_val - y_val) ** 2)
            if mse < best_val_mse:
                best_val_mse = mse
                best_model = ('trig', freqs, alpha)
                F_all = trig_features(X_train, freqs)
                w_all = ridge_solve(F_all, y_train, alpha)
                best_predict = lambda X, F_fn=lambda X_: trig_features(X_, freqs), w_=w_all: F_fn(X) @ w_
    
    # Combined poly + trig
    if time_left() > 20:
        for deg in [2, 3]:
            for freqs in [[1.0], [0.5, 1.0]]:
                F_tr = np.column_stack([poly_features(X_tr, deg), trig_features(X_tr, freqs)])
                F_val = np.column_stack([poly_features(X_val, deg), trig_features(X_val, freqs)])
                for alpha in [1e-8, 1e-6, 1e-4]:
                    w = ridge_solve(F_tr, y_tr, alpha)
                    pred_val = F_val @ w
                    mse = np.mean((pred_val - y_val) ** 2)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_model = ('combined', deg, freqs, alpha)
                        F_all = np.column_stack([poly_features(X_train, deg), trig_features(X_train, freqs)])
                        w_all = ridge_solve(F_all, y_train, alpha)
                        best_predict = lambda X, d_=deg, f_=freqs, w_=w_all: np.column_stack([poly_features(X, d_), trig_features(X, f_)]) @ w_
    
    # RBF models
    if time_left() > 15:
        n_centers = min(20, n)
        center_idx = np.random.choice(n, n_centers, replace=False)
        centers = X_train[center_idx]
        for n_gamma in [3, 5]:
            gammas = np.logspace(-2, 2, n_gamma)
            F_tr = rbf_features(X_tr, centers, gammas)
            F_val = rbf_features(X_val, centers, gammas)
            for alpha in [1e-6, 1e-4, 1e-2]:
                w = ridge_solve(F_tr, y_tr, alpha)
                pred_val = F_val @ w
                mse = np.mean((pred_val - y_val) ** 2)
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_model = ('rbf', n_gamma, alpha)
                    F_all = rbf_features(X_train, centers, gammas)
                    w_all = ridge_solve(F_all, y_train, alpha)
                    best_predict = lambda X, c_=centers, g_=gammas, w_=w_all: rbf_features(X, c_, g_) @ w_
    
    # Final: use best model on full training data
    if best_model is None:
        # Fallback to simple linear
        F_all = np.column_stack([np.ones(n), X_train])
        w_all = ridge_solve(F_all, y_train, 1e-6)
        return lambda X: np.column_stack([np.ones(len(X)), X]) @ w_all
    
    # Ensure predictions are finite
    def safe_predict(X):
        pred = best_predict(X)
        pred = np.where(np.isfinite(pred), pred, y_mean)
        return pred
    
    return safe_predict
# EVOLVE-BLOCK-END

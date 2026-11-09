"""Symbolic regression using a hybrid basis of polynomials, trigonometric functions, and RBFs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    t_start = time.time()
    DEADLINE = 325.0  # 5 seconds margin before 330s kill
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Standardize X
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    X_std = (X - mu) / sd
    
    # Function to generate features for any X given mu, sd
    def make_features(X_raw):
        X_r = np.asarray(X_raw, dtype=np.float64)
        X_s = (X_r - mu) / sd
        n_s = X_s.shape[0]
        
        # 1. Polynomials (up to degree 4 for univariate, degree 2 for multivariate interactions)
        # Constant
        cols = [np.ones(n_s)]
        
        if d == 1:
            x = X_s[:, 0]
            # Polynomials
            for p in range(1, 5):
                cols.append(x**p)
            # Trig
            for k in [1, 2, 3]:
                cols.append(np.sin(k * x))
                cols.append(np.cos(k * x))
            # Exp
            cols.append(np.exp(x))
            cols.append(np.exp(-x))
        else:
            # Linear terms
            for j in range(d):
                cols.append(X_s[:, j])
            
            # Quadratic terms (including interactions)
            for i in range(d):
                for j in range(i, d):
                    cols.append(X_s[:, i] * X_s[:, j])
            
            # Univariate Trig
            for j in range(d):
                x_j = X_s[:, j]
                for k in [1, 2, 3]:
                    cols.append(np.sin(k * x_j))
                    cols.append(np.cos(k * x_j))
            
            # RBFs
            # Define grid centers in standardized space
            # For d=2: 6x6 grid in [-1.5, 1.5]
            # For d=3: 4x4x4 grid in [-1.5, 1.5]
            if d == 2:
                grid_range = np.linspace(-1.5, 1.5, 6)
                centers = np.array([[c1, c2] for c1 in grid_range for c2 in grid_range])
            else: # d == 3
                grid_range = np.linspace(-1.5, 1.5, 4)
                centers = np.array([[c1, c2, c3] for c1 in grid_range for c2 in grid_range for c3 in grid_range])
            
            # Calculate distances
            # X_s is (n_s, d), centers is (n_c, d)
            # diff = X_s[:, np.newaxis, :] - centers[np.newaxis, :, :]
            # dist_sq = np.sum(diff**2, axis=2)
            
            # Efficient calculation
            # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2
            X_sq = np.sum(X_s**2, axis=1, keepdims=True) # (n_s, 1)
            C_sq = np.sum(centers**2, axis=1, keepdims=True).T # (1, n_c)
            dist_sq = X_sq - 2 * X_s @ centers.T + C_sq
            np.maximum(dist_sq, 0, out=dist_sq) # numerical safety
            
            gamma = 1.5
            rbf_vals = np.exp(-gamma * dist_sq)
            cols.extend(rbf_vals.T)
        
        return np.column_stack(cols)
    
    # Generate training features
    Phi = make_features(X)
    
    # Ridge Regression
    # lambda = 1e-3 is a reasonable start. 
    # We can try a small grid search for lambda if time permits, but static is safer.
    lam = 1e-3
    
    # Solve (Phi^T Phi + lam I) w = Phi^T y
    A = Phi.T @ Phi
    b = Phi.T @ y
    A += lam * np.eye(A.shape[0])
    
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        
    def predict(X_test):
        X_t = np.asarray(X_test, dtype=np.float64)
        # Ensure 2D
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        Phi_t = make_features(X_t)
        preds = Phi_t @ w
        # Ensure finite
        preds = np.where(np.isfinite(preds), preds, 0.0)
        return preds
    
    return predict
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin for 330s kill
    
    n, d = X_train.shape
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_mean = np.mean(y_train)
        return lambda X: np.full(len(X), y_mean)
    
    # Try multiple feature expansions and pick the best one based on training MSE
    # We use training MSE as a proxy since we don't have test data
    
    def make_features(X, degree):
        """Create polynomial features up to given degree."""
        cols = [np.ones(len(X))]
        # Add linear terms
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        # Add higher degree terms
        if degree >= 2:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
        if degree >= 3:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(cols)
    
    def make_features_with_basis(X, basis_funcs):
        """Create features using custom basis functions."""
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            cols.append(X[:, j])
            # Add some non-linear basis functions
            for bf in basis_funcs:
                cols.append(bf(X[:, j]))
        # Add interactions
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)
    
    # Define various basis functions
    def basis_funcs_list():
        fns = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: np.exp(-x),
            lambda x: np.log1p(x),
            lambda x: np.sqrt(x),
            lambda x: x**2,
            lambda x: np.sin(np.pi * x),
            lambda x: np.cos(np.pi * x),
        ]
        return fns
    
    best_predict = None
    best_train_mse = np.inf
    
    # Strategy 1: Polynomial features of degree 2 and 3
    for degree in [2, 3]:
        if time.time() > deadline - 10:
            break
        try:
            F = make_features(X_train, degree)
            w, residuals, rank, sv = np.linalg.lstsq(F, y_train, rcond=None)
            pred = F @ w
            mse = np.mean((pred - y_train)**2)
            if mse < best_train_mse:
                best_train_mse = mse
                F_fn = lambda X: make_features(X, degree)
                best_predict = lambda X, F_fn=F_fn, w=w: F_fn(X) @ w
        except Exception:
            pass
    
    # Strategy 2: Basis function features
    if time.time() < deadline - 15:
        bfs = basis_funcs_list()
        # Try different subsets of basis functions
        for n_basis in [2, 4, len(bfs)]:
            if time.time() > deadline - 10:
                break
            selected_bfs = bfs[:n_basis]
            try:
                F = make_features_with_basis(X_train, selected_bfs)
                w, residuals, rank, sv = np.linalg.lstsq(F, y_train, rcond=None)
                pred = F @ w
                mse = np.mean((pred - y_train)**2)
                if mse < best_train_mse:
                    best_train_mse = mse
                    bfs_copy = selected_bfs
                    best_predict = lambda X, bfs=bfs_copy, w=w: make_features_with_basis(X, bfs) @ w
            except Exception:
                pass
    
    # Strategy 3: Ridge regression with polynomial features (regularized)
    if time.time() < deadline - 15:
        for degree in [2, 3]:
            if time.time() > deadline - 10:
                break
            try:
                F = make_features(X_train, degree)
                # Try different lambda values
                for lam in [0.001, 0.01, 0.1, 1.0]:
                    if time.time() > deadline - 5:
                        break
                    A = F.T @ F + lam * np.eye(F.shape[1])
                    b = F.T @ y_train
                    w = np.linalg.solve(A, b)
                    pred = F @ w
                    mse = np.mean((pred - y_train)**2)
                    if mse < best_train_mse:
                        best_train_mse = mse
                        F_fn = lambda X: make_features(X, degree)
                        best_predict = lambda X, F_fn=F_fn, w=w: F_fn(X) @ w
            except Exception:
                pass
    
    # Strategy 4: Spline-like features using piecewise linear basis
    if time.time() < deadline - 15:
        try:
            def make_spline_features(X):
                cols = [np.ones(len(X))]
                for j in range(X.shape[1]):
                    cols.append(X[:, j])
                    # Add piecewise linear features
                    for knot_frac in [0.25, 0.5, 0.75]:
                        knot = np.quantile(X[:, j], knot_frac)
                        cols.append(np.maximum(0, X[:, j] - knot))
                        cols.append(np.maximum(0, knot - X[:, j]))
                # Add interactions
                for i in range(X.shape[1]):
                    for j in range(i + 1, X.shape[1]):
                        cols.append(X[:, i] * X[:, j])
                return np.column_stack(cols)
            
            F = make_spline_features(X_train)
            w, residuals, rank, sv = np.linalg.lstsq(F, y_train, rcond=None)
            pred = F @ w
            mse = np.mean((pred - y_train)**2)
            if mse < best_train_mse:
                best_train_mse = mse
                best_predict = lambda X, w=w: make_spline_features(X) @ w
        except Exception:
            pass
    
    # Fallback to simple linear model if nothing else worked
    if best_predict is None:
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X_train]), y_train, rcond=None)
        best_predict = lambda X: np.column_stack([np.ones(len(X)), X]) @ w
    
    # Ensure predictions are finite
    def safe_predict(X):
        try:
            result = best_predict(X)
            result = np.asarray(result, dtype=float)
            if not np.all(np.isfinite(result)):
                result = np.nan_to_num(result, nan=np.mean(y_train), posinf=np.max(y_train), neginf=np.min(y_train))
            return result
        except Exception:
            return np.full(len(X), np.mean(y_train))
    
    return safe_predict
# EVOLVE-BLOCK-END

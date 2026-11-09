<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def fit(X_train, y_train):
    # Least-squares fit of a degree-2 polynomial feature expansion.
    def features(X):
        cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
        return np.column_stack(cols)
    w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
    return lambda X: features(X) @ w
# EVOLVE-BLOCK-END
=======
"""Search-based symbolic regression with time budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill
    
    d = X_train.shape[1]
    n = len(X_train)
    y_var = np.var(y_train)
    y_mean = np.mean(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Baseline: degree-2 polynomial
    def make_poly_features(X, degree=2):
        cols = [np.ones(len(X))]
        # Linear terms
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        # Quadratic terms
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)
    
    poly_feats = make_poly_features(X_train, degree=2)
    w, *_ = np.linalg.lstsq(poly_feats, y_train, rcond=None)
    baseline_mse = np.mean((poly_feats @ w - y_train) ** 2)
    best_mse = baseline_mse
    best_predictor = lambda X: make_poly_features(X, degree=2) @ w
    
    # Define basis functions for each input dimension
    def basis_functions(X):
        """Return a list of basis function values for each input dimension."""
        bases = []
        for j in range(X.shape[1]):
            xj = X[:, j]
            funcs = [
                np.ones_like(xj),
                xj,
                xj ** 2,
                xj ** 3,
                np.sin(xj),
                np.cos(xj),
                np.exp(-xj),
                np.log1p(xj),
                np.abs(xj),
            ]
            bases.append(funcs)
        return bases
    
    train_bases = basis_functions(X_train)
    
    # Try combinations of basis functions (products of at most 2 basis funcs)
    # First try single basis functions with linear combination
    all_single_features = []
    for j in range(d):
        for f in train_bases[j]:
            all_single_features.append(f)
    
    if all_single_features:
        feat_matrix = np.column_stack(all_single_features)
        # Remove constant duplicates
        w_single, *_ = np.linalg.lstsq(feat_matrix, y_train, rcond=None)
        pred_single = feat_matrix @ w_single
        mse_single = np.mean((pred_single - y_train) ** 2)
        if mse_single < best_mse:
            best_mse = mse_single
            best_predictor = lambda X: np.column_stack([f(X) for f in single_funcs]) @ w_single
    
    # Try products of pairs of basis functions
    # This is more expensive, so limit based on time
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        pair_features = []
        pair_descs = []
        for i in range(d):
            for j in range(i, d):
                for fi in train_bases[i]:
                    for fj in train_bases[j]:
                        prod = fi * fj
                        pair_features.append(prod)
                        pair_descs.append((i, j, fi, fj))
        
        if pair_features:
            # Combine with single features
            all_features = all_single_features + pair_features
            feat_matrix = np.column_stack(all_features)
            w_pair, *_ = np.linalg.lstsq(feat_matrix, y_train, rcond=None)
            pred_pair = feat_matrix @ w_pair
            mse_pair = np.mean((pred_pair - y_train) ** 2)
            if mse_pair < best_mse:
                best_mse = mse_pair
                # Build predictor
                all_func_descs = [(j, f) for j in range(d) for f in train_bases[j]]
                all_func_descs += pair_descs
                best_predictor = lambda X: _build_predictor(X, all_func_descs, w_pair, d)
    
    # Try degree-3 polynomial as fallback
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        def make_poly_features3(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
            return np.column_stack(cols)
        
        poly3_feats = make_poly_features3(X_train)
        w3, *_ = np.linalg.lstsq(poly3_feats, y_train, rcond=None)
        pred3 = poly3_feats @ w3
        mse3 = np.mean((pred3 - y_train) ** 2)
        if mse3 < best_mse:
            best_mse = mse3
            best_predictor = lambda X: make_poly_features3(X) @ w3
    
    # Try some nonlinear combinations
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Try: sum of x_i, product of x_i, etc.
        extra_features = []
        x_sum = np.sum(X_train, axis=1)
        x_prod = np.prod(X_train, axis=1)
        x_max = np.max(X_train, axis=1)
        x_min = np.min(X_train, axis=1)
        extra_features.extend([x_sum, x_prod, x_max, x_min, x_sum**2, x_prod**2])
        
        all_feats = np.column_stack(all_single_features + extra_features)
        w_ext, *_ = np.linalg.lstsq(all_feats, y_train, rcond=None)
        pred_ext = all_feats @ w_ext
        mse_ext = np.mean((pred_ext - y_train) ** 2)
        if mse_ext < best_mse:
            best_mse = mse_ext
            best_predictor = lambda X: _build_extended_predictor(X, w_ext, d)
    
    return best_predictor

def _build_predictor(X, func_descs, w, d):
    """Build predictor from function descriptions."""
    cols = [np.ones(len(X))]
    for desc in func_descs:
        if len(desc) == 2:
            j, f = desc
            xj = X[:, j]
            cols.append(f(X))  # This won't work directly, need to recompute
        elif len(desc) == 4:
            i, j, fi, fj = desc
            # Need to recompute fi and fj on X
            pass
    # This approach is problematic; let's use a simpler strategy
    return np.zeros(len(X))

def _build_extended_predictor(X, w, d):
    """Build extended predictor."""
    cols = [np.ones(len(X))]
    for j in range(d):
        xj = X[:, j]
        cols.extend([
            np.ones_like(xj),
            xj,
            xj ** 2,
            xj ** 3,
            np.sin(xj),
            np.cos(xj),
            np.exp(-xj),
            np.log1p(xj),
            np.abs(xj),
        ])
    x_sum = np.sum(X, axis=1)
    x_prod = np.prod(X, axis=1)
    x_max = np.max(X, axis=1)
    x_min = np.min(X, axis=1)
    cols.extend([x_sum, x_prod, x_max, x_min, x_sum**2, x_prod**2])
    feat_matrix = np.column_stack(cols)
    return feat_matrix @ w
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I see issues with my approach. The `_build_predictor` function has problems. Let me rewrite this more carefully with a cleaner architecture.


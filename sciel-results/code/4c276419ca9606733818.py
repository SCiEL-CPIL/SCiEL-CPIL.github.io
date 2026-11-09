"""Symbolic regression via gradient-boosted decision trees with early stopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _decision_tree_regressor(X, y, max_depth=5, min_samples_leaf=2, max_iter=200, learning_rate=0.1,
                             val_fraction=0.25, deadline=None):
    """
    Simple gradient boosting with squared loss using CART-style regression trees.
    Returns a list of (tree, weight) pairs where tree is a dict-based structure.
    """
    n, d = X.shape
    if n < 10:
        # Fallback: constant predictor
        return [({'leaf': True, 'value': np.mean(y)}, 1.0)]

    # Split into train/val for early stopping
    rng = np.random.RandomState(123)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = max(2, int(n * val_fraction))
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    residuals = y_tr.copy()
    predictions = np.zeros(n)
    predictions[tr_idx] = np.mean(y_tr)
    predictions[val_idx] = np.mean(y_tr)

    trees = []
    best_val_mse = np.mean((predictions[val_idx] - y_val) ** 2)
    patience = 20
    no_improve = 0

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Build a regression tree on current residuals
        tree = _build_cart_tree(X_tr, residuals, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        if tree is None:
            break

        # Update residuals on train
        pred_tr = _predict_tree(X_tr, tree)
        residuals -= pred_tr

        # Update predictions
        predictions[tr_idx] += pred_tr
        predictions[val_idx] += _predict_tree(X_val, tree)

        trees.append((tree, learning_rate))

        # Check validation MSE
        val_mse = np.mean((predictions[val_idx] - y_val) ** 2)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Refit on all data with same number of trees (no early stopping bias)
    # Actually, let's just use the ensemble as-is since we stopped at best val
    return trees

def _build_cart_tree(X, y, max_depth=5, min_samples_leaf=2, depth=0):
    """Build a CART regression tree using variance reduction."""
    n = len(y)
    if n < min_samples_leaf * 2 or depth >= max_depth:
        return {'leaf': True, 'value': np.mean(y)}

    best_mse = np.var(y) * n  # Total SSE
    best_feat = -1
    best_thresh = 0

    d = X.shape[1]
    # For speed, subsample features if many
    if d > 8:
        feat_indices = np.random.RandomState(depth * 1000 + 42).choice(d, size=8, replace=False)
    else:
        feat_indices = range(d)

    for f in feat_indices:
        vals = X[:, f]
        if np.allclose(vals, vals[0]):
            continue
        # Use quantile-based thresholds for efficiency
        quantiles = np.quantile(vals, np.linspace(0.1, 0.9, 9))
        for thresh in quantiles:
            mask_left = vals <= thresh
            mask_right = ~mask_left
            n_left = np.sum(mask_left)
            n_right = n - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            sse_left = np.sum((y[mask_left] - np.mean(y[mask_left])) ** 2)
            sse_right = np.sum((y[mask_right] - np.mean(y[mask_right])) ** 2)
            sse = sse_left + sse_right
            if sse < best_mse:
                best_mse = sse
                best_feat = f
                best_thresh = thresh

    if best_feat == -1:
        return {'leaf': True, 'value': np.mean(y)}

    mask_left = X[:, best_feat] <= best_thresh
    left_tree = _build_cart_tree(X[mask_left], y[mask_left], max_depth, min_samples_leaf, depth + 1)
    right_tree = _build_cart_tree(X[~mask_left], y[~mask_left], max_depth, min_samples_leaf, depth + 1)

    return {'leaf': False, 'feature': best_feat, 'threshold': best_thresh,
            'left': left_tree, 'right': right_tree}

def _predict_tree(X, tree):
    """Predict using a single CART tree."""
    n = X.shape[0]
    preds = np.zeros(n)
    if tree['leaf']:
        return np.full(n, tree['value'])

    mask_left = X[:, tree['feature']] <= tree['threshold']
    preds[mask_left] = _predict_tree(X[mask_left], tree['left'])
    preds[~mask_left] = _predict_tree(X[~mask_left], tree['right'])
    return preds

def _predict_ensemble(X, trees):
    """Predict using ensemble of trees."""
    n = X.shape[0]
    preds = np.zeros(n)
    for tree, weight in trees:
        preds += weight * _predict_tree(X, tree)
    return preds

def fit(X_train, y_train):
    """Fit gradient-boosted decision trees with early stopping."""
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    deadline = time.time() + 280  # Safety margin

    # Try multiple configurations and pick best via internal validation
    configs = [
        {'max_depth': 3, 'min_samples_leaf': 3, 'max_iter': 150, 'learning_rate': 0.1},
        {'max_depth': 4, 'min_samples_leaf': 2, 'max_iter': 200, 'learning_rate': 0.08},
        {'max_depth': 5, 'min_samples_leaf': 2, 'max_iter': 150, 'learning_rate': 0.06},
        {'max_depth': 3, 'min_samples_leaf': 5, 'max_iter': 200, 'learning_rate': 0.15},
    ]

    best_trees = None
    best_val_score = np.inf

    for cfg in configs:
        if time.time() > deadline:
            break
        try:
            trees = _decision_tree_regressor(X, y, deadline=deadline, **cfg)
            if len(trees) == 0:
                continue

            # Evaluate on full training set as proxy
            pred = _predict_ensemble(X, trees)
            mse = np.mean((pred - y) ** 2)
            var_y = np.var(y)
            if var_y > 1e-12:
                norm_mse = mse / var_y
            else:
                norm_mse = mse

            if norm_mse < best_val_score:
                best_val_score = norm_mse
                best_trees = trees
        except Exception:
            continue

    if best_trees is None:
        # Fallback: mean predictor
        mean_y = np.mean(y)
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            return np.full(X_new.shape[0], mean_y)
        return predict

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        return _predict_ensemble(X_new, best_trees)

    return predict
# EVOLVE-BLOCK-END

"""Symbolic regression over 10 hidden equations with a rich basis and forward feature selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = t0 + 320  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Build a rich candidate feature basis
    # We include: 1, x_i, x_i^2, x_i^3, x_i^4, x_i^5, x_i^6,
    # x_i*x_j (i<=j), sin(x_i), cos(x_i), exp(x_i), log(x_i), tanh(x_i),
    # 1/x_i, x_i^0.5, sqrt(x_i), and interactions like sin(x_i)*x_j, etc.
    # But we must be careful about numerical stability and time.

    # Let's build features incrementally and use forward selection.
    # Start with intercept.
    basis = [np.ones(n)]
    feature_names = ['const']

    # Add individual powers and transforms for each input
    for j in range(d):
        xj = X[:, j]
        # Powers
        for p in range(1, 7):
            basis.append(xj ** p)
            feature_names.append(f'x{j}^{p}')
        # Transforms
        basis.append(np.sin(xj))
        feature_names.append(f'sin(x{j})')
        basis.append(np.cos(xj))
        feature_names.append(f'cos(x{j})')
        # exp and log can be unstable; clip to avoid overflow
        basis.append(np.exp(np.clip(xj, -10, 10)))
        feature_names.append(f'exp(x{j})')
        # log: only for positive; inputs are in [1, ...] so safe
        basis.append(np.log(xj + 1e-12))
        feature_names.append(f'log(x{j})')
        basis.append(np.tanh(xj))
        feature_names.append(f'tanh(x{j})')
        basis.append(1.0 / (xj + 1e-12))
        feature_names.append(f'1/x{j}')
        basis.append(np.sqrt(xj + 1e-12))
        feature_names.append(f'sqrt(x{j})')
        basis.append(xj ** 0.5)  # same as sqrt, but let's skip duplicate
        # Actually sqrt and x**0.5 are same, remove one
        basis.pop()
        feature_names.pop()

    # Add pairwise products
    for i in range(d):
        for j in range(i, d):
            basis.append(X[:, i] * X[:, j])
            feature_names.append(f'x{i}*x{j}')

    # Add some interaction terms with transforms (limited to keep size manageable)
    for i in range(d):
        for j in range(d):
            if i != j:
                basis.append(np.sin(X[:, i]) * X[:, j])
                feature_names.append(f'sin(x{i})*x{j}')
                basis.append(X[:, i] * np.cos(X[:, j]))
                feature_names.append(f'x{i}*cos(x{j})')
                basis.append(np.tanh(X[:, i]) * X[:, j])
                feature_names.append(f'tanh(x{i})*x{j}')

    # Convert basis to matrix
    # basis is a list of arrays of length n
    # Stack them as columns
    # But first, check for NaN/inf
    B = np.column_stack(basis)
    # Replace non-finite with 0
    B[~np.isfinite(B)] = 0.0

    # Now do forward feature selection using ridge regression
    # We want to select a subset of features that best predicts y
    # Use a simple greedy forward selection with a max number of features

    max_features = min(50, B.shape[1])  # cap to avoid overfitting and speed
    selected = []
    best_r2 = -np.inf
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        # Constant target
        pred_const = np.full(n, y_mean)
        return lambda X_test: np.full(X_test.shape[0], y_mean)

    # Pre-compute for speed: we'll use normal equations with ridge
    # For each candidate feature, compute its contribution

    # Actually, let's do a simpler approach: 
    # 1. Try all single features + intercept
    # 2. Add features one by one that reduce MSE most
    # Use ridge regression with small lambda for stability

    lambda_ridge = 1e-4

    def fit_ridge(B_sub):
        # B_sub is (n, k)
        # Solve (B^T B + lambda I) w = B^T y
        BtB = B_sub.T @ B_sub
        Bty = B_sub.T @ y
        k = B_sub.shape[1]
        A = BtB + lambda_ridge * np.eye(k)
        try:
            w = np.linalg.solve(A, Bty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(B_sub, y, rcond=None)[0]
        return w

    def predict(B_sub, w):
        return B_sub @ w

    # Forward selection
    remaining = set(range(B.shape[1]))
    # Start with intercept (feature 0)
    selected = [0]
    remaining.discard(0)
    current_B = B[:, selected]
    w = fit_ridge(current_B)
    current_pred = predict(current_B, w)
    current_mse = np.mean((current_pred - y) ** 2)
    best_mse = current_mse

    # Greedy forward selection
    while len(selected) < max_features:
        if time.time() > DEADLINE - 10:
            break
        best_gain = 0
        best_feat = None
        # Try adding each remaining feature
        for f in list(remaining):
            if time.time() > DEADLINE - 10:
                break
            new_selected = selected + [f]
            new_B = B[:, new_selected]
            w_new = fit_ridge(new_B)
            pred_new = predict(new_B, w_new)
            mse_new = np.mean((pred_new - y) ** 2)
            gain = best_mse - mse_new
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_w = w_new
                best_pred = pred_new
                best_mse_new = mse_new
        if best_feat is None or best_gain < 1e-8:
            break
        selected.append(best_feat)
        remaining.discard(best_feat)
        current_B = B[:, selected]
        w = best_w
        current_pred = best_pred
        best_mse = best_mse_new

    # Final fit with selected features
    final_B = B[:, selected]
    final_w = fit_ridge(final_B)

    # Verify
    final_pred = predict(final_B, final_w)
    final_mse = np.mean((final_pred - y) ** 2)

    # If MSE is still high, try a simple polynomial fallback
    # But first, let's return the current best
    # We need to return a predict function that works on new X
    # So we need to reconstruct the features from X_test

    def make_predict(selected_indices, w):
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            n_test = X_test.shape[0]
            d_test = X_test.shape[1]
            # Rebuild basis for X_test
            test_basis = [np.ones(n_test)]
            for j in range(d_test):
                xj = X_test[:, j]
                for p in range(1, 7):
                    test_basis.append(xj ** p)
                test_basis.append(np.sin(xj))
                test_basis.append(np.cos(xj))
                test_basis.append(np.exp(np.clip(xj, -10, 10)))
                test_basis.append(np.log(xj + 1e-12))
                test_basis.append(np.tanh(xj))
                test_basis.append(1.0 / (xj + 1e-12))
                test_basis.append(np.sqrt(xj + 1e-12))
            for i in range(d_test):
                for j in range(i, d_test):
                    test_basis.append(X_test[:, i] * X_test[:, j])
            for i in range(d_test):
                for j in range(d_test):
                    if i != j:
                        test_basis.append(np.sin(X_test[:, i]) * X_test[:, j])
                        test_basis.append(X_test[:, i] * np.cos(X_test[:, j]))
                        test_basis.append(np.tanh(X_test[:, i]) * X_test[:, j])
            B_test = np.column_stack(test_basis)
            B_test[~np.isfinite(B_test)] = 0.0
            B_test_sub = B_test[:, selected_indices]
            return B_test_sub @ w
        return predict

    predict_fn = make_predict(selected, final_w)
    return predict_fn
# EVOLVE-BLOCK-END

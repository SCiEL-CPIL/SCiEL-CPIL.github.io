# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin for 330s kill

    n, d = X_train.shape

    def check_time():
        return time.time() < deadline

    best_predict = None
    best_val_mse = np.inf

    def evaluate_model(model):
        # Use last 20% as validation set
        if n >= 50:
            val_idx = np.arange(int(n * 0.8), n)
            tr_idx = np.arange(0, int(n * 0.8))
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
        else:
            X_val = X_train
            y_val = y_train
        try:
            preds = model(X_val)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y_val) ** 2)
            return mse
        except Exception:
            return np.inf

    # Helper to build polynomial features up to given degree
    def poly_features(X, degree):
        d = X.shape[1]
        cols = [np.ones(len(X))]
        # Degree 1
        for j in range(d):
            cols.append(X[:, j])
        # Degree 2
        if degree >= 2:
            for i in range(d):
                for j in range(i, d):
                    cols.append(X[:, i] * X[:, j])
        # Degree 3
        if degree >= 3:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        # Degree 4 (only for d <= 2 to avoid explosion)
        if degree >= 4 and d <= 2:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        for l in range(k, d):
                            cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
        return np.column_stack(cols)

    def make_poly_predictor(F, w):
        return lambda X: poly_features(X, 4 if d <= 2 else 3) @ w

    # Candidate 1: Polynomial regression with ridge regularization
    def fit_poly(degree, lam=1e-4):
        F = poly_features(X_train, degree)
        A = F.T @ F
        b = F.T @ y_train
        A_reg = A.copy()
        A_reg[1:, 1:] += lam
        try:
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        return make_poly_predictor(F, w)

    # Candidate 2: Fourier features
    def fit_fourier(max_freq=5):
        cols = [np.ones(len(X_train))]
        for j in range(d):
            for k in range(1, max_freq + 1):
                cols.append(np.sin(k * X_train[:, j]))
                cols.append(np.cos(k * X_train[:, j]))
        F = np.column_stack(cols)
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        
        def predict(X):
            cols = [np.ones(len(X))]
            for j in range(d):
                for k in range(1, max_freq + 1):
                    cols.append(np.sin(k * X[:, j]))
                    cols.append(np.cos(k * X[:, j]))
            return np.column_stack(cols) @ w
        return predict

    # Candidate 3: RBF kernel regression
    def fit_rbf(gamma, n_centers=50):
        if n > n_centers:
            idx = np.linspace(0, n - 1, n_centers, dtype=int)
            centers = X_train[idx]
        else:
            centers = X_train
        F = np.zeros((n, len(centers)))
        for i, c in enumerate(centers):
            F[:, i] = np.exp(-gamma * np.sum((X_train - c) ** 2, axis=1))
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        
        def predict(X):
            F_test = np.zeros((X.shape[0], len(centers)))
            for i, c in enumerate(centers):
                F_test[:, i] = np.exp(-gamma * np.sum((X - c) ** 2, axis=1))
            return F_test @ w
        return predict

    # Candidate 4: Neural network-like with tanh activation (simple linear model on transformed features)
    def fit_tanh(degree=2, n_neurons=10):
        # Create random projections
        rng = np.random.RandomState(42)
        W1 = rng.randn(d, n_neurons) * 0.5
        b1 = rng.randn(n_neurons) * 0.1
        cols = [np.ones(len(X_train))]
        for _ in range(n_neurons):
            # Add tanh of linear combinations
            for i in range(d):
                for j in range(d):
                    if i <= j:
                        z = X_train[:, i] * X_train[:, j]
                        cols.append(np.tanh(0.5 * z))
        F = np.column_stack(cols)
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        
        def predict(X):
            cols = [np.ones(len(X))]
            for i in range(d):
                for j in range(d):
                    if i <= j:
                        z = X[:, i] * X[:, j]
                        cols.append(np.tanh(0.5 * z))
            return np.column_stack(cols) @ w
        return predict

    # Try various candidates
    candidates = []

    # Polynomials of different degrees
    for deg in [2, 3]:
        if d <= 4:
            candidates.append(("poly" + str(deg), fit_poly(deg)))
        elif deg == 2:
            candidates.append(("poly2", fit_poly(2)))

    # Higher degree for low dimensions
    if d <= 2:
        candidates.append(("poly4", fit_poly(4)))

    # Fourier with different frequencies
    for freq in [3, 5]:
        candidates.append(("fourier" + str(freq), fit_fourier(freq)))

    # RBF with different gammas
    for gamma in [0.1, 0.5, 1.0, 5.0]:
        candidates.append(("rbf_" + str(gamma), fit_rbf(gamma)))

    # Tanh features
    if d <= 3:
        candidates.append(("tanh", fit_tanh()))

    # Evaluate all candidates
    for name, model in candidates:
        if not check_time():
            break
        try:
            mse = evaluate_model(model)
            if mse < best_val_mse:
                best_val_mse = mse
                best_predict = model
        except Exception:
            pass

    # If no model found, fall back to mean prediction
    if best_predict is None:
        y_mean = np.mean(y_train)
        return lambda X: np.full(X.shape[0], y_mean)

    return best_predict

# EVOLVE-BLOCK-END

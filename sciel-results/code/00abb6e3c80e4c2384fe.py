# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # 50s margin for 330s kill

    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)

    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)

    def check_time():
        return time.time() < deadline

    # Normalize inputs to [0, 1]
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X_train - X_min) / X_range

    def denormalize_predict(predict_norm):
        def predict(X):
            Xn = (X - X_min) / X_range
            return predict_norm(Xn)
        return predict

    def safe_evaluate(model, X, y):
        try:
            preds = model(X)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf

    best_predict = None
    best_val_mse = np.inf

    def update_best(model):
        nonlocal best_predict, best_val_mse
        if model is None:
            return
        mse = safe_evaluate(model, X_norm, y_train)
        if mse < best_val_mse:
            best_val_mse = mse
            best_predict = model

    # --- Model 1: Polynomial Regression ---
    def make_poly_model(degree):
        def features(Xn):
            cols = [np.ones(len(Xn))]
            for j in range(Xn.shape[1]):
                cols.append(Xn[:, j])
            for i in range(Xn.shape[1]):
                for j in range(i, Xn.shape[1]):
                    cols.append(Xn[:, i] * Xn[:, j])
            if degree >= 3:
                for i in range(Xn.shape[1]):
                    for j in range(i, Xn.shape[1]):
                        for k in range(j, Xn.shape[1]):
                            cols.append(Xn[:, i] * Xn[:, j] * Xn[:, k])
            if degree >= 4:
                for i in range(Xn.shape[1]):
                    for j in range(i, Xn.shape[1]):
                        for k in range(j, Xn.shape[1]):
                            for l in range(k, Xn.shape[1]):
                                cols.append(Xn[:, i] * Xn[:, j] * Xn[:, k] * Xn[:, l])
            return np.column_stack(cols)

        F_train = features(X_norm)
        try:
            lam = 1e-5
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)

        def predict_norm(Xn):
            return features(Xn) @ w
        return predict_norm

    # --- Model 2: Fourier Regression ---
    def make_fourier_model(max_freq):
        def features(Xn):
            cols = [np.ones(len(Xn))]
            for j in range(Xn.shape[1]):
                for k in range(1, max_freq + 1):
                    cols.append(np.sin(k * Xn[:, j] * np.pi))
                    cols.append(np.cos(k * Xn[:, j] * np.pi))
            if Xn.shape[1] >= 2:
                for i in range(Xn.shape[1]):
                    for j in range(i + 1, Xn.shape[1]):
                        cols.append(np.sin(Xn[:, i] * Xn[:, j] * np.pi))
                        cols.append(np.cos(Xn[:, i] * Xn[:, j] * np.pi))
            return np.column_stack(cols)

        F_train = features(X_norm)
        try:
            lam = 1e-5
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)

        def predict_norm(Xn):
            return features(Xn) @ w
        return predict_norm

    # --- Model 3: RBF Regression ---
    def make_rbf_model(gamma, n_centers=50):
        if n >= n_centers:
            idx = np.linspace(0, n - 1, n_centers, dtype=int)
            centers = X_norm[idx]
        else:
            centers = X_norm

        diffs = X_norm[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        F_train = np.exp(-gamma * dists)

        try:
            lam = 1e-5
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)

        def predict_norm(Xn):
            diffs = Xn[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=1)
            F_test = np.exp(-gamma * dists)
            return F_test @ w
        return predict_norm

    # --- Model 4: Sum-of-Sines Kernel Ridge Regression ---
    def make_sos_krr_model(frequencies, lam=1e-4):
        """
        Kernel: K(x, x') = sum_j (a_j * sin(w_j * x + b_j))
        We use a fixed set of frequencies and optimize amplitudes via ridge regression.
        """
        # Precompute features for training data
        def compute_features(Xn):
            cols = [np.ones(len(Xn))]
            for w, b in frequencies:
                cols.append(np.sin(w * Xn + b))
            return np.column_stack(cols)

        F_train = compute_features(X_norm)
        try:
            A = F_train.T @ F_train
            b_vec = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b_vec)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)

        def predict_norm(Xn):
            return compute_features(Xn) @ w
        return predict_norm

    # Generate a set of frequencies for Sum-of-Sines
    # Use a grid of frequencies from 0.5 to 10, with phases 0 and pi/2
    freq_list = []
    for w in [0.5, 1, 2, 3, 4, 5, 6, 8, 10]:
        for b in [0, np.pi / 2, np.pi]:
            for dim in range(d):
                # Create a function that applies to the specific dimension
                freq_list.append((w, b, dim))

    def make_sos_krr_model_v2(freq_list, lam=1e-4):
        def compute_features(Xn):
            cols = [np.ones(len(Xn))]
            for w, b, dim in freq_list:
                cols.append(np.sin(w * Xn[:, dim] + b))
            return np.column_stack(cols)

        F_train = compute_features(X_norm)
        try:
            A = F_train.T @ F_train
            b_vec = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b_vec)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)

        def predict_norm(Xn):
            return compute_features(Xn) @ w
        return predict_norm

    # --- Model 5: Neural Network (simple MLP) ---
    def make_mlp_model(n_hidden=20, lr=0.01, epochs=100):
        # Simple gradient descent MLP
        np.random.seed(42)
        W1 = np.random.randn(d, n_hidden) * 0.1
        b1 = np.zeros(n_hidden)
        W2 = np.random.randn(n_hidden, 1) * 0.1
        b2 = np.zeros(1)

        X = X_norm
        y = y_train.reshape(-1, 1)

        for epoch in range(epochs):
            if not check_time():
                break
            # Forward
            z1 = X @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            pred = z2
            error = pred - y
            dW2 = a1.T @ error / n
            db2 = np.mean(error, axis=0)
            da1 = error @ W2.T
            dz1 = da1 * (1 - a1 ** 2)
            dW1 = X.T @ dz1 / n
            db1 = np.mean(dz1, axis=0)

            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        def predict_norm(Xn):
            z1 = Xn @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            return z2.flatten()
        return predict_norm

    # --- Strategy ---

    # Polynomials
    for deg in [1, 2, 3, 4]:
        if not check_time():
            break
        model = make_poly_model(deg)
        update_best(model)

    # Fourier
    for freq in [3, 5, 8]:
        if not check_time():
            break
        model = make_fourier_model(freq)
        update_best(model)

    # RBF
    for g in [1, 5, 10, 20]:
        if not check_time():
            break
        model = make_rbf_model(g, n_centers=50)
        update_best(model)

    # Sum-of-Sines KRR
    if check_time():
        model = make_sos_krr_model_v2(freq_list, lam=1e-4)
        update_best(model)

    # MLP
    if check_time():
        model = make_mlp_model(n_hidden=20, lr=0.01, epochs=50)
        update_best(model)

    # If we have time, try higher-degree polynomials for 1D case
    if check_time() and d == 1:
        for deg in [5, 6, 8]:
            if not check_time():
                break
            model = make_poly_model(deg)
            update_best(model)

    # Fallback
    if best_predict is None:
        return lambda X: np.full(X.shape[0], y_mean)

    return denormalize_predict(best_predict)

# EVOLVE-BLOCK-END

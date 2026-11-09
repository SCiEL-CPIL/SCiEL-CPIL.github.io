# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin for 330s kill

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

    # --- Model 4: Neural Network (simple MLP) with multiple architectures ---
    def make_mlp_model(n_hidden1, n_hidden2, lr=0.01, epochs=200, seed=42):
        np.random.seed(seed)
        
        # Layer 1
        W1 = np.random.randn(d, n_hidden1) * np.sqrt(2.0 / d)
        b1 = np.zeros(n_hidden1)
        
        # Layer 2
        W2 = np.random.randn(n_hidden1, n_hidden2) * np.sqrt(2.0 / n_hidden1)
        b2 = np.zeros(n_hidden2)
        
        # Output layer
        W3 = np.random.randn(n_hidden2, 1) * np.sqrt(2.0 / n_hidden2)
        b3 = np.zeros(1)

        X = X_norm
        y = y_train.reshape(-1, 1)

        # Use Adam-like optimizer with momentum
        v_W1 = np.zeros_like(W1)
        v_b1 = np.zeros_like(b1)
        v_W2 = np.zeros_like(W2)
        v_b2 = np.zeros_like(b2)
        v_W3 = np.zeros_like(W3)
        v_b3 = np.zeros_like(b3)
        
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 0

        for epoch in range(epochs):
            if not check_time():
                break
            t += 1
            
            # Forward pass
            z1 = X @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            a2 = np.tanh(z2)
            z3 = a2 @ W3 + b3
            pred = z3
            
            # Loss
            error = pred - y
            mse = np.mean(error ** 2)
            
            # Backward pass
            dW3 = a2.T @ error / n
            db3 = np.mean(error, axis=0)
            da2 = error @ W3.T
            dz2 = da2 * (1 - a2 ** 2)
            dW2 = a1.T @ dz2 / n
            db2 = np.mean(dz2, axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (1 - a1 ** 2)
            dW1 = X.T @ dz1 / n
            db1 = np.mean(dz1, axis=0)
            
            # Adam update
            m_W1 = beta1 * v_W1 + (1 - beta1) * dW1
            v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1 ** 2)
            W1 -= lr * m_W1 / (np.sqrt(v_W1) + eps)
            
            m_b1 = beta1 * v_b1 + (1 - beta1) * db1
            v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
            b1 -= lr * m_b1 / (np.sqrt(v_b1) + eps)
            
            m_W2 = beta1 * v_W2 + (1 - beta1) * dW2
            v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2 ** 2)
            W2 -= lr * m_W2 / (np.sqrt(v_W2) + eps)
            
            m_b2 = beta1 * v_b2 + (1 - beta1) * db2
            v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
            b2 -= lr * m_b2 / (np.sqrt(v_b2) + eps)
            
            m_W3 = beta1 * v_W3 + (1 - beta1) * dW3
            v_W3 = beta2 * v_W3 + (1 - beta2) * (dW3 ** 2)
            W3 -= lr * m_W3 / (np.sqrt(v_W3) + eps)
            
            m_b3 = beta1 * v_b3 + (1 - beta1) * db3
            v_b3 = beta2 * v_b3 + (1 - beta2) * (db3 ** 2)
            b3 -= lr * m_b3 / (np.sqrt(v_b3) + eps)

        def predict_norm(Xn):
            z1 = Xn @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            a2 = np.tanh(z2)
            z3 = a2 @ W3 + b3
            return z3.flatten()
        return predict_norm

    # --- Model 5: Gaussian Process Regression (using RBF kernel) ---
    def make_gp_model(length_scale, sigma_f, n_centers=50):
        if n >= n_centers:
            idx = np.linspace(0, n - 1, n_centers, dtype=int)
            centers = X_norm[idx]
        else:
            centers = X_norm

        # Compute kernel matrix
        diffs = X_norm[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        K = sigma_f ** 2 * np.exp(-0.5 * dists / (length_scale ** 2))
        
        # Add noise for numerical stability
        K += 1e-6 * np.eye(K.shape[0])
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            # Fallback to ridge regression
            K_reg = K + 1e-4 * np.eye(K.shape[0])
            alpha = np.linalg.solve(K_reg, y_train)

        def predict_norm(Xn):
            diffs = Xn[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=1)
            K_star = sigma_f ** 2 * np.exp(-0.5 * dists / (length_scale ** 2))
            return K_star @ alpha
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
    for g in [1, 5, 10, 20, 50]:
        if not check_time():
            break
        model = make_rbf_model(g, n_centers=50)
        update_best(model)

    # Gaussian Process with different hyperparameters
    if check_time():
        for ls in [0.1, 0.5, 1.0, 2.0]:
            for sf in [0.5, 1.0, 2.0]:
                if not check_time():
                    break
                model = make_gp_model(ls, sf, n_centers=50)
                update_best(model)

    # MLP with different architectures
    if check_time():
        for nh1, nh2 in [(10, 10), (20, 20), (30, 15), (50, 25)]:
            if not check_time():
                break
            model = make_mlp_model(nh1, nh2, lr=0.01, epochs=150, seed=42)
            update_best(model)
            # Try a different seed
            if check_time():
                model = make_mlp_model(nh1, nh2, lr=0.01, epochs=150, seed=123)
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

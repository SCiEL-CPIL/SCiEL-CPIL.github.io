import numpy as np
import time
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

START_TIME = time.time()
TIME_LIMIT = 320.0

def check_time():
    return (time.time() - START_TIME) > TIME_LIMIT

def generate_features(X):
    n, d = X.shape
    features = [np.ones(n)]
    
    # Polynomials up to degree 4
    if d == 1:
        for p in range(1, 5):
            features.append(X[:, 0] ** p)
    elif d == 2:
        for p1 in range(1, 5):
            for p2 in range(0, 5 - p1):
                if p1 == 0 and p2 == 0: continue
                features.append((X[:, 0] ** p1) * (X[:, 1] ** p2))
    elif d == 3:
        for p1 in range(1, 5):
            for p2 in range(0, 5 - p1):
                for p3 in range(0, 5 - p1 - p2):
                    if p1 == 0 and p2 == 0 and p3 == 0: continue
                    features.append((X[:, 0] ** p1) * (X[:, 1] ** p2) * (X[:, 2] ** p3))
    
    # Nonlinear functions
    funcs = [np.sin, np.cos, np.tanh, np.exp]
    for i in range(d):
        for f in funcs:
            features.append(f(X[:, i]))
            
    # Interactions
    for i in range(d):
        for j in range(d):
            if i == j: continue
            features.append(X[:, i] * np.sin(X[:, j]))
            features.append(X[:, i] * np.cos(X[:, j]))

    Phi = np.column_stack(features)
    mask = np.all(np.isfinite(Phi), axis=0)
    Phi = Phi[:, mask]
    return Phi

def fit(X_train, y_train):
    if check_time():
        return lambda X: np.full(X.shape[0], np.mean(y_train))

    scaler_X = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train)
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-8:
        y_std = 1.0
    y_train_norm = (y_train - y_mean) / y_std

    try:
        Phi_train = generate_features(X_train_norm)
    except Exception:
        # Fallback
        def predict_poly(X):
            Xs = scaler_X.transform(X)
            cols = [np.ones(len(Xs))]
            for j in range(Xs.shape[1]):
                cols.append(Xs[:, j])
            for i in range(Xs.shape[1]):
                for j in range(i, Xs.shape[1]):
                    cols.append(Xs[:, i] * Xs[:, j])
            P = np.column_stack(cols)
            w, *_ = np.linalg.lstsq(P, y_train, rcond=None)
            return P @ w
        return predict_poly

    # Validation Split
    n_samples = len(X_train_norm)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.7 * n_samples)
    val_idx = indices[:split]
    test_idx = indices[split:]
    
    X_val = X_train_norm[val_idx]
    y_val = y_train_norm[val_idx]
    X_test = X_train_norm[test_idx]
    y_test = y_train_norm[test_idx]
    
    Phi_val = generate_features(X_val)
    Phi_test = generate_features(X_test)
    
    if Phi_val.shape[1] == 0:
         return lambda X: np.full(X.shape[0], y_mean)

    best_score = np.inf
    best_model_type = 'lasso'
    best_params = {}
    
    # Try Lasso
    alphas = np.logspace(-5, 0, 15)
    for alpha in alphas:
        if check_time(): break
        try:
            model = Lasso(alpha=alpha, max_iter=1000, tol=1e-4)
            model.fit(Phi_val, y_val)
            y_pred = model.predict(Phi_test)
            score = np.mean((y_pred - y_test) ** 2)
            if score < best_score:
                best_score = score
                best_model_type = 'lasso'
                best_params = {'alpha': alpha}
        except Exception:
            pass
            
    # Try Ridge
    alphas_ridge = np.logspace(-5, 2, 15)
    for alpha in alphas_ridge:
        if check_time(): break
        try:
            model = Ridge(alpha=alpha)
            model.fit(Phi_val, y_val)
            y_pred = model.predict(Phi_test)
            score = np.mean((y_pred - y_test) ** 2)
            if score < best_score:
                best_score = score
                best_model_type = 'ridge'
                best_params = {'alpha': alpha}
        except Exception:
            pass

    # Final Fit
    if best_model_type == 'lasso':
        final_model = Lasso(alpha=best_params['alpha'], max_iter=1000, tol=1e-4)
    else:
        final_model = Ridge(alpha=best_params['alpha'])
        
    final_model.fit(Phi_train, y_train_norm)

    def predict(X):
        if check_time():
             return np.full(X.shape[0], y_mean)
        Xs = scaler_X.transform(X)
        try:
            P = generate_features(Xs)
        except Exception:
            P = np.column_stack([np.ones(Xs.shape[0]), Xs])
        y_pred_norm = final_model.predict(P)
        y_pred = y_pred_norm * y_std + y_mean
        return np.where(np.isfinite(y_pred), y_pred, y_mean)

    return predict

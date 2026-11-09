import numpy as np
import time

def fit(X_train, y_train):
    t_start = time.time()
    deadline = t_start + 310.0  # 20s margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Handle constant y
    y_var = np.var(y)
    if y_var < 1e-12:
        y_mean = np.mean(y)
        return lambda X: np.full(len(X), y_mean)
    
    # Standardize X
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-10] = 1.0
    X_std = (X - mu) / sigma
    
    def standardize(X_new):
        return (np.asarray(X_new, dtype=np.float64) - mu) / sigma
    
    def matern_kernel(X1, X2, l):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        np.maximum(sq_dists, 0, out=sq_dists)
        r = np.sqrt(sq_dists) / l
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    
    # Strategy: Ensemble of diverse models, pick best via CV
    
    def safe_evaluate(model, X_eval, y_eval):
        try:
            preds = model(X_eval)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y_eval.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y_eval)**2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf
    
    # CV setup
    n_folds = 5
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    def cv_score(model_factory):
        """Evaluate a model factory via K-fold CV. Returns mean CV MSE."""
        fold_errors = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            try:
                model = model_factory(X_tr, y_tr)
                err = safe_evaluate(model, X_val, y_val)
                if np.isinf(err):
                    return np.inf
                fold_errors.append(err)
            except Exception:
                return np.inf
        
        return np.mean(fold_errors)
    
    # Model factories
    
    # 1. GP with Matern kernel
    def make_gp_factory(l, alpha):
        def factory(X_tr, y_tr):
            X_tr_std = (X_tr - mu) / sigma
            K = matern_kernel(X_tr_std, X_tr_std, l)
            K_reg = K + alpha * np.eye(len(X_tr))
            try:
                w = np.linalg.solve(K_reg, y_tr)
            except:
                w, *_ = np.linalg.lstsq(K_reg, y_tr, rcond=None)
            
            def predict(X_test):
                X_test_std = (np.asarray(X_test, dtype=np.float64) - mu) / sigma
                K_test = matern_kernel(X_test_std, X_tr_std, l)
                return K_test @ w
            return predict
        return factory
    
    # 2. Polynomial regression
    def make_poly_factory(degree):
        def features(X):
            X = np.asarray(X, dtype=np.float64)
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            if degree >= 3:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        for k in range(j, X.shape[1]):
                            cols.append(X[:, i] * X[:, j] * X[:, k])
            if degree >= 4:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        for k in range(j, X.shape[1]):
                            for l in range(k, X.shape[1]):
                                cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
            return np.column_stack(cols)
        
        def factory(X_tr, y_tr):
            F = features(X_tr)
            lam = 1e-4
            A = F.T @ F
            b = F.T @ y_tr
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            try:
                w = np.linalg.solve(A_reg, b)
            except:
                w, *_ = np.linalg.lstsq(F, y_tr, rcond=None)
            
            def predict(X_test):
                F_test = features(X_test)
                return F_test @ w
            return predict
        return factory
    
    # 3. RBF regression
    def make_rbf_factory(gamma, n_centers=50):
        def factory(X_tr, y_tr):
            if len(X_tr) > n_centers:
                idx = np.linspace(0, len(X_tr) - 1, n_centers, dtype=int)
                centers = X_tr[idx]
            else:
                centers = X_tr
            
            diffs = X_tr[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs**2, axis=2)
            F = np.exp(-gamma * dists)
            
            lam = 1e-4
            A = F.T @ F
            b = F.T @ y_tr
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            try:
                w = np.linalg.solve(A_reg, b)
            except:
                w, *_ = np.linalg.lstsq(F, y_tr, rcond=None)
            
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                diffs = X_test[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists = np.sum(diffs**2, axis=2)
                F_test = np.exp(-gamma * dists)
                return F_test @ w
            return predict
        return factory
    
    # 4. Fourier regression
    def make_fourier_factory(max_freq):
        def features(X):
            X = np.asarray(X, dtype=np.float64)
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                for k in range(1, max_freq + 1):
                    cols.append(np.sin(k * X[:, j]))
                    cols.append(np.cos(k * X[:, j]))
            if X.shape[1] >= 2:
                for i in range(X.shape[1]):
                    for j in range(i + 1, X.shape[1]):
                        cols.append(np.sin(X[:, i] + X[:, j]))
                        cols.append(np.cos(X[:, i] + X[:, j]))
                        cols.append(np.sin(X[:, i] - X[:, j]))
                        cols.append(np.cos(X[:, i] - X[:, j]))
            return np.column_stack(cols)
        
        def factory(X_tr, y_tr):
            F = features(X_tr)
            lam = 1e-4
            A = F.T @ F
            b = F.T @ y_tr
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            try:
                w = np.linalg.solve(A_reg, b)
            except:
                w, *_ = np.linalg.lstsq(F, y_tr, rcond=None)
            
            def predict(X_test):
                F_test = features(X_test)
                return F_test @ w
            return predict
        return factory
    
    # 5. Chebyshev polynomial
    def make_cheb_factory(degree):
        def features(X):
            X = np.asarray(X, dtype=np.float64)
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                x_min = X[:, j].min()
                x_max = X[:, j].max()
                if x_max > x_min:
                    x_norm = (2 * X[:, j] - (x_max + x_min)) / (x_max - x_min)
                else:
                    x_norm = np.zeros_like(X[:, j])
                
                T0 = np.ones_like(x_norm)
                T1 = x_norm
                cols.append(T1)
                for k in range(2, degree + 1):
                    Tk = 2 * x_norm * T1 - T0
                    cols.append(Tk)
                    T0 = T1
                    T1 = Tk
            return np.column_stack(cols)
        
        def factory(X_tr, y_tr):
            F = features(X_tr)
            lam = 1e-4
            A = F.T @ F
            b = F.T @ y_tr
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            try:
                w = np.linalg.solve(A_reg, b)
            except:
                w, *_ = np.linalg.lstsq(F, y_tr, rcond=None)
            
            def predict(X_test):
                F_test = features(X_test)
                return F_test @ w
            return predict
        return factory
    
    # Candidate models to evaluate
    candidates = []
    
    # GP with different length scales
    for l in [0.3, 0.5, 1.0, 2.0, 4.0]:
        candidates.append(('gp_l%.1f' % l, make_gp_factory(l, 1e-6)))
    
    # Polynomials
    for deg in [1, 2, 3]:
        if d <= 3:
            candidates.append(('poly_d%d' % deg, make_poly_factory(deg)))
    
    # RBF with different gammas
    X_range = np.ptp(X, axis=0)
    avg_range = np.mean(X_range)
    if avg_range == 0:
        avg_range = 1.0
    base_gamma = 1.0 / (d * avg_range**2)
    for mult in [0.1, 0.5, 1.0, 2.0, 5.0]:
        candidates.append(('rbf_g%.2f' % (base_gamma * mult), make_rbf_factory(base_gamma * mult, 50)))
    
    # Fourier
    for freq in [2, 4, 6]:
        if d <= 3:
            candidates.append(('fourier_f%d' % freq, make_fourier_factory(freq)))
    
    # Chebyshev
    for deg in [3, 5, 7]:
        if d <= 3:
            candidates.append(('cheb_d%d' % deg, make_cheb_factory(deg)))
    
    # Evaluate candidates
    best_factory = None
    best_cv_err = np.inf
    
    for name, factory in candidates:
        if time.time() > deadline:
            break
        try:
            err = cv_score(factory)
            if err < best_cv_err:
                best_cv_err = err
                best_factory = factory
        except Exception:
            pass
    
    # If no valid model found, fall back to simple GP
    if best_factory is None:
        best_factory = make_gp_factory(1.0, 1e-6)
    
    # Final fit with best factory on all data
    try:
        predict = best_factory(X, y)
    except Exception:
        # Fallback
        predict = lambda X: np.full(len(X), np.mean(y))
    
    # Verify predict works
    try:
        test_pred = predict(X[:1])
        if np.any(np.isnan(test_pred)) or np.any(np.isinf(test_pred)):
            predict = lambda X: np.full(len(X), np.mean(y))
    except Exception:
        predict = lambda X: np.full(len(X), np.mean(y))
    
    return predict

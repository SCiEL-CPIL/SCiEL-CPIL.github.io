import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 320.0
    
    d = X_train.shape[1]
    n = len(X_train)
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    best_score = np.inf
    best_predictor = None
    
    def update_best(score, predictor):
        nonlocal best_score, best_predictor
        if score < best_score:
            best_score = score
            best_predictor = predictor
    
    # Strategy 1: Polynomial regression (degrees 2-6)
    for deg in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 5:
            break
        
        def poly_features(X, d=d, deg=deg):
            cols = [np.ones(len(X))]
            for total_deg in range(1, deg + 1):
                for indices in product(range(d), repeat=total_deg):
                    if sorted(indices) != indices:
                        continue
                    feat = np.ones(len(X))
                    for idx in indices:
                        feat = feat * X[:, idx]
                    cols.append(feat)
            return np.column_stack(cols)
        
        try:
            F = poly_features(X_train)
            n_feat = F.shape[1]
            if n_feat >= n:
                # Use ridge if features >= samples
                A = F.T @ F + 1.0 * np.eye(n_feat)
                b = F.T @ y_train
                w = np.linalg.solve(A, b)
            else:
                w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            pred = F @ w
            score = np.mean((pred - y_train)**2) / y_var
            update_best(score, (lambda X, F_fn=poly_features, w=w: F_fn(X) @ w))
        except:
            pass
    
    # Strategy 2: GP Regression
    if time.time() < deadline - 10:
        try:
            from scipy.spatial.distance import cdist
            
            def gp_predict(X_pred, X_tr, y_tr, length_scale, sigma_f, noise):
                K_tr = sigma_f**2 * np.exp(-cdist(X_tr, X_tr)**2 / (2 * length_scale**2)) + noise * np.eye(len(X_tr))
                K_pred = sigma_f**2 * np.exp(-cdist(X_pred, X_tr)**2 / (2 * length_scale**2))
                L = np.linalg.cholesky(K_tr)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr - y_mean))
                mu = y_mean + K_pred @ alpha
                return mu
            
            for ls in [0.3, 0.5, 1.0, 2.0, 4.0, 8.0]:
                for sf in [0.5, 1.0, 5.0, 10.0]:
                    if time.time() > deadline - 5:
                        break
                    try:
                        pred_tr = gp_predict(X_train, X_train, y_train, ls, sf, 1e-6)
                        score = np.mean((pred_tr - y_train)**2) / y_var
                        if score < best_score:
                            X_tr_copy = X_train.copy()
                            y_tr_copy = y_train.copy()
                            ls_copy, sf_copy = ls, sf
                            update_best(score, (lambda X, X_t=X_tr_copy, y_t=y_tr_copy, l=ls_copy, s=sf_copy: gp_predict(X, X_t, y_t, l, s, 1e-6)))
                    except:
                        pass
        except ImportError:
            pass
    
    # Strategy 3: Symbolic expression search with diverse features
    if time.time() < deadline - 15:
        base_funcs = [
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3,
            np.sin,
            np.cos,
            np.tanh,
            lambda x: np.exp(-x**2 / 5),
            np.log1p,
            lambda x: 1.0 / (1.0 + np.exp(-2*x)),
            lambda x: x * np.sin(x),
            lambda x: x * np.cos(x),
        ]
        
        candidate_features = []
        
        candidate_features.append(np.ones(n))
        
        for j in range(d):
            for fi, f in enumerate(base_funcs):
                try:
                    feat = f(X_train[:, j])
                    if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                        candidate_features.append(feat)
                except:
                    pass
        
        if d >= 2:
            for i in range(d):
                for j in range(i+1, d):
                    for fi, f1 in enumerate(base_funcs[:5]):
                        for fj, f2 in enumerate(base_funcs[:5]):
                            try:
                                feat1 = f1(X_train[:, i])
                                feat2 = f2(X_train[:, j])
                                feat = feat1 * feat2
                                if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                                    candidate_features.append(feat)
                            except:
                                pass
                    
                    for fi, f1 in enumerate(base_funcs[:3]):
                        for fj, f2 in enumerate(base_funcs[:3]):
                            try:
                                feat1 = f1(X_train[:, i])
                                feat2 = f2(X_train[:, j])
                                feat = feat1 + feat2
                                if np.all(np.isfinite(feat)) and np.std(feat) > 1e-10:
                                    candidate_features.append(feat)
                            except:
                                pass
        
        if len(candidate_features) > 0:
            F_cand = np.column_stack(candidate_features)
            n_feat = F_cand.shape[1]
            
            for alpha in [0, 0.001, 0.01, 0.1, 1.0, 10.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    if alpha == 0 and n_feat < n:
                        w, *_ = np.linalg.lstsq(F_cand, y_train, rcond=None)
                    else:
                        A = F_cand.T @ F_cand + max(alpha, 1e-6) * np.eye(n_feat)
                        b = F_cand.T @ y_train
                        w = np.linalg.solve(A, b)
                    pred = F_cand @ w
                    score = np.mean((pred - y_train)**2) / y_var
                    if score < best_score:
                        w_copy = w.copy()
                        d_copy = d
                        
                        def sym_predict(X, w_c=w_copy, d_c=d_copy):
                            n_new = len(X)
                            new_feats = [np.ones(n_new)]
                            for j in range(d_c):
                                for fi, f in enumerate(base_funcs):
                                    try:
                                        feat = f(X[:, j])
                                        if np.all(np.isfinite(feat)):
                                            new_feats.append(feat)
                                        else:
                                            new_feats.append(np.zeros(n_new))
                                    except:
                                        new_feats.append(np.zeros(n_new))
                            if d_c >= 2:
                                for i in range(d_c):
                                    for j in range(i+1, d_c):
                                        for fi, f1 in enumerate(base_funcs[:5]):
                                            for fj, f2 in enumerate(base_funcs[:5]):
                                                try:
                                                    feat1 = f1(X[:, i])
                                                    feat2 = f2(X[:, j])
                                                    feat = feat1 * feat2
                                                    if np.all(np.isfinite(feat)):
                                                        new_feats.append(feat)
                                                    else:
                                                        new_feats.append(np.zeros(n_new))
                                                except:
                                                    new_feats.append(np.zeros(n_new))
                                        for fi, f1 in enumerate(base_funcs[:3]):
                                            for fj, f2 in enumerate(base_funcs[:3]):
                                                try:
                                                    feat1 = f1(X[:, i])
                                                    feat2 = f2(X[:, j])
                                                    feat = feat1 + feat2
                                                    if np.all(np.isfinite(feat)):
                                                        new_feats.append(feat)
                                                    else:
                                                        new_feats.append(np.zeros(n_new))
                                                except:
                                                    new_feats.append(np.zeros(n_new))
                            F_new = np.column_stack(new_feats)
                            return F_new @ w_c
                        
                        update_best(score, sym_predict)
                except:
                    pass
    
    # Strategy 4: Small neural network
    if time.time() < deadline - 20:
        try:
            from sklearn.neural_network import MLPRegressor
            for hidden in [(16,), (32,), (16, 16)]:
                if time.time() > deadline - 5:
                    break
                try:
                    mlp = MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000, random_state=42, 
                                        early_stopping=False, activation='relu', solver='adam',
                                        learning_rate_init=0.01)
                    mlp.fit(X_train, y_train)
                    pred = mlp.predict(X_train)
                    score = np.mean((pred - y_train)**2) / y_var
                    update_best(score, mlp.predict)
                except:
                    pass
        except ImportError:
            pass
    
    # Fallback: constant predictor
    if best_predictor is None:
        best_predictor = lambda X: np.full(len(X), y_mean)
    
    return best_predictor

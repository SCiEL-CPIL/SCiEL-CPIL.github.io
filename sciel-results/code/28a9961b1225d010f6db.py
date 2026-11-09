# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    def check_time():
        return time.time() < deadline
    
    # Normalize features for numerical stability
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    
    # Normalize y
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
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
    best_mse = np.inf
    
    def update_best(model):
        nonlocal best_predict, best_mse
        if model is None:
            return
        mse = safe_evaluate(model, X_train, y_train)
        if mse < best_mse:
            best_mse = mse
            best_predict = model
    
    def ridge_solve(F, target, lam=1e-5):
        A = F.T @ F + lam * np.eye(F.shape[1])
        A[0, 0] -= lam  # Don't regularize intercept
        return solve(A, F.T @ target)
    
    # Structural change: Use Chebyshev Polynomial Basis
    # Chebyshev polynomials are orthogonal on [-1, 1] and provide
    # efficient approximation of smooth functions on bounded domains.
    
    def make_chebyshev_features(X, max_degree, include_interactions=True):
        """
        Create Chebyshev polynomial features.
        Maps X to [-1, 1] and generates T_k(x) terms.
        Interactions are products of Chebyshev terms from different dimensions,
        subject to a total degree constraint.
        """
        # Map to [-1, 1]
        Xn = 2.0 * (X - X_min) / X_range - 1.0
        
        cols = [np.ones(len(Xn))]
        
        # Precompute Chebyshev polynomials for each dimension up to max_degree
        # T_0(x) = 1, T_1(x) = x, T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
        # We store them in a list of arrays: cheb_terms[j][k] is T_k(X[:, j])
        cheb_terms = []
        for j in range(Xn.shape[1]):
            x = Xn[:, j]
            terms_j = []
            if max_degree >= 0:
                terms_j.append(np.ones(len(x))) # T_0
            if max_degree >= 1:
                terms_j.append(x) # T_1
            for k in range(2, max_degree + 1):
                t_k = 2 * x * terms_j[-1] - terms_j[-2]
                terms_j.append(t_k)
            cheb_terms.append(terms_j)
        
        # Add main terms (single variable)
        for j in range(Xn.shape[1]):
            for k in range(1, max_degree + 1):
                cols.append(cheb_terms[j][k])
        
        # Interaction terms
        if include_interactions and Xn.shape[1] >= 2:
            # Pairwise interactions
            # We limit the product degree to keep feature count manageable
            # e.g., T_i(x_j) * T_k(x_l) where i+k <= max_interact_degree
            max_interact_degree = max_degree
            for i in range(Xn.shape[1]):
                for j in range(i + 1, Xn.shape[1]):
                    # We only need to consider degrees that sum to <= max_interact_degree
                    # But to keep it simple and dense for low degrees, we just take products
                    # of lower order terms. 
                    # To avoid explosion, we restrict individual degrees to max(1, max_degree//2)
                    max_deg_pair = max(1, max_degree // 2)
                    for k1 in range(1, max_deg_pair + 1):
                        for k2 in range(1, max_deg_pair + 1):
                            if k1 + k2 > max_interact_degree:
                                continue
                            # Product of Chebyshev terms
                            term = cheb_terms[i][k1] * cheb_terms[j][k2]
                            cols.append(term)
            
            # Triple interactions for 3+ dimensions
            if Xn.shape[1] >= 3:
                max_deg_triple = max(1, max_degree // 3)
                for i in range(Xn.shape[1]):
                    for j in range(i + 1, Xn.shape[1]):
                        for k in range(j + 1, Xn.shape[1]):
                            for k1 in range(1, max_deg_triple + 1):
                                for k2 in range(1, max_deg_triple + 1):
                                    for k3 in range(1, max_deg_triple + 1):
                                        if k1 + k2 + k3 > max_interact_degree:
                                            continue
                                        term = cheb_terms[i][k1] * cheb_terms[j][k2] * cheb_terms[k][k3]
                                        cols.append(term)
        
        return np.column_stack(cols)
    
    # Try a range of degrees and regularization levels
    degree_candidates = [3, 4, 5, 6, 8, 10]
    lam_candidates = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for max_deg in degree_candidates:
        if not check_time():
            break
        for lam in lam_candidates:
            if not check_time():
                break
            try:
                F = make_chebyshev_features(X_train, max_deg, include_interactions=True)
                if F.shape[1] > 500:  # Cap feature dimension for stability
                    continue
                w = ridge_solve(F, y_norm, lam)
                
                def predict_chebyshev(X, w, y_mean, y_std, md, inter=True):
                    F_test = make_chebyshev_features(X, md, include_interactions=inter)
                    return y_mean + y_std * (F_test @ w)
                
                update_best(lambda X, w=w, md=max_deg: predict_chebyshev(X, w, y_mean, y_std, md, True))
            except Exception:
                pass
    
    # Also try without interactions to see if simpler model works better for some cases
    for max_deg in [4, 6, 8]:
        if not check_time():
            break
        for lam in [1e-4, 1e-3, 1e-2]:
            if not check_time():
                break
            try:
                F = make_chebyshev_features(X_train, max_deg, include_interactions=False)
                w = ridge_solve(F, y_norm, lam)
                
                def predict_chebyshev_no_int(X, w, y_mean, y_std, md):
                    F_test = make_chebyshev_features(X, md, include_interactions=False)
                    return y_mean + y_std * (F_test @ w)
                
                update_best(lambda X, w=w, md=max_deg: predict_chebyshev_no_int(X, w, y_mean, y_std, md))
            except Exception:
                pass
    
    # Fallback: simple linear model
    if best_predict is None:
        Xn = (X_train - X_min) / X_range
        F = np.column_stack([np.ones(n), Xn])
        w = ridge_solve(F, y_norm, 1e-5)
        
        def predict_linear(X, w, y_mean, y_std):
            Xn = (X - X_min) / X_range
            F_test = np.column_stack([np.ones(len(Xn)), Xn])
            return y_mean + y_std * (F_test @ w)
        
        return lambda X: predict_linear(X, w, y_mean, y_std)
    
    return best_predict

# EVOLVE-BLOCK-END

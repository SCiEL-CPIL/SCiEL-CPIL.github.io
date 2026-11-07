"""Symbolic regression with rich feature expansion and regularized selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Global time tracker
_total_start = None
_time_limit = 320.0  # Leave 10s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n, d = X.shape
    
    if time.time() > _time_limit:
        # Fallback: return mean
        mean_y = np.mean(y)
        return lambda X_test: np.full(len(X_test), mean_y)

    # 1. Feature Generation
    def build_features(X):
        n = X.shape[0]
        d = X.shape[1]
        cols = [np.ones(n)]
        
        # Linear terms
        for j in range(d):
            cols.append(X[:, j])
        
        # Polynomial terms up to degree 3 (degree 4 might be too many for 200 pts with d=3)
        # For d=1: deg 3 is fine. For d=3, deg 3 has 10 terms. 
        # Let's include degree 2 and 3.
        for deg in range(2, 4):
            for combo in product(range(d), repeat=deg):
                if sorted(combo) != combo:
                    continue
                term = np.ones(n)
                for j in combo:
                    term = term * X[:, j]
                cols.append(term)
        
        # Trigonometric terms
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
        
        # Interactions with trig
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(X[:, i] * np.cos(X[:, j]))
        
        # Exponential and log terms
        # Scale exp to avoid overflow/underflow issues in standardization
        for j in range(d):
            # X is in [1, 10], so X/5 is in [0.2, 2], exp is safe
            cols.append(np.exp(X[:, j] / 5.0))
            cols.append(np.log1p(X[:, j]))
            cols.append(X[:, j] * np.log1p(X[:, j]))
        
        # Rational terms
        for j in range(d):
            cols.append(X[:, j] / (X[:, j] + 1.0))
            cols.append(X[:, j] ** 2 / (X[:, j] ** 2 + 1.0))
        
        # Pairwise products and ratios
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
                # Safe division
                denom = X[:, j] + 1e-8
                cols.append(X[:, i] / denom)
                cols.append(np.log1p(X[:, i]) * X[:, j])

        F = np.column_stack(cols)
        # Remove any non-finite columns
        finite_mask = np.all(np.isfinite(F), axis=0)
        F = F[:, finite_mask]
        
        return F

    F_full = build_features(X)
    
    # 2. Preprocessing: Standardize Features
    F_mean = F_full.mean(axis=0)
    F_std = F_full.std(axis=0)
    F_std[F_std < 1e-10] = 1.0
    F_norm = (F_full - F_mean) / F_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # 3. Model Selection: Ridge Regression with Lambda Sweep via CV
    # We want to find the best lambda.
    # Since n=200, we can do simple K-Fold.
    
    def ridge_solve(F_mat, y_vec, lam):
        n_feat = F_mat.shape[1]
        # A = F^T F + lam * I
        # We can use np.linalg.solve for efficiency
        A = F_mat.T @ F_mat + lam * np.eye(n_feat)
        b = F_mat.T @ y_vec
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        return w

    def cross_validate(F, y, k=5):
        n = F.shape[0]
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // k
        folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k)]
        folds[-1] = indices[(k-1)*fold_size:] # Handle remainder
        
        mse_sum = 0.0
        for i in range(k):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            F_tr = F[tr_idx]
            y_tr = y[tr_idx]
            F_val = F[val_idx]
            y_val = y[val_idx]
            
            # We need to select lambda here. 
            # To save time, we'll pass a fixed lambda range and return best MSE for that lambda.
            # Actually, let's structure it as: for each lambda, compute CV error.
            pass # This helper is just for clarity, we'll implement the loop below.

    # Lambda sweep
    # Log spaced lambdas
    lam_values = np.logspace(-2, 6, 50) # 0.01 to 1,000,000
    
    best_lambda = 1.0
    best_cv_mse = np.inf
    
    # To speed up CV, we can precompute F^T F and F^T y? No, because F changes per fold.
    # But n=200, d_features ~ 50-100. Ridge solve is fast.
    # 50 lambdas * 5 folds * 10 equations = 2500 solves. Should be fast enough.
    
    if time.time() > _time_limit - 10:
        # Not enough time for CV, use a default lambda
        best_lambda = 1.0
    else:
        # Simple 5-Fold CV
        n = F_norm.shape[0]
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        k = 5
        fold_size = n // k
        folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k)]
        folds[-1] = indices[(k-1)*fold_size:]
        
        for lam in lam_values:
            cv_mse = 0.0
            for i in range(k):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                
                F_tr = F_norm[tr_idx]
                y_tr = y_norm[tr_idx]
                F_val = F_norm[val_idx]
                y_val = y_norm[val_idx]
                
                w = ridge_solve(F_tr, y_tr, lam)
                pred_val = F_val @ w
                mse = np.mean((pred_val - y_val) ** 2)
                cv_mse += mse
            
            avg_cv_mse = cv_mse / k
            if avg_cv_mse < best_cv_mse:
                best_cv_mse = avg_cv_mse
                best_lambda = lam
                
            if time.time() > _time_limit - 5:
                break

    # 4. Final Fit
    w_final = ridge_solve(F_norm, y_norm, best_lambda)
    
    # 5. Prediction Function
    def predict(X_test):
        X_t = np.asarray(X_test, dtype=np.float64)
        F_t = build_features(X_t)
        # Ensure same feature count/order
        # build_features should be deterministic. 
        # However, if X_test has different range, standardization uses training stats.
        F_t_norm = (F_t - F_mean) / F_std
        
        # Handle potential shape mismatch if feature generation logic changed (it shouldn't)
        if F_t_norm.shape[1] != w_final.shape[0]:
             # Fallback: linear only?
             # This shouldn't happen if build_features is stable.
             pass
             
        y_pred_norm = F_t_norm @ w_final
        return y_pred_norm * y_std + y_mean

    return predict
# EVOLVE-BLOCK-END

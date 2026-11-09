import numpy as np
import time
from scipy.optimize import lasso_lars
from scipy.linalg import lstsq

def fit(X_train, y_train):
    start_time = time.time()
    # Deadline guard: 330s total limit. 
    # We have 4 equations. Let's aim for ~75s per equation to be safe.
    deadline = start_time + 75
    
    X = X_train.reshape(-1, 1)
    y = y_train.reshape(-1)
    n = len(y)
    
    # If time is already tight, just return mean
    if time.time() > deadline - 5:
        y_mean = np.mean(y)
        def predict_fallback(X_test):
            return np.full(X_test.shape[0], y_mean)
        return predict_fallback

    # Generate a diverse set of basis functions
    # The input ranges are roughly [-3, 3], [-2, 2], [0.1, 6].
    # We need functions that cover these domains well.
    
    x = X[:, 0]
    
    # 1. Polynomials (up to degree 6)
    polys = [np.ones_like(x)]
    for p in range(1, 7):
        polys.append(x ** p)
        
    # 2. Trigonometric functions (periods matched to range ~ 2*pi, pi, 2*pi/3)
    # Using frequencies that might match hidden equations
    freqs = [1.0, 2.0, 3.0, 0.5, np.pi, 2*np.pi, 1.0/np.pi]
    trigs = []
    for f in freqs:
        trigs.append(np.sin(f * x))
        trigs.append(np.cos(f * x))
        
    # 3. Exponential functions
    # Small exponents to avoid overflow in [0.1, 6] or [-3, 3]
    exps = []
    for e in [0.5, 1.0, 2.0, -0.5, -1.0, -2.0]:
        try:
            # Clip to prevent inf
            val = np.exp(e * x)
            if np.all(np.isfinite(val)):
                exps.append(val)
        except:
            pass
            
    # 4. Rational functions (safe division)
    rats = []
    for d in [1.0, 2.0, 0.5]:
        try:
            val = x / (x**2 + d**2)
            if np.all(np.isfinite(val)):
                rats.append(val)
        except:
            pass
    try:
        val = 1.0 / (x + 1.0)
        if np.all(np.isfinite(val)):
            rats.append(val)
    except:
        pass
    try:
        val = 1.0 / (x - 1.0)
        if np.all(np.isfinite(val)):
            rats.append(val)
    except:
        pass

    # Combine all features
    features_list = polys + trigs + exps + rats
    
    # If we ran out of time building features, just use polynomials
    if time.time() > deadline - 10:
        features_list = polys[:4] # Just up to cubic

    # Stack into matrix
    # Check for any non-finite values and replace with 0
    for i in range(len(features_list)):
        if not np.all(np.isfinite(features_list[i])):
            features_list[i] = np.where(np.isfinite(features_list[i]), features_list[i], 0)

    X_features = np.column_stack(features_list)
    
    # Remove any columns that are all zeros or have zero variance to save computation
    col_vars = np.var(X_features, axis=0)
    valid_cols = np.where(col_vars > 1e-10)[0]
    X_features = X_features[:, valid_cols]
    
    if X_features.shape[1] == 0:
        y_mean = np.mean(y)
        def predict_empty(X_test):
            return np.full(X_test.shape[0], y_mean)
        return predict_empty

    # Standardize features to help LASSO
    feat_mean = np.mean(X_features, axis=0)
    feat_std = np.std(X_features, axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_norm = (X_features - feat_mean) / feat_std
    
    # Standardize y
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Use LASSO via LARS (Least Angle Regression) for speed and stability
    # We want to find sparse coefficients.
    # We will try a few alpha values (regularization strengths)
    # Alpha = 0 is OLS. Higher alpha is sparser.
    
    best_score = np.inf
    best_coefs = None
    best_alpha = None
    
    # Grid of alpha values
    # 0.01 to 1.0 log-spaced, plus 0.0 (OLS)
    alphas = [0.0] + list(np.logspace(-2, 0, 20))
    
    for alpha in alphas:
        if time.time() > deadline - 5:
            break
            
        try:
            if alpha == 0.0:
                # OLS
                coefs, residuals, rank, s = lstsq(X_norm, y_norm, lapack_driver='gelsd')
            else:
                # LARS with L1 penalty
                # lasso_lars returns coefs for a given alpha
                coefs, lars_path, alphas_path = lasso_lars(X_norm, y_norm, 
                                                           max_iter=100, 
                                                           return_path=True)
                # Find the coef corresponding to our target alpha
                # The path gives alphas in decreasing order.
                # We need the coefs at the step where alpha_path <= alpha
                # lasso_lars returns the full path.
                # Actually, simpler to use the direct solver if available or just pick from path.
                # scipy.optimize.lars is for LARS. lasso_lars is for LASSO path.
                # Let's use the coefs from the path closest to our target alpha.
                
                # The alphas_path is decreasing.
                # Find index where alphas_path[i] <= alpha < alphas_path[i-1]
                # Or just interpolate? No, LASSO is piecewise linear.
                # Easiest: find the point in the path closest to alpha.
                idx = np.argmin(np.abs(alphas_path - alpha))
                # coefs is a 2D array (n_features, n_steps)
                coefs = coefs[:, idx]
            
            # Check if coefs are valid
            if not np.all(np.isfinite(coefs)):
                continue
                
            # Calculate training error to pick best alpha (bias-variance tradeoff)
            # Since we don't have validation set, we use training error but penalize complexity?
            # Actually, LASSO minimizes L2 + L1. 
            # A common heuristic is to pick the alpha that gives a "knee" in the error curve,
            # or simply use the training error of the OLS if LASSO is too aggressive.
            # However, for symbolic regression, sparsity is key.
            # Let's evaluate the training MSE for each alpha.
            # We want low training error, but also simplicity.
            # Let's just pick the alpha that minimizes training MSE among the sparse solutions?
            # No, that will always pick alpha=0 (OLS) which is dense.
            
            # Better strategy: Pick the sparsest solution that has training MSE close to the minimum.
            # Let's first find the min training MSE (likely alpha=0 or small alpha).
            
        except Exception:
            continue

    # Refined Strategy:
    # 1. Get OLS solution (dense)
    # 2. Get a few sparse LASSO solutions
    # 3. Select the sparsest one that doesn't degrade training error significantly.
    
    try:
        # OLS
        coefs_ols, _, _, _ = lstsq(X_norm, y_norm, lapack_driver='gelsd')
        pred_ols = X_norm @ coefs_ols
        mse_ols = np.mean((pred_ols - y_norm) ** 2)
        
        # LASSO Path
        coefs_path, lars_path, alphas_path = lasso_lars(X_norm, y_norm, max_iter=50, return_path=True)
        
        best_coefs = coefs_ols
        best_mse = mse_ols
        best_nz = np.sum(np.abs(coefs_ols) > 1e-8)
        
        # Iterate through the LASSO path
        for i in range(len(alphas_path)):
            if time.time() > deadline - 5:
                break
            coefs_i = coefs_path[:, i]
            if not np.all(np.isfinite(coefs_i)):
                continue
            pred_i = X_norm @ coefs_i
            mse_i = np.mean((pred_i - y_norm) ** 2)
            
            # We accept this sparse solution if its error is within 5% of OLS error
            # and it is sparser than the current best.
            if mse_i < best_mse * 1.05:
                nz_i = np.sum(np.abs(coefs_i) > 1e-8)
                if nz_i < best_nz:
                    best_coefs = coefs_i
                    best_mse = mse_i
                    best_nz = nz_i
                    # If we have a very sparse model (e.g., < 5 terms), we can stop early
                    if best_nz < 5:
                        break
                        
    except Exception:
        # Fallback to simple linear regression if LARS fails
        try:
            coefs_ols, _, _, _ = lstsq(X_norm, y_norm, lapack_driver='gelsd')
            best_coefs = coefs_ols
        except:
            best_coefs = np.zeros(X_norm.shape[1])

    # Map back to original scale
    # y_pred = (X_norm @ coefs) * y_std + y_mean
    # X_norm = (X_features - feat_mean) / feat_std
    
    def predict(X_test):
        x_test = X_test.reshape(-1, 1)[:, 0]
        
        # Reconstruct features
        # We must use the EXACT same feature generation logic
        polys = [np.ones_like(x_test)]
        for p in range(1, 7):
            polys.append(x_test ** p)
            
        trigs = []
        for f in freqs:
            trigs.append(np.sin(f * x_test))
            trigs.append(np.cos(f * x_test))
            
        exps = []
        for e in [0.5, 1.0, 2.0, -0.5, -1.0, -2.0]:
            try:
                val = np.exp(e * x_test)
                if np.all(np.isfinite(val)):
                    exps.append(val)
            except:
                pass
                
        rats = []
        for d in [1.0, 2.0, 0.5]:
            try:
                val = x_test / (x_test**2 + d**2)
                if np.all(np.isfinite(val)):
                    rats.append(val)
            except:
                pass
        try:
            val = 1.0 / (x_test + 1.0)
            if np.all(np.isfinite(val)):
                rats.append(val)
        except:
            pass
        try:
            val = 1.0 / (x_test - 1.0)
            if np.all(np.isfinite(val)):
                rats.append(val)
        except:
            pass

        features_list = polys + trigs + exps + rats
        
        # Handle non-finites in test set
        for i in range(len(features_list)):
            if not np.all(np.isfinite(features_list[i])):
                features_list[i] = np.where(np.isfinite(features_list[i]), features_list[i], 0)

        X_test_features = np.column_stack(features_list)
        
        # Apply same column filtering
        # Note: We filtered training columns based on variance. 
        # We must ensure test columns align. 
        # Since the feature set is deterministic, the columns that had 0 variance in train 
        # will likely have 0 variance in test or be constant. 
        # It is safer to just compute all and then select the same indices.
        
        # Recalculate col_vars on train? No, we didn't store valid_cols indices explicitly in a way that's easy to reuse if code changes.
        # Let's store valid_cols in a closure variable or recalculate.
        # Recalculating is safer if we don't store state.
        col_vars_train = np.var(X_features, axis=0) # X_features is from fit scope
        valid_cols = np.where(col_vars_train > 1e-10)[0]
        
        # Select corresponding columns in test
        X_test_features = X_test_features[:, valid_cols]
        
        # Normalize
        X_test_norm = (X_test_features - feat_mean) / feat_std
        
        # Predict
        y_pred_norm = X_test_norm @ best_coefs
        y_pred = y_pred_norm * y_std + y_mean
        
        # Ensure finite
        y_pred = np.where(np.isfinite(y_pred), y_pred, y_mean)
        
        return y_pred

    return predict

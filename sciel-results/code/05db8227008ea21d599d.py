"""
Symbolic regression using Local Polynomial Regression (LOESS) with adaptive bandwidth
selected via Leave-One-Out Cross-Validation.
"""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _loess_predict(X_test, X_train, y_train, bandwidth, degree=3):
    """
    Predict using LOESS.
    X_train, X_test: (N, 1) or (N,)
    y_train: (N,)
    bandwidth: fraction of N points in neighborhood (0.1 to 1.0)
    degree: polynomial degree
    """
    X_train = np.asarray(X_train).flatten()
    X_test = np.asarray(X_test).flatten()
    y_train = np.asarray(y_train).flatten()
    
    n = len(X_train)
    # Ensure at least degree+1 points in neighborhood
    min_k = degree + 1
    k = max(min_k, int(bandwidth * n))
    
    preds = np.zeros(len(X_test))
    
    # Sort training data for efficiency in finding neighbors?
    # For LOESS, we just need distances. With N=200, brute force is fine.
    
    for i in range(len(X_test)):
        x_test = X_test[i]
        
        # Compute distances
        dists = np.abs(X_train - x_test)
        
        # Get indices of k nearest neighbors
        # argsort is O(N log N). N=200, so 200*8 = 1600 ops. 
        # For 200 test points, 200*1600 = 320k ops. Very fast.
        # However, we can just take the k smallest.
        # Using argsort for simplicity and robustness.
        idx = np.argsort(dists)[:k]
        
        x_nb = X_train[idx]
        y_nb = y_train[idx]
        
        # Weights: tricube kernel
        # d_max = distance to k-th neighbor
        d_max = dists[idx[-1]]
        if d_max == 0:
            # All neighbors are at the same point? Unlikely with continuous data, but handle it.
            w = np.ones(k)
        else:
            # Tricube: (1 - (d/d_max)^3)^3
            u = dists[idx] / d_max
            w = (1 - u**3)**3
            
        # Ensure weights are not all zero
        if np.sum(w) < 1e-10:
            w = np.ones(k)
            
        # Weighted Least Squares
        # Design matrix for polynomial degree
        # x_nb is centered? LOESS usually centers at x_test for stability.
        # Let's center x_nb at x_test.
        x_c = x_nb - x_test
        
        # Vandermonde matrix
        # Columns: 1, x_c, x_c^2, ... x_c^degree
        # Using np.vander
        # np.vander(x_c, N=degree+1, increasing=True)
        A = np.vander(x_c, N=degree+1, increasing=True)
        
        # Weighted LS: solve (A^T W A) beta = A^T W y
        W = np.diag(w)
        # To avoid explicit diag, multiply rows of A by sqrt(w)
        sqrt_w = np.sqrt(w)
        A_w = A * sqrt_w[:, None]
        y_w = y_nb * sqrt_w
        
        try:
            # Use lstsq for robustness
            beta, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=None)
            pred = beta[0] # Since x_c=0 for the prediction point, only intercept matters
        except np.linalg.LinAlgError:
            # Fallback to weighted mean if LS fails
            pred = np.sum(w * y_nb) / np.sum(w)
            
        preds[i] = pred
        
    return preds

def fit(X_train, y_train):
    global _t_start
    _t_start = time.time()
    
    X = np.asarray(X_train, dtype=np.float64).flatten()
    y = np.asarray(y_train, dtype=np.float64).flatten()
    
    n = len(X)
    
    # Standardize X and y for numerical stability in polynomial fitting?
    # LOESS handles local scaling via centering, but global standardization helps with bandwidth interpretation.
    X_mean = X.mean()
    X_std = X.std() + 1e-8
    Xn = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    yn = (y - y_mean) / y_std
    
    # LOOCV for bandwidth selection
    # Bandwidth candidates: fraction of N
    # Too small -> high variance, too large -> high bias
    # Typical range: 0.1 to 0.8
    bandwidths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Degrees to consider: 1 (linear), 2 (quadratic), 3 (cubic)
    # Cubic is standard for LOESS. Let's stick to 3 for simplicity and speed, 
    # or try 1 and 3 if time permits. 
    # Given N=200, LOOCV is N * (N log N + N^3) approx.
    # Actually, LOOCV for LOESS can be computed efficiently using hat matrix, 
    # but for N=200, brute force LOOCV is N * (cost of one LOESS predict).
    # Cost of one LOESS predict for 1 point: O(N log N) for sorting + O(N^3) for LS (small N).
    # 200 * 200 * log(200) is fine.
    
    best_bw = 0.5
    best_degree = 3
    best_cv_mse = np.inf
    
    # Try different degrees and bandwidths
    # To save time, we can do a coarse search first.
    # Let's try degree 3 for all bandwidths first.
    
    for degree in [3, 2, 1]:
        for bw in bandwidths:
            if time.time() - _t_start > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            # Compute LOOCV MSE
            # For each i, predict y_i using all other points
            # We can just call _loess_predict with X_train with i-th point removed?
            # That's expensive. 
            # Alternative: Use the fact that LOESS is a local smoother.
            # For N=200, we can just loop i=0..N-1 and do a quick local fit.
            
            cv_mses = []
            for i in range(n):
                # Mask out i-th point
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                X_train_cv = Xn[mask]
                y_train_cv = yn[mask]
                
                # Predict y_i
                # We need to predict at Xn[i] using X_train_cv, y_train_cv
                # _loess_predict expects arrays.
                # It will find k nearest neighbors from X_train_cv.
                # Note: bandwidth is fraction of N_cv.
                try:
                    pred_i = _loess_predict(np.array([Xn[i]]), X_train_cv, y_train_cv, bw, degree)[0]
                    cv_mses.append((pred_i - yn[i])**2)
                except Exception:
                    cv_mses.append(np.inf)
                    
            mse = np.mean(cv_mses)
            if mse < best_cv_mse:
                best_cv_mse = mse
                best_bw = bw
                best_degree = degree
                
        if time.time() - _t_start > TIME_LIMIT - SAFETY_MARGIN:
            break
            
    # Refine bandwidth around best_bw?
    # If time permits, do a finer search around best_bw
    if time.time() - _t_start < TIME_LIMIT - SAFETY_MARGIN - 20:
        bw_range = np.linspace(max(0.05, best_bw - 0.1), min(0.95, best_bw + 0.1), 5)
        for bw in bw_range:
            if time.time() - _t_start > TIME_LIMIT - SAFETY_MARGIN:
                break
            cv_mses = []
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                X_train_cv = Xn[mask]
                y_train_cv = yn[mask]
                try:
                    pred_i = _loess_predict(np.array([Xn[i]]), X_train_cv, y_train_cv, bw, best_degree)[0]
                    cv_mses.append((pred_i - yn[i])**2)
                except Exception:
                    cv_mses.append(np.inf)
            mse = np.mean(cv_mses)
            if mse < best_cv_mse:
                best_cv_mse = mse
                best_bw = bw
                
    # Final prediction function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64).flatten()
        # Standardize
        Xn_test = (X_test - X_mean) / X_std
        # Predict in standardized space
        y_pred_std = _loess_predict(Xn_test, Xn, yn, best_bw, best_degree)
        # Denormalize
        return y_pred_std * y_std + y_mean
        
    return predict

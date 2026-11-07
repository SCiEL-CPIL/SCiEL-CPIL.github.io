"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import solve

def fit(X_train, y_train):
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin before 300s limit
    
    X = X_train
    y = y_train
    d = X.shape[1]
    n = X.shape[0]
    
    # Normalize inputs and target
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Kernel: Matern 5/2
    def matern52(X1, X2, length_scale):
        # X1: (n, d), X2: (m, d)
        # length_scale: scalar
        X1_scaled = X1 / length_scale
        X2_scaled = X2 / length_scale
        
        sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] + np.sum(X2_scaled**2, axis=1)[None, :] - 2 * np.dot(X1_scaled, X2_scaled.T)
        sq_dists = np.maximum(sq_dists, 0)
        r = np.sqrt(sq_dists)
        
        sqrt3_r = np.sqrt(3) * r
        K = (1 + sqrt3_r + (sqrt3_r**2) / 3) * np.exp(-sqrt3_r)
        return K

    # We will use a fixed length scale for simplicity in the ensemble, 
    # or optimize a few. Given time constraints, let's use a few fixed scales 
    # and rely on the ensemble + CV.
    # Scales: 0.5, 1.0, 2.0, 4.0
    
    scales = [0.5, 1.0, 2.0, 4.0]
    n_landmarks = 50  # Sparse approximation size
    
    candidates = []
    
    # Generate diverse landmark subsets
    # We'll create 5 different random subsets for each scale
    n_subsets = 5
    rng = np.random.default_rng(42)
    
    for scale in scales:
        K_full = matern52(X_norm, X_norm, scale)
        # Add small regularization to K for stability
        K_reg = K_full + 1e-6 * np.eye(n)
        
        for subset_idx in range(n_subsets):
            if time.time() > deadline:
                break
                
            # Select landmarks
            # Use random selection, but ensure diversity by picking from different parts of data
            # Simple random is often sufficient for Nyström
            landmark_indices = rng.choice(n, size=n_landmarks, replace=False)
            
            # Nyström approximation
            # K_ff: (m, m), K_ff_inv: (m, m)
            K_ff = K_full[np.ix_(landmark_indices, landmark_indices)]
            K_ff_reg = K_ff + 1e-6 * np.eye(n_landmarks)
            
            try:
                K_ff_inv = np.linalg.inv(K_ff_reg)
            except np.linalg.LinAlgError:
                continue
                
            # K_xf: (n, m)
            K_xf = K_full[:, landmark_indices]
            
            # Projection matrix P = K_xf K_ff^{-1}
            # We want to fit y = P w + epsilon
            # Actually, the Nyström GP mean is K_xf K_ff^{-1} K_ff^{-1} K_xf^T y? 
            # No, standard Nyström regression: 
            # f(x) = k(x, Z) K_ZZ^{-1} y_Z
            # This is linear in y_Z. So we can just use w = K_ff^{-1} y_Z?
            # Wait, that's the standard GP interpolation.
            # For sparse GP (VFE/Nyström), we often solve for coefficients alpha such that f(x) = K(x, Z) alpha.
            # The optimal alpha is (K_ZZ + sigma^2 I)^{-1} y_Z.
            # Let's use Elastic Net on the features K(x, Z).
            
            # Features for training: K_xf (n, m)
            # We fit y_norm ~ K_xf @ alpha
            # Use Ridge/Elastic Net via closed form or simple solver.
            # Since m is small (50), we can use least squares with regularization.
            
            # Regularized Least Squares:
            # alpha = (K_xf^T K_xf + lambda I)^{-1} K_xf^T y
            # Let's try a few regularization strengths
            lambdas = [1e-4, 1e-3, 1e-2]
            
            for lam in lambdas:
                if time.time() > deadline:
                    break
                    
                A = K_xf.T @ K_xf + lam * np.eye(n_landmarks)
                b = K_xf.T @ y_norm
                
                try:
                    alpha = solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                
                # Calculate LOO-CV Score (approximate or exact?)
                # Exact LOO is O(n^2) which is 40k ops, fine.
                # But we have many candidates. 
                # Fast approximation: Use the hat matrix H = K_xf (K_xf^T K_xf + lam I)^{-1} K_xf^T
                # LOO error = (y - y_hat)^2 / (1 - H_ii)^2
                # H is (n, n). n=200. 200x200 is fine.
                
                # Compute H
                # H = K_xf @ inv(A) @ K_xf.T
                # We already have A.
                A_inv = np.linalg.inv(A)
                H = K_xf @ (A_inv @ K_xf.T)
                
                y_pred_train = K_xf @ alpha
                residuals = y_norm - y_pred_train
                
                # LOO CV MSE
                diag_H = np.diag(H)
                # Avoid division by zero
                denom = (1 - diag_H) ** 2
                denom[denom < 1e-8] = 1e-8
                
                loo_mse = np.mean((residuals / (1 - diag_H)) ** 2)
                
                if np.isfinite(loo_mse):
                    candidates.append({
                        'scale': scale,
                        'indices': landmark_indices,
                        'alpha': alpha,
                        'loo_mse': loo_mse,
                        'lam': lam
                    })
                    
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if not candidates:
        # Fallback: Mean predictor
        def predict_fallback(X_test):
            X_test = np.asarray(X_test)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            return np.ones(X_test.shape[0]) * y_mean
        return predict_fallback
    
    # Sort by LOO MSE
    candidates.sort(key=lambda x: x['loo_mse'])
    
    # Select top 3 models
    top_models = candidates[:3]
    
    # Precompute prediction matrices for top models to speed up predict?
    # Or just store the data needed.
    # For each model, we need:
    # 1. Scale
    # 2. Landmark indices
    # 3. Alpha coefficients
    # 4. We need to compute K(X_test, X_landmarks) at predict time.
    
    model_data = []
    for m in top_models:
        model_data.append({
            'scale': m['scale'],
            'indices': m['indices'],
            'alpha': m['alpha']
        })
    
    def predict(X_test):
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
            
        X_test_norm = (X_test - X_mean) / X_std
        
        y_pred_total = np.zeros(X_test.shape[0])
        
        for m in model_data:
            # Compute K(X_test, X_landmarks)
            X_landmarks = X_norm[m['indices']]
            K_test_land = matern52(X_test_norm, X_landmarks, m['scale'])
            
            y_pred_model = K_test_land @ m['alpha']
            y_pred_total += y_pred_model
            
        y_pred_total /= len(model_data)
        
        y_pred = y_pred_total * y_std + y_mean
        
        # Ensure finite
        y_pred = np.nan_to_num(y_pred, nan=y_mean, posinf=y_mean, neginf=y_mean)
        
        return y_pred

    return predict
# EVOLVE-BLOCK-END

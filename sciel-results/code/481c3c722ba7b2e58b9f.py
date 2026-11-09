import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 285.0  # Safety margin before 300s kill
    
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    n_samples, n_features = X_train.shape
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_std_val = float(y_std) if y_std > 1e-8 else 1.0
    y_train_norm = (y_train - y_mean) / y_std_val
    
    # Standardize X for feature generation stability
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    
    def generate_features(X):
        """
        Generate a rich set of basis functions.
        X: (N, d) raw input
        Returns: Phi (N, M) feature matrix
        """
        X = np.asarray(X, dtype=float)
        N, d = X.shape
        
        # Standardized version for polynomial terms
        X_norm = (X - X_mean) / X_std
        
        feats = [np.ones((N, 1))]
        
        # Per-variable features
        for i in range(d):
            x = X[:, i:i+1]
            xn = X_norm[:, i:i+1]
            
            # Polynomials (up to degree 4)
            feats.append(x)
            feats.append(xn)
            feats.append(x**2)
            feats.append(xn**2)
            feats.append(x**3)
            feats.append(xn**3)
            feats.append(x**4)
            feats.append(xn**4)
            
            # Trigonometric (use raw x to capture actual periods)
            for freq in [0.5, 1.0, 1.5, 2.0]:
                feats.append(np.sin(freq * x))
                feats.append(np.cos(freq * x))
            
            # Exponential (dampened for safety)
            feats.append(np.exp(-0.1 * x))
            feats.append(np.exp(-0.25 * x))
            feats.append(np.exp(0.1 * x))
            
            # Absolute value
            feats.append(np.abs(x))
            feats.append(np.abs(xn))
            
        # Pairwise interactions
        if d > 1:
            for i in range(d):
                for j in range(i+1, d):
                    xi = X[:, i:i+1]
                    xj = X[:, j:j+1]
                    xni = X_norm[:, i:i+1]
                    xnj = X_norm[:, j:j+1]
                    
                    feats.append(xi * xj)
                    feats.append(xni * xnj)
                    feats.append(np.sin(xi) * np.cos(xj))
                    feats.append(np.cos(xi) * np.sin(xj))
                    feats.append(np.sin(xi) * np.sin(xj))
                    feats.append(np.cos(xi) * np.cos(xj))
                    feats.append(xni * xnj**2)
                    feats.append(xnj * xni**2)
                    
        # Triple interactions if d >= 3
        if d >= 3:
            for i in range(d):
                for j in range(i+1, d):
                    for k in range(j+1, d):
                        xni = X_norm[:, i:i+1]
                        xnj = X_norm[:, j:j+1]
                        xnk = X_norm[:, k:k+1]
                        
                        feats.append(xni * xnj * xnk)
                        feats.append(np.sin(X[:, i:i+1]) * np.sin(X[:, j:j+1]) * np.sin(X[:, k:k+1]))
                        
        Phi = np.hstack(feats)
        
        # Remove constant-variance columns
        Phi_std = Phi.std(axis=0)
        valid_mask = Phi_std > 1e-8
        if valid_mask.sum() < Phi.shape[1]:
            Phi = Phi[:, valid_mask]
            
        return Phi

    # Generate training features
    Phi_train = generate_features(X_train)
    
    # Standardize features
    Phi_mean = Phi_train.mean(axis=0)
    Phi_std = Phi_train.std(axis=0)
    Phi_std[Phi_std < 1e-8] = 1.0
    Phi_train_norm = (Phi_train - Phi_mean) / Phi_std
    
    N, M = Phi_train_norm.shape
    
    # 2. Ridge Regression with LOOCV for alpha selection
    def ridge_loocv_score(Phi, y, alpha):
        """
        Compute LOOCV MSE for Ridge Regression using hat matrix formula.
        """
        PhiT_Phi = Phi.T @ Phi
        A = PhiT_Phi + alpha * np.eye(M)
        
        try:
            # Solve A X = Phi^T, X is (M, N)
            X_sol = np.linalg.solve(A, Phi.T)
            H_diag = np.sum(Phi * X_sol.T, axis=1)
            y_hat = Phi @ X_sol
        except np.linalg.LinAlgError:
            X_sol, _, _, _ = np.linalg.lstsq(A, Phi.T, rcond=None)
            H_diag = np.sum(Phi * X_sol.T, axis=1)
            y_hat = Phi @ X_sol
            
        # LOOCV error: (y_i - y_hat_i) / (1 - H_ii)
        denom = 1 - H_diag
        # Avoid division by zero
        denom[np.abs(denom) < 1e-10] = 1e-10
        errors = (y - y_hat) / denom
        mse = np.mean(errors**2)
        return mse
    
    # Grid search for alpha
    alphas = np.logspace(-5, 5, 150)
    
    best_alpha = 1.0
    best_score = np.inf
    
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
        try:
            score = ridge_loocv_score(Phi_train_norm, y_train_norm, alpha)
            if score < best_score:
                best_score = score
                best_alpha = alpha
        except Exception:
            continue
            
    # 3. Final Fit with best alpha
    A = Phi_train_norm.T @ Phi_train_norm + best_alpha * np.eye(M)
    try:
        w = np.linalg.solve(A, Phi_train_norm.T @ y_train_norm)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, Phi_train_norm.T @ y_train_norm, rcond=None)[0]
        
    # 4. Try to improve with a second stage: add RBF features with learned centers
    # If time permits, try a hybrid approach
    if time.time() < deadline - 30:
        # Try adding RBF features with k-means-like centers
        try:
            # Select centers via simple k-means
            n_centers = min(20, n_samples // 2)
            if n_centers >= 2:
                # Initialize centers
                idx = np.random.choice(n_samples, n_centers, replace=False)
                centers = X_train[idx].copy()
                
                # Simple k-means iterations
                for _ in range(5):
                    if time.time() > deadline - 30:
                        break
                    # Assign points to nearest center
                    dists = np.linalg.norm(X_train[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                    assignments = np.argmin(dists, axis=1)
                    
                    # Update centers
                    new_centers = centers.copy()
                    for c in range(n_centers):
                        mask = assignments == c
                        if mask.sum() > 0:
                            new_centers[c] = X_train[mask].mean(axis=0)
                    if np.allclose(centers, new_centers):
                        break
                    centers = new_centers
                
                # Generate RBF features
                def rbf_features(X, centers, bandwidth):
                    X = np.asarray(X, dtype=float)
                    N = X.shape[0]
                    dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                    return np.exp(-dists**2 / (2 * bandwidth**2))
                
                # Try a few bandwidths
                best_rbf_score = best_score
                best_rbf_bw = None
                best_rbf_alpha = best_alpha
                best_rbf_w = w
                
                for bw in [0.1, 0.2, 0.5, 1.0, 2.0]:
                    if time.time() > deadline - 15:
                        break
                    try:
                        Phi_rbf_train = rbf_features(X_train, centers, bw)
                        
                        # Combine with original features
                        Phi_combined = np.hstack([Phi_train_norm, Phi_rbf_train])
                        
                        # Standardize RBF features
                        rbf_mean = Phi_rbf_train.mean(axis=0)
                        rbf_std = Phi_rbf_train.std(axis=0)
                        rbf_std[rbf_std < 1e-8] = 1.0
                        Phi_rbf_train_norm = (Phi_rbf_train - rbf_mean) / rbf_std
                        
                        Phi_combined_norm = np.hstack([Phi_train_norm, Phi_rbf_train_norm])
                        
                        N_c, M_c = Phi_combined_norm.shape
                        
                        # Quick alpha search
                        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                            if time.time() > deadline - 15:
                                break
                            try:
                                score = ridge_loocv_score(Phi_combined_norm, y_train_norm, alpha)
                                if score < best_rbf_score:
                                    best_rbf_score = score
                                    best_rbf_bw = bw
                                    best_rbf_alpha = alpha
                                    A_c = Phi_combined_norm.T @ Phi_combined_norm + alpha * np.eye(M_c)
                                    best_rbf_w = np.linalg.solve(A_c, Phi_combined_norm.T @ y_train_norm)
                            except Exception:
                                continue
                    except Exception:
                        continue
                
                # If RBF helped, use it
                if best_rbf_bw is not None and best_rbf_score < best_score:
                    best_alpha = best_rbf_alpha
                    w = best_rbf_w
                    # Store centers and bandwidth for prediction
                    rbf_centers = centers
                    rbf_bw = best_rbf_bw
                    rbf_mean = Phi_rbf_train.mean(axis=0)
                    rbf_std = Phi_rbf_train.std(axis=0)
                    rbf_std[rbf_std < 1e-8] = 1.0
                    has_rbf = True
        except Exception:
            has_rbf = False
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Generate features for test data
        Phi_test = generate_features(X)
        
        # Standardize features using training stats
        Phi_test_norm = (Phi_test - Phi_mean) / Phi_std
        
        # Check if we have RBF features
        if 'has_rbf' in locals() and has_rbf:
            try:
                Phi_rbf_test = rbf_features(X, rbf_centers, rbf_bw)
                Phi_rbf_test_norm = (Phi_rbf_test - rbf_mean) / rbf_std
                Phi_test_combined = np.hstack([Phi_test_norm, Phi_rbf_test_norm])
                y_pred_norm = Phi_test_combined @ w
            except Exception:
                y_pred_norm = Phi_test_norm @ w
        else:
            y_pred_norm = Phi_test_norm @ w
            
        y_pred = y_pred_norm * y_std_val + y_mean
        
        # Ensure finite predictions
        y_pred = np.where(np.isfinite(y_pred), y_pred, y_mean)
        
        return y_pred
    
    return predict

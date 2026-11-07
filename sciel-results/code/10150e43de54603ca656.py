"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin for 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n_samples, d = X.shape
    
    # Normalize inputs to [0, 1] for stable frequency sampling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Strategy: Ensemble of Random Fourier Feature (RFF) Ridge Regressors
    # Each model approximates a GP with a specific spectral mixture.
    # We use multiple models with different random frequencies to average out variance.
    
    models = []
    num_models = 5  # Number of ensemble members
    
    # Hyperparameters for RFF
    # We sample frequencies from a log-uniform distribution to capture multiple scales
    # This is equivalent to a spectral mixture kernel
    n_features_per_model = 512  # Number of random features per model
    
    # Precompute base features if possible? No, X is different for each model's freqs.
    # But we can compute the base X_norm once.
    
    for i in range(num_models):
        if time.time() > DEADLINE - 5.0:
            break
            
        # Sample frequencies: log-uniform in [0.1, 10.0]
        # This allows capturing both high-frequency and low-frequency variations
        log_freqs = np.random.uniform(np.log(0.1), np.log(10.0), size=(n_features_per_model, d))
        freqs = np.exp(log_freqs)
        
        # Sample phases uniformly in [0, 2*pi]
        phases = np.random.uniform(0, 2 * np.pi, size=(n_features_per_model, d))
        
        # Compute Random Fourier Features: phi(x) = [cos(x^T w_1 + b_1), sin(x^T w_1 + b_1), ...]
        # x is (N, D), w is (M, D). x^T w is (N, M)
        proj = X_norm @ freqs.T  # (N, M)
        proj += phases.T  # (N, M)
        
        phi = np.concatenate([np.cos(proj), np.sin(proj)], axis=1)  # (N, 2M)
        
        # Add bias term
        phi = np.hstack([phi, np.ones((n_samples, 1))])
        
        # Ridge Regression to solve for weights
        # Minimize ||Phi w - y||^2 + lambda ||w||^2
        # (Phi^T Phi + lambda I) w = Phi^T y
        lambda_reg = 1e-4
        A = phi.T @ phi + lambda_reg * np.eye(phi.shape[1])
        b = phi.T @ y_norm
        
        try:
            w = np.linalg.solve(A, b)
            # Store model parameters
            models.append({
                'freqs': freqs,
                'phases': phases,
                'weights': w,
                'bias_index': phi.shape[1] - 1
            })
        except np.linalg.LinAlgError:
            # Fallback to least squares if solve fails
            w, *_ = np.linalg.lstsq(phi, y_norm, rcond=None)
            models.append({
                'freqs': freqs,
                'phases': phases,
                'weights': w,
                'bias_index': phi.shape[1] - 1
            })

    # If no models were created (unlikely), use a simple linear fallback
    if not models:
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)
        def predict_fallback(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            return features(X_test_norm) @ w * y_std + y_mean
        return predict_fallback

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        
        predictions = []
        
        for model in models:
            freqs = model['freqs']
            phases = model['phases']
            w = model['weights']
            
            proj = X_test_norm @ freqs.T
            proj += phases.T
            
            phi = np.concatenate([np.cos(proj), np.sin(proj)], axis=1)
            phi = np.hstack([phi, np.ones((X_test.shape[0], 1))])
            
            pred_norm = phi @ w
            predictions.append(pred_norm)
        
        # Average the predictions from the ensemble
        avg_pred_norm = np.mean(predictions, axis=0)
        
        # Denormalize
        return avg_pred_norm * y_std + y_mean

    return predict
# EVOLVE-BLOCK-END

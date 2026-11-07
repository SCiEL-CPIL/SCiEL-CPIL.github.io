import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 300.0
    
    X = X_train[:, 0]
    y = y_train
    
    # 1. Generate Feature Library
    # We assume 1D input based on task description
    # Library of candidate basis functions
    def generate_features(x):
        feats = []
        feats.append(np.ones_like(x)) # Intercept
        feats.append(x)
        feats.append(x**2)
        feats.append(x**3)
        feats.append(x**4)
        feats.append(x**5)
        feats.append(x**6)
        
        feats.append(np.sin(x))
        feats.append(np.cos(x))
        feats.append(np.sin(2*x))
        feats.append(np.cos(2*x))
        feats.append(np.sin(3*x))
        feats.append(np.cos(3*x))
        
        feats.append(np.exp(x))
        feats.append(np.exp(-x))
        
        feats.append(x * np.sin(x))
        feats.append(x * np.cos(x))
        
        feats.append(np.log(np.abs(x) + 1e-6))
        
        return np.column_stack(feats)

    F_train = generate_features(X)
    
    # 2. Greedy Forward Selection
    n_samples, n_features = F_train.shape
    selected_indices = [0] # Start with intercept
    current_mse = np.mean((y - np.mean(y))**2) # MSE with only intercept
    
    max_features = 10
    improvement_threshold = 1e-4
    
    for _ in range(max_features):
        best_idx = -1
        best_mse = current_mse
        
        for j in range(n_features):
            if j in selected_indices:
                continue
            
            new_indices = selected_indices + [j]
            # Fit least squares on selected features
            try:
                w, *_ = np.linalg.lstsq(F_train[:, new_indices], y, rcond=None)
                y_pred = F_train[:, new_indices] @ w
                mse = np.mean((y - y_pred)**2)
            except:
                continue
            
            if mse < best_mse:
                best_mse = mse
                best_idx = j
        
        if best_idx == -1:
            break
        
        # Check for significant improvement
        if current_mse > 1e-10:
            relative_improvement = (current_mse - best_mse) / current_mse
        else:
            relative_improvement = 1.0 if best_mse < current_mse else 0.0
            
        if relative_improvement < improvement_threshold:
            break
            
        selected_indices.append(best_idx)
        current_mse = best_mse
        
        # Time check
        if time.time() > deadline - 10:
            break

    # 3. Final Fit
    w, *_ = np.linalg.lstsq(F_train[:, selected_indices], y, rcond=None)
    
    # 4. Define Predict
    def predict(X):
        X_flat = X[:, 0] if X.ndim > 1 else X
        F_test = generate_features(X_flat)
        return F_test[:, selected_indices] @ w

    return predict

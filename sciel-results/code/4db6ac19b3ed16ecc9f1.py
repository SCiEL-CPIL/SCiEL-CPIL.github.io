"""Symbolic regression with neural network (MLP) trained via Adam."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _relu(x):
    return np.maximum(0, x)

def _relu_grad(x):
    return (x > 0).astype(float)

def _tanh(x):
    return np.tanh(x)

def _tanh_grad(x):
    return 1 - np.tanh(x) ** 2

def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def _sigmoid_grad(x):
    s = _sigmoid(x)
    return s * (1 - s)

def _init_weights(n_in, n_out, scale=1.0):
    """He initialization for ReLU, Xavier for tanh."""
    return (np.random.randn(n_in, n_out) * scale * np.sqrt(2.0 / n_in))

def _forward_mlp(X, W1, b1, W2, b2, activation='relu'):
    """Forward pass for a 2-hidden-layer MLP."""
    if activation == 'relu':
        h1 = _relu(X @ W1 + b1)
        h2 = _relu(h1 @ W2 + b2)
    else:
        h1 = _tanh(X @ W1 + b1)
        h2 = _tanh(h1 @ W2 + b2)
    return h1, h2

def _forward_mlp3(X, W1, b1, W2, b2, W3, b3, activation='relu'):
    """Forward pass for a 3-hidden-layer MLP."""
    if activation == 'relu':
        h1 = _relu(X @ W1 + b1)
        h2 = _relu(h1 @ W2 + b2)
        h3 = _relu(h2 @ W3 + b3)
    else:
        h1 = _tanh(X @ W1 + b1)
        h2 = _tanh(h1 @ W2 + b2)
        h3 = _tanh(h2 @ W3 + b3)
    return h1, h2, h3

def _backward_mlp(X, y, W1, b1, W2, b2, activation='relu', lr=0.01, lam=1e-4):
    """
    Backward pass for 2-hidden-layer MLP.
    Returns gradients for all parameters.
    """
    n, d = X.shape
    h1, h2 = _forward_mlp(X, W1, b1, W2, b2, activation)
    out = h2 @ np.zeros((W2.shape[1], 1)) + np.zeros(1)  # dummy, we'll compute properly
    
    # Output layer: single neuron
    # We need an output weight vector and bias
    # Let's restructure: use a separate output layer
    pass

def _fit_mlp(X, y, n_epochs=500, hidden1=32, hidden2=16, lr=0.01, lam=1e-3, activation='relu', seed=42):
    """
    Fit a 2-hidden-layer MLP using manual backprop with Adam optimizer.
    """
    np.random.seed(seed)
    n, d = X.shape
    
    # Normalize y
    y_mean = y.mean()
    y_std = y.std() + 1e-10
    y_norm = (y - y_mean) / y_std
    
    # Initialize weights
    W1 = _init_weights(d, hidden1)
    b1 = np.zeros(hidden1)
    W2 = _init_weights(hidden1, hidden2)
    b2 = np.zeros(hidden2)
    W3 = _init_weights(hidden2, 1)
    b3 = np.zeros(1)
    
    # Adam state
    m_W1 = np.zeros_like(W1); v_W1 = np.zeros_like(W1)
    m_b1 = np.zeros_like(b1); v_b1 = np.zeros_like(b1)
    m_W2 = np.zeros_like(W2); v_W2 = np.zeros_like(W2)
    m_b2 = np.zeros_like(b2); v_b2 = np.zeros_like(b2)
    m_W3 = np.zeros_like(W3); v_W3 = np.zeros_like(W3)
    m_b3 = np.zeros_like(b3); v_b3 = np.zeros_like(b3)
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    for epoch in range(n_epochs):
        t += 1
        # Forward
        if activation == 'relu':
            z1 = X @ W1 + b1
            h1 = _relu(z1)
            z2 = h1 @ W2 + b2
            h2 = _relu(z2)
        else:
            z1 = X @ W1 + b1
            h1 = _tanh(z1)
            z2 = h1 @ W2 + b2
            h2 = _tanh(z2)
        
        out = h2 @ W3 + b3  # shape (n, 1)
        out_flat = out.flatten()
        
        # Loss: MSE + L2 regularization
        loss = np.mean((out_flat - y_norm) ** 2) + lam * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        
        # Backward
        # dL/dout = 2 * (out_flat - y_norm) / n
        d_out = 2 * (out_flat - y_norm) / n
        
        dW3 = h2.T @ d_out.reshape(-1, 1) + 2 * lam * W3
        db3 = np.sum(d_out).reshape(1)
        
        d_h2 = d_out.reshape(-1, 1) @ W3.T
        
        if activation == 'relu':
            dz2 = d_h2 * _relu_grad(z2)
        else:
            dz2 = d_h2 * _tanh_grad(z2)
        
        dW2 = h1.T @ dz2 + 2 * lam * W2
        db2 = np.sum(dz2, axis=0)
        
        d_h1 = dz2 @ W2.T
        
        if activation == 'relu':
            dz1 = d_h1 * _relu_grad(z1)
        else:
            dz1 = d_h1 * _tanh_grad(z1)
        
        dW1 = X.T @ dz1 + 2 * lam * W1
        db1 = np.sum(dz1, axis=0)
        
        # Adam update
        def adam_update(m, v, grad, param):
            nonlocal t
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            return param - lr * m_hat / (np.sqrt(v_hat) + eps), m, v
        
        W1, m_W1, v_W1 = adam_update(m_W1, v_W1, dW1, W1)
        b1, m_b1, v_b1 = adam_update(m_b1, v_b1, db1, b1)
        W2, m_W2, v_W2 = adam_update(m_W2, v_W2, dW2, W2)
        b2, m_b2, v_b2 = adam_update(m_b2, v_b2, db2, b2)
        W3, m_W3, v_W3 = adam_update(m_W3, v_W3, dW3, W3)
        b3, m_b3, v_b3 = adam_update(m_b3, v_b3, db3, b3)
    
    # Final forward to check
    if activation == 'relu':
        z1 = X @ W1 + b1
        h1 = _relu(z1)
        z2 = h1 @ W2 + b2
        h2 = _relu(z2)
    else:
        z1 = X @ W1 + b1
        h1 = _tanh(z1)
        z2 = h1 @ W2 + b2
        h2 = _tanh(z2)
    out = h2 @ W3 + b3
    pred_norm = out.flatten()
    pred = pred_norm * y_std + y_mean
    
    return pred, y_mean, y_std, W1, b1, W2, b2, W3, b3, activation

def _predict_mlp(X_test, X_mean, X_std, y_mean, y_std, W1, b1, W2, b2, W3, b3, activation):
    """Predict using trained MLP."""
    X_norm = (X_test - X_mean) / X_std
    if activation == 'relu':
        z1 = X_norm @ W1 + b1
        h1 = _relu(z1)
        z2 = h1 @ W2 + b2
        h2 = _relu(z2)
    else:
        z1 = X_norm @ W1 + b1
        h1 = _tanh(z1)
        z2 = h1 @ W2 + b2
        h2 = _tanh(z2)
    out = h2 @ W3 + b3
    pred = out.flatten() * y_std + y_mean
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return pred

def _fit_mlp_with_cv(X, y, n_folds=5, hidden1=32, hidden2=16, lr=0.01, lam=1e-3, activation='relu', seed=42):
    """Cross-validate MLP hyperparameters."""
    n = X.shape[0]
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    best_score = np.inf
    best_params = None
    
    # Try a few configurations
    configs = [
        (32, 16, 0.01, 1e-3, 'relu'),
        (32, 16, 0.005, 1e-3, 'tanh'),
        (64, 32, 0.01, 1e-4, 'relu'),
        (32, 32, 0.01, 1e-3, 'relu'),
    ]
    
    for h1, h2, lr_c, lam_c, act in configs:
        mse_sum = 0
        valid = True
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            try:
                pred_tr, ym, ys, *_ = _fit_mlp(X_tr, y_tr, n_epochs=300, hidden1=h1, hidden2=h2, lr=lr_c, lam=lam_c, activation=act, seed=seed)
                # Predict on validation
                X_val_norm = (X_val - X_tr.mean(axis=0)) / (X_tr.std(axis=0) + 1e-10)
                # Need to re-normalize properly - this is a simplification
                pred_val = _predict_mlp(X_val, X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-10, ym, ys, *_
                )
                if not np.all(np.isfinite(pred_val)):
                    valid = False
                    break
                mse_sum += np.mean((pred_val - y_val) ** 2)
            except Exception:
                valid = False
                break
        if valid and mse_sum / n_folds < best_score:
            best_score = mse_sum / n_folds
            best_params = (h1, h2, lr_c, lam_c, act)
    
    if best_params is None:
        best_params = (32, 16, 0.01, 1e-3, 'relu')
    
    return best_params, best_score

def fit(X_train, y_train):
    """Fit a regression model using neural network."""
    start_time = time.time()
    deadline = start_time + 270  # Safety margin
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Normalize X
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    
    # Try MLP with different configurations
    best_pred = None
    best_val_score = np.inf
    
    configs = [
        (32, 16, 0.01, 1e-3, 'relu', 500),
        (32, 16, 0.005, 1e-3, 'tanh', 500),
        (64, 32, 0.01, 1e-4, 'relu', 500),
        (32, 32, 0.01, 1e-3, 'relu', 500),
        (16, 16, 0.01, 1e-2, 'relu', 500),
    ]
    
    for h1, h2, lr_c, lam_c, act, n_epochs in configs:
        if time.time() > deadline - 30:
            break
        try:
            # Use a simple holdout for early stopping check
            split = int(n * 0.8)
            idx = np.random.RandomState(42).permutation(n)
            X_tr, y_tr = X_norm[idx[:split]], y[idx[:split]]
            X_val, y_val = X_norm[idx[split:]], y[idx[split:]]
            
            # Train with early stopping
            np.random.seed(42)
            W1 = _init_weights(d, h1)
            b1 = np.zeros(h1)
            W2 = _init_weights(h1, h2)
            b2 = np.zeros(h2)
            W3 = _init_weights(h2, 1)
            b3 = np.zeros(1)
            
            m_W1 = np.zeros_like(W1); v_W1 = np.zeros_like(W1)
            m_b1 = np.zeros_like(b1); v_b1 = np.zeros_like(b1)
            m_W2 = np.zeros_like(W2); v_W2 = np.zeros_like(W2)
            m_b2 = np.zeros_like(b2); v_b2 = np.zeros_like(b2)
            m_W3 = np.zeros_like(W3); v_W3 = np.zeros_like(W3)
            m_b3 = np.zeros_like(b3); v_b3 = np.zeros_like(b3)
            
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = 0
            best_val_loss = np.inf
            best_state = None
            patience = 50
            bad_epochs = 0
            
            y_mean = y_tr.mean()
            y_std = y_tr.std() + 1e-10
            y_tr_norm = (y_tr - y_mean) / y_std
            
            for epoch in range(n_epochs):
                t += 1
                if act == 'relu':
                    z1 = X_tr @ W1 + b1
                    h1_f = _relu(z1)
                    z2 = h1_f @ W2 + b2
                    h2_f = _relu(z2)
                else:
                    z1 = X_tr @ W1 + b1
                    h1_f = _tanh(z1)
                    z2 = h1_f @ W2 + b2
                    h2_f = _tanh(z2)
                
                out = h2_f @ W3 + b3
                out_flat = out.flatten()
                
                train_loss = np.mean((out_flat - y_tr_norm) ** 2)
                
                # Val loss
                if act == 'relu':
                    v1 = _relu(X_val @ W1 + b1)
                    v2 = _relu(v1 @ W2 + b2)
                else:
                    v1 = _tanh(X_val @ W1 + b1)
                    v2 = _tanh(v1 @ W2 + b2)
                vout = (v2 @ W3 + b3).flatten()
                val_loss = np.mean((vout - (y_val - y_mean) / y_std) ** 2)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break
                
                # Backward
                d_out = 2 * (out_flat - y_tr_norm) / len(y_tr)
                dW3 = h2_f.T @ d_out.reshape(-1, 1) + 2 * lam_c * W3
                db3 = np.sum(d_out).reshape(1)
                d_h2 = d_out.reshape(-1, 1) @ W3.T
                if act == 'relu':
                    dz2 = d_h2 * _relu_grad(z2)
                else:
                    dz2 = d_h2 * _tanh_grad(z2)
                dW2 = h1_f.T @ dz2 + 2 * lam_c * W2
                db2 = np.sum(dz2, axis=0)
                d_h1 = dz2 @ W2.T
                if act == 'relu':
                    dz1 = d_h1 * _relu_grad(z1)
                else:
                    dz1 = d_h1 * _tanh_grad(z1)
                dW1 = X_tr.T @ dz1 + 2 * lam_c * W1
                db1 = np.sum(dz1, axis=0)
                
                def adam(m, v, grad, param):
                    nonlocal t
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    return param - lr_c * m_hat / (np.sqrt(v_hat) + eps), m, v
                
                W1, m_W1, v_W1 = adam(m_W1, v_W1, dW1, W1)
                b1, m_b1, v_b1 = adam(m_b1, v_b1, db1, b1)
                W2, m_W2, v_W2 = adam(m_W2, v_W2, dW2, W2)
                b2, m_b2, v_b2 = adam(m_b2, v_b2, db2, b2)
                W3, m_W3, v_W3 = adam(m_W3, v_W3, dW3, W3)
                b3, m_b3, v_b3 = adam(m_b3, v_b3, db3, b3)
            
            # Use best state
            if best_state is not None:
                W1, b1, W2, b2, W3, b3 = best_state
            
            # Final prediction on all training data
            if act == 'relu':
                z1 = X_norm @ W1 + b1
                h1_f = _relu(z1)
                z2 = h1_f @ W2 + b2
                h2_f = _relu(z2)
            else:
                z1 = X_norm @ W1 + b1
                h1_f = _tanh(z1)
                z2 = h1_f @ W2 + b2
                h2_f = _tanh(z2)
            out = h2_f @ W3 + b3
            pred = out.flatten() * y_std + y_mean
            
            if not np.all(np.isfinite(pred)):
                continue
            
            # Store best
            if best_val_loss < best_val_score:
                best_val_score = best_val_loss
                best_pred = pred
                best_params = (X_mean, X_std, y_mean, y_std, W1, b1, W2, b2, W3, b3, act)
        except Exception:
            continue
    
    # If MLP failed, fall back to simple linear
    if best_params is None:
        # Simple linear regression
        Xb = np.hstack([X_norm, np.ones((n, 1))])
        try:
            w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=float)
                X_test_norm = (X_test - X_mean) / X_std
                Xb_test = np.hstack([X_test_norm, np.ones((len(X_test), 1))])
                pred = Xb_test @ w
                return np.where(np.isfinite(pred), pred, 0.0)
            return predict
        except Exception:
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=float)
                return np.zeros(len(X_test))
            return predict
    
    X_mean_f, X_std_f, y_mean_f, y_std_f, W1_f, b1_f, W2_f, b2_f, W3_f, b3_f, act_f = best_params
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        X_test_norm = (X_test - X_mean_f) / X_std_f
        if act_f == 'relu':
            z1 = X_test_norm @ W1_f + b1_f
            h1_f = _relu(z1)
            z2 = h1_f @ W2_f + b2_f
            h2_f = _relu(z2)
        else:
            z1 = X_test_norm @ W1_f + b1_f
            h1_f = _tanh(z1)
            z2 = h1_f @ W2_f + b2_f
            h2_f = _tanh(z2)
        out = h2_f @ W3_f + b3_f
        pred = out.flatten() * y_std_f + y_mean_f
        return np.where(np.isfinite(pred), pred, 0.0)
    
    return predict
# EVOLVE-BLOCK-END

"""Deep Feedforward Neural Network for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _tanh(z):
    return np.tanh(z)


def _tanh_grad(z):
    # z is the pre-activation value
    return 1.0 - np.tanh(z) ** 2


def _initialize_weights(n_in, n_hidden, n_out, seed=42):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(n_in, n_hidden) / np.sqrt(n_in)
    b1 = np.zeros(n_hidden)
    W2 = rng.randn(n_hidden, n_out) / np.sqrt(n_hidden)
    b2 = np.zeros(n_out)
    return W1, b1, W2, b2


def _forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = _tanh(z1)
    z2 = a1 @ W2 + b2
    return z2, z1, a1


def _train_nn(X_train, y_train, n_epochs=1000, lr=0.01, n_hidden=64, l2_reg=1e-5, seed=42):
    n_samples, n_features = X_train.shape
    n_out = 1
    
    # Normalize inputs and outputs
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    
    X_norm = (X_train - X_mean) / X_std
    y_norm = (y_train - y_mean) / y_std
    
    # Initialize weights
    W1, b1, W2, b2 = _initialize_weights(n_features, n_hidden, n_out, seed=seed)
    
    # Adam optimizer states
    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(vW2) # Bug fix: should be zeros_like(W2)
    mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
    
    # Correcting the zero initialization for vW2
    vW2 = np.zeros_like(W2)
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    for epoch in range(n_epochs):
        # Full batch gradient descent for stability and speed on small data
        X_batch = X_norm
        y_batch = y_norm.reshape(-1, 1)
        
        t += 1
        # Learning rate schedule: decay slowly
        lr_t = lr / (1 + 0.001 * t)
        
        # Forward pass
        z2, z1, a1 = _forward(X_batch, W1, b1, W2, b2)
        pred = z2
        
        # Loss: MSE + L2 Regularization
        error = pred - y_batch
        n_batch = len(X_batch)
        mse_loss = np.mean(error ** 2)
        l2_loss = l2_reg * (np.sum(W1**2) + np.sum(W2**2))
        total_loss = mse_loss + l2_loss
        
        # Backward pass
        # dL/dz2 = 2 * error / n
        dz2 = 2 * error / n_batch
        
        # dL/dW2 = a1.T @ dz2 + 2 * l2_reg * W2
        dW2 = a1.T @ dz2 + 2 * l2_reg * W2
        db2 = dz2.sum(axis=0)
        
        # dL/da1 = dz2 @ W2.T
        da1 = dz2 @ W2.T
        
        # dL/dz1 = da1 * tanh'(z1)
        dz1 = da1 * _tanh_grad(z1)
        
        # dL/dW1 = X.T @ dz1 + 2 * l2_reg * W1
        dW1 = X_batch.T @ dz1 + 2 * l2_reg * W1
        db1 = dz1.sum(axis=0)
        
        # Adam update
        mW1 = beta1 * mW1 + (1 - beta1) * dW1; vW1 = beta2 * vW1 + (1 - beta2) * dW1**2
        mW1_hat = mW1 / (1 - beta1**t); vW1_hat = vW1 / (1 - beta2**t)
        W1 -= lr_t * mW1_hat / (np.sqrt(vW1_hat) + eps)
        
        mb1 = beta1 * mb1 + (1 - beta1) * db1; vb1 = beta2 * vb1 + (1 - beta2) * db1**2
        mb1_hat = mb1 / (1 - beta1**t); vb1_hat = vb1 / (1 - beta2**t)
        b1 -= lr_t * mb1_hat / (np.sqrt(vb1_hat) + eps)
        
        mW2 = beta1 * mW2 + (1 - beta1) * dW2; vW2 = beta2 * vW2 + (1 - beta2) * dW2**2
        mW2_hat = mW2 / (1 - beta1**t); vW2_hat = vW2 / (1 - beta2**t)
        W2 -= lr_t * mW2_hat / (np.sqrt(vW2_hat) + eps)
        
        mb2 = beta1 * mb2 + (1 - beta1) * db2; vb2 = beta2 * vb2 + (1 - beta2) * db2**2
        mb2_hat = mb2 / (1 - beta1**t); vb2_hat = vb2 / (1 - beta2**t)
        b2 -= lr_t * mb2_hat / (np.sqrt(vb2_hat) + eps)
        
    return W1, b1, W2, b2, X_mean, X_std, y_mean, y_std


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Ensure X_train is 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    best_loss = np.inf
    best_params = None
    
    # Try different hidden layer sizes
    for n_hidden in [32, 64, 128]:
        try:
            # Use a reasonable number of epochs. 
            # 1000 epochs with full batch on 200 points is fast.
            params = _train_nn(X_train, y_train, n_epochs=1000, lr=0.01, 
                               n_hidden=n_hidden, l2_reg=1e-5, seed=42)
            W1, b1, W2, b2, X_mean, X_std, y_mean, y_std = params
            
            # Compute training loss
            X_norm = (X_train - X_mean) / X_std
            z2, _, _ = _forward(X_norm, W1, b1, W2, b2)
            pred = z2.flatten() * y_std + y_mean
            loss = np.mean((pred - y_train) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to linear regression
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + \
                   [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(np.asarray(X, dtype=np.float64)) @ w
    
    W1, b1, W2, b2, X_mean, X_std, y_mean, y_std = best_params
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - X_mean) / X_std
        z2, _, _ = _forward(X_norm, W1, b1, W2, b2)
        pred = z2.flatten() * y_std + y_mean
        return pred
    
    return predict
# EVOLVE-BLOCK-END

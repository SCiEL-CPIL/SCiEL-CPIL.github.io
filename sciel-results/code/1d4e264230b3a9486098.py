"""Symbolic regression via Sparse Neural Network with Adam optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target for stable training
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Define a small neural network architecture
    # Input: d -> Hidden1: 32 -> Hidden2: 32 -> Output: 1
    # Using tanh activations which are good for general function approximation
    hidden_size = 32
    
    # Initialize weights using He initialization for tanh
    def init_weights(fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    
    W1 = init_weights(d, hidden_size)
    b1 = np.zeros(hidden_size)
    W2 = init_weights(hidden_size, hidden_size)
    b2 = np.zeros(hidden_size)
    W3 = init_weights(hidden_size, 1)
    b3 = np.zeros(1)
    
    params = [W1, b1, W2, b2, W3, b3]
    param_grads = [np.zeros_like(p) for p in params]
    
    # Adam optimizer state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    def forward_pass(X_input):
        h1 = np.tanh(X_input @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = h2 @ W3 + b3
        return out, h1, h2
    
    def compute_loss_and_grads(X_input, y_target):
        # Forward
        pred, h1, h2 = forward_pass(X_input)
        loss = np.mean((pred.flatten() - y_target) ** 2)
        
        # Backward
        # dLoss/dPred
        d_pred = 2.0 * (pred.flatten() - y_target) / len(y_target)
        d_pred = d_pred.reshape(-1, 1)
        
        # dLoss/dW3, dLoss/db3
        d_W3 = h2.T @ d_pred
        d_b3 = np.sum(d_pred, axis=0)
        
        # dLoss/dh2
        d_h2 = d_pred @ W3.T
        # dLoss/dPreAct2
        d_pre_act2 = d_h2 * (1.0 - h2**2)
        
        # dLoss/dW2, dLoss/db2
        d_W2 = h1.T @ d_pre_act2
        d_b2 = np.sum(d_pre_act2, axis=0)
        
        # dLoss/dh1
        d_h1 = d_pre_act2 @ W2.T
        # dLoss/dPreAct1
        d_pre_act1 = d_h1 * (1.0 - h1**2)
        
        # dLoss/dW1, dLoss/db1
        d_W1 = X_input.T @ d_pre_act1
        d_b1 = np.sum(d_pre_act1, axis=0)
        
        grads = [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]
        return loss, grads

    # Training loop
    best_loss = np.inf
    best_params = [p.copy() for p in params]
    
    # We will train for as long as we have time, but check periodically
    # Since N=200 and D is small, one full batch is cheap.
    # We can do many steps.
    
    while time.time() < DEADLINE:
        # Compute gradients
        loss, grads = compute_loss_and_grads(X_norm, y_norm)
        
        if loss < best_loss:
            best_loss = loss
            best_params = [p.copy() for p in params]
            
        # Update Adam state
        t += 1
        for i in range(len(params)):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            params[i] = params[i] - lr * m_hat / (np.sqrt(v_hat) + eps)
            
        # Early stopping if loss is very low
        if best_loss < 1e-10:
            break
            
    # Restore best parameters
    for i in range(len(params)):
        params[i] = best_params[i]
        
    W1, b1, W2, b2, W3, b3 = params
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        
        h1 = np.tanh(X_test_norm @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out_norm = h2 @ W3 + b3
        out = out_norm.flatten() * y_std + y_mean
        return out

    return predict
# EVOLVE-BLOCK-END

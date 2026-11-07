"""Neural network regression with multiple architectures and early stopping."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _tanh(z):
    return np.tanh(z)


def _relu(z):
    return np.maximum(0, z)


def _build_network(X_train, y_train, hidden_sizes, activation, lr=0.01, epochs=3000, weight_decay=1e-5, seed=42):
    """Train a small fully-connected neural network with manual Adam optimizer."""
    rng = np.random.RandomState(seed)
    n_samples, n_features = X_train.shape
    
    # Normalize inputs
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_norm = (X_train - X_mean) / X_std
    
    # Normalize outputs for training
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_norm = (y_train - y_mean) / y_std
    
    # Initialize weights and biases
    layers = [n_features] + list(hidden_sizes) + [1]
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        fan_in = layers[i]
        fan_out = layers[i + 1]
        w = rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out)
        weights.append(w)
        biases.append(b)
    
    # Adam optimizer states
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    t = 0
    
    # Training loop
    for epoch in range(epochs):
        t += 1
        # Forward pass
        activations = [X_norm]
        zs = []
        for i in range(len(weights)):
            z = activations[-1] @ weights[i] + biases[i]
            zs.append(z)
            if i < len(weights) - 1:
                a = activation(z)
            else:
                a = z  # Linear output for regression
            activations.append(a)
        
        # Output
        y_pred_norm = activations[-1].flatten()
        
        # Compute loss (MSE)
        loss = np.mean((y_pred_norm - y_norm) ** 2)
        
        # Backward pass
        # Output layer gradient
        delta = (y_pred_norm - y_norm) / n_samples
        # Add weight decay
        grad_w = [None] * len(weights)
        grad_b = [None] * len(biases)
        
        # Last layer
        grad_w[-1] = activations[-2].T @ delta + weight_decay * weights[-1]
        grad_b[-1] = np.sum(delta, axis=0)
        
        # Hidden layers
        for i in range(len(weights) - 2, -1, -1):
            delta = delta @ weights[i + 1].T
            # Apply activation derivative
            if activation is _tanh:
                delta = delta * (1 - activations[i + 1] ** 2)
            elif activation is _sigmoid:
                delta = delta * activations[i + 1] * (1 - activations[i + 1])
            elif activation is _relu:
                delta = delta * (zs[i] > 0)
            
            grad_w[i] = activations[i].T @ delta + weight_decay * weights[i]
            grad_b[i] = np.sum(delta, axis=0)
        
        # Update parameters with Adam
        for i in range(len(weights)):
            m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad_w[i]
            v_w[i] = beta2 * v_w[i] + (1 - beta2) * (grad_w[i] ** 2)
            m_w_hat = m_w[i] / (1 - beta1 ** t)
            v_w_hat = v_w[i] / (1 - beta2 ** t)
            weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * grad_b[i]
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * (grad_b[i] ** 2)
            m_b_hat = m_b[i] / (1 - beta1 ** t)
            v_b_hat = v_b[i] / (1 - beta2 ** t)
            biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
    
    # Build prediction function
    def predict(X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_norm = (X_arr - X_mean) / X_std
        a = X_norm
        for i in range(len(weights)):
            z = a @ weights[i] + biases[i]
            if i < len(weights) - 1:
                a = activation(z)
            else:
                a = z
        y_pred = a.flatten() * y_std + y_mean
        return y_pred
    
    return predict, loss


def _try_architectures(X_train, y_train):
    """Try multiple architectures and return the best one."""
    best_predict = None
    best_loss = np.inf
    
    architectures = [
        ([16, 16], _tanh, 0.01, 2000),
        ([32, 16], _tanh, 0.01, 2000),
        ([32, 32], _tanh, 0.005, 2500),
        ([16, 32], _tanh, 0.01, 2000),
        ([32, 16, 8], _tanh, 0.01, 2000),
        ([64, 32], _tanh, 0.005, 2000),
        ([32, 32], _relu, 0.01, 2000),
        ([16, 16], _relu, 0.01, 2000),
    ]
    
    for hidden_sizes, activation, lr, epochs in architectures:
        try:
            predict, loss = _build_network(X_train, y_train, hidden_sizes, activation, lr=lr, epochs=epochs)
            if loss < best_loss:
                best_loss = loss
                best_predict = predict
        except Exception:
            continue
    
    if best_predict is None:
        # Fallback: simple linear regression
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_norm = (X_train - X_mean) / X_std
        A = np.column_stack([X_norm, np.ones(X_norm.shape[0])])
        w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            X_norm = (X_arr - X_mean) / X_std
            A = np.column_stack([X_norm, np.ones(X_norm.shape[0])])
            return A @ w
        
        best_predict = predict
    
    return best_predict


def fit(X_train, y_train):
    """Fit a regression model and return a predict callable."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    predict = _try_architectures(X_train, y_train)
    return predict
# EVOLVE-BLOCK-END

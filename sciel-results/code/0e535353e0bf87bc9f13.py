"""Symbolic regression over hidden equations using GP + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def _mlp_forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = np.tanh(Z1)
    Z2 = A1 @ W2 + b2
    return Z2.flatten(), A1

def _fit_mlp(X_train, y_train, n_restarts=5, hidden_size=32, reg_lambda=1e-4):
    n, d = X_train.shape
    
    # Normalize
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-10:
        # Constant function
        return lambda X: np.full(len(X), y_mean)
    y_norm = (y_train - y_mean) / y_std
    
    def loss_and_grad(params):
        h = hidden_size
        W1 = params[0:d*h].reshape(d, h)
        b1 = params[d*h:d*h+h]
        W2 = params[d*h+h:d*h+2*h].reshape(h, 1)
        b2 = params[d*h+2*h:d*h+2*h+1]
        
        # Forward
        preds, A1 = _mlp_forward(X_norm, W1, b1, W2, b2)
        
        # Loss (MSE + L2 Reg)
        mse = np.mean((preds - y_norm) ** 2)
        reg = reg_lambda * (np.sum(W1**2) + np.sum(W2**2))
        loss = mse + reg
        
        # Gradients
        dZ2 = 2.0 / n * (preds - y_norm).reshape(-1, 1)
        
        dW2 = A1.T @ dZ2 + 2.0 * reg_lambda * W2
        db2 = np.sum(dZ2, axis=0)
        
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (1 - A1**2)
        
        dW1 = X_norm.T @ dZ1 + 2.0 * reg_lambda * W1
        db1 = np.sum(dZ1, axis=0)
        
        grad = np.concatenate([dW1.flatten(), db1, dW2.flatten(), db2])
        
        return loss, grad

    best_params = None
    best_loss = np.inf
    
    for i in range(n_restarts):
        # Deterministic seed for reproducibility of the restarts themselves
        np.random.seed(42 + i)
        W1_init = np.random.randn(d, hidden_size) * np.sqrt(2.0 / d)
        b1_init = np.zeros(hidden_size)
        W2_init = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        b2_init = np.zeros(1)
        
        x0 = np.concatenate([W1_init.flatten(), b1_init, W2_init.flatten(), b2_init])
        
        try:
            res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except:
            continue

    if best_params is None:
        # Fallback: Linear Regression
        X_aug = np.column_stack([np.ones(n), X_train])
        w_lin = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
        def predict_fallback(X):
            X_aug_test = np.column_stack([np.ones(len(X)), X])
            return X_aug_test @ w_lin
        return predict_fallback

    h = hidden_size
    W1 = best_params[0:d*h].reshape(d, h)
    b1 = best_params[d*h:d*h+h]
    W2 = best_params[d*h+h:d*h+2*h].reshape(h, 1)
    b2 = best_params[d*h+2*h:d*h+2*h+1]
    
    def predict_mlp(X):
        X_norm_ = (X - X_mean) / X_std
        preds, _ = _mlp_forward(X_norm_, W1, b1, W2, b2)
        return preds * y_std + y_mean
    
    return predict_mlp

def fit(X_train, y_train):
    n, d = X_train.shape
    try:
        return _fit_mlp(X_train, y_train, n_restarts=5, hidden_size=32, reg_lambda=1e-4)
    except Exception:
        # Ultimate fallback
        X_aug = np.column_stack([np.ones(n), X_train])
        w_lin = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
        def predict_fallback(X):
            X_aug_test = np.column_stack([np.ones(len(X)), X])
            return X_aug_test @ w_lin
        return predict_fallback
# EVOLVE-BLOCK-END

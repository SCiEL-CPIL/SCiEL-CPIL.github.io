"""Neural ODE based symbolic regression: learn smooth continuous mappings via RK4 integration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# Deadline guard: 330s wall-clock limit, use 290s for safety
_DEADLINE = time.time() + 290.0

def _remaining_time():
    return _DEADLINE - time.time()

def _mlp_forward(x, W1, b1, W2, b2, W3, b3):
    """3-layer MLP with tanh activations."""
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    out = h2 @ W3 + b3
    return out

def _mlp_jacobian(x, W1, b1, W2, b2, W3, b3):
    """Compute Jacobian of 3-layer MLP for RK4 step."""
    # x: (n, 1)
    # W1: (1, h1), b1: (h1,)
    # W2: (h1, h2), b2: (h2,)
    # W3: (h2, 1), b3: (1,)
    
    n = x.shape[0]
    h1_size = W1.shape[1]
    h2_size = W2.shape[1]
    
    # Forward pass with intermediates
    z1 = x @ W1 + b1  # (n, h1)
    a1 = np.tanh(z1)  # (n, h1)
    z2 = a1 @ W2 + b2  # (n, h2)
    a2 = np.tanh(z2)  # (n, h2)
    y = a2 @ W3 + b3  # (n, 1)
    
    # Jacobian: dy/dx
    # dy/dz2 = a2 @ W3 (element-wise)
    # dz2/da1 = W2
    # da1/dz1 = 1 - a1^2
    # dz1/dx = W1
    
    # For each sample i, J[i] = (1 - a1[i]^2) * (W2 * (W3 * a2[i]))^T * W1
    # Actually: J = W1 * diag(1-a1^2) @ W2 * diag(a2) @ W3
    # But W3 is (h2, 1), so this is (1, h1) @ (h1, h1) @ (h1, h2) @ (h2, 1) -> (1, 1)
    # Wait, x is scalar input, y is scalar output. So Jacobian is scalar.
    
    # dy/dx = sum over paths
    # dy/dx = W3.T @ diag(a2) @ W2.T @ diag(1-a1^2) @ W1.T
    # All shapes: (1, h2) @ (h2, h2) @ (h2, h1) @ (h1, h1) @ (h1, 1) -> (1, 1)
    
    # Vectorized:
    # da2/dz2 = 1 - a2^2 (not needed for output, but for backprop)
    # Actually for Jacobian dy/dx:
    # dy/dz2 = a2 * W3 (element-wise per sample)
    # dz2/da1 = W2
    # da1/dz1 = 1 - a1^2
    # dz1/dx = W1
    
    # J = W1 * (1-a1^2) @ W2 * (a2) @ W3
    # Let's compute per sample:
    # J_i = W1[0, :] * (1 - a1[i, :]^2) @ W2 * (a2[i, :] * W3[:, 0])
    
    # More efficient:
    # grad_y_w3 = a2  # (n, h2)
    # grad_z2 = grad_y_w3 * W3[0, :]  # (n, h2) -- wait W3 is (h2, 1)
    # grad_z2 = a2 * W3.ravel()  # (n, h2)
    # grad_a1 = grad_z2 @ W2.T  # (n, h1)
    # grad_z1 = grad_a1 * (1 - a1**2)  # (n, h1)
    # grad_x = grad_z1 @ W1.T  # (n, 1)
    
    grad_z2 = a2 * W3.ravel()  # (n, h2)
    grad_a1 = grad_z2 @ W2.T  # (n, h1)
    grad_z1 = grad_a1 * (1 - a1**2)  # (n, h1)
    grad_x = grad_z1 @ W1.T  # (n, 1)
    
    return y, grad_x

def _rk4_step(x, t, dt, params):
    """Single RK4 step for ODE dy/dt = f(x, t) where f = mlp(x)."""
    W1, b1, W2, b2, W3, b3 = params
    
    k1, _ = _mlp_jacobian(x, W1, b1, W2, b2, W3, b3)
    k1 = k1.ravel()
    
    x2 = x + 0.5 * dt * k1
    k2, _ = _mlp_jacobian(x2, W1, b1, W2, b2, W3, b3)
    k2 = k2.ravel()
    
    x3 = x + 0.5 * dt * k2
    k3, _ = _mlp_jacobian(x3, W1, b1, W2, b2, W3, b3)
    k3 = k3.ravel()
    
    x4 = x + dt * k3
    k4, _ = _mlp_jacobian(x4, W1, b1, W2, b2, W3, b3)
    k4 = k4.ravel()
    
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _neural_ode_predict(x, params, n_steps=10, t0=0.0, t1=1.0):
    """Integrate ODE from t0 to t1 using RK4."""
    W1, b1, W2, b2, W3, b3 = params
    dt = (t1 - t0) / n_steps
    x = x.copy()
    for i in range(n_steps):
        t = t0 + i * dt
        x = _rk4_step(x, t, dt, params)
    return x

def _compute_loss_and_grads(x_train, y_train, params, n_steps=10):
    """Compute MSE loss and gradients via backprop through RK4."""
    W1, b1, W2, b2, W3, b3 = params
    n = x_train.shape[0]
    
    # Forward pass with stored intermediates for backprop
    # This is complex for RK4. Use finite differences or autograd-like approach.
    # For simplicity, use central finite differences for gradients.
    
    pred = _neural_ode_predict(x_train, params, n_steps)
    loss = np.mean((pred - y_train)**2)
    
    # Finite difference gradients
    eps = 1e-6
    grads = []
    for i, param in enumerate(params):
        grad = np.zeros_like(param)
        flat_idx = 0
        for idx in np.ndindex(param.shape):
            old_val = param[idx]
            param[idx] = old_val + eps
            pred_plus = _neural_ode_predict(x_train, params, n_steps)
            loss_plus = np.mean((pred_plus - y_train)**2)
            
            param[idx] = old_val - eps
            pred_minus = _neural_ode_predict(x_train, params, n_steps)
            loss_minus = np.mean((pred_minus - y_train)**2)
            
            param[idx] = old_val
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            flat_idx += 1
        grads.append(grad)
    
    return loss, grads

def _fit_neural_ode(x_train, y_train, time_budget):
    """Fit Neural ODE using Adam optimizer with finite difference gradients."""
    n, d = x_train.shape
    
    # Initialize parameters
    np.random.seed(42)
    h1_size = 20
    h2_size = 20
    
    W1 = np.random.randn(d, h1_size) * 0.1
    b1 = np.zeros(h1_size)
    W2 = np.random.randn(h1_size, h2_size) * 0.1
    b2 = np.zeros(h2_size)
    W3 = np.random.randn(h2_size, d) * 0.1
    b3 = np.zeros(d)
    
    params = [W1, b1, W2, b2, W3, b3]
    
    # Adam optimizer state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    lr = 0.01
    t_adam = 0
    
    best_params = [p.copy() for p in params]
    best_loss = np.inf
    
    start_time = time.time()
    
    # Use fewer steps for speed
    n_steps = 5
    
    for step in range(2000):
        if time.time() > time_budget:
            break
        
        # Compute loss and gradients
        loss, grads = _compute_loss_and_grads(x_train, y_train, params, n_steps)
        
        if np.isnan(loss) or np.isinf(loss):
            # Reset to best params
            for i, p in enumerate(params):
                params[i] = best_params[i].copy()
            continue
        
        if loss < best_loss:
            best_loss = loss
            best_params = [p.copy() for p in params]
        
        # Adam update
        t_adam += 1
        for i in range(len(params)):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i]**2
            m_hat = m[i] / (1 - beta1**t_adam)
            v_hat = v[i] / (1 - beta2**t_adam)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
    
    return best_params, best_loss

def _try_polynomial(x_train, y_train, max_deg=6):
    """Try polynomial regression as baseline."""
    n, d = x_train.shape
    cols = [np.ones(n)]
    for deg in range(1, max_deg + 1):
        for i in range(d):
            cols.append(x_train[:, i] ** deg)
    F = np.column_stack(cols)
    try:
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        pred = F @ w
        mse = np.mean((pred - y_train)**2)
        return w, mse, F
    except Exception:
        return None, np.inf, None

def fit(X_train, y_train):
    global _DEADLINE
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_var = 1e-12
    
    best_predict = lambda X: np.full(X.shape[0], y_mean)
    best_score = np.inf
    
    # Time allocation: 60% for Neural ODE, 40% for polynomial fallback
    total_time = _remaining_time()
    ode_time = total_time * 0.6
    poly_time = total_time * 0.4
    
    # Try Neural ODE
    try:
        params, loss = _fit_neural_ode(X_train, y_train, time.time() + ode_time)
        
        def predict_ode(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            pred = _neural_ode_predict(X, params, n_steps=10)
            pred = pred.ravel()
            pred = np.where(np.isfinite(pred), pred, y_mean)
            return pred
        
        # Evaluate on training data
        pred_train = predict_ode(X_train)
        mse_train = np.mean((pred_train - y_train)**2)
        score = mse_train / y_var
        
        if score < best_score and np.isfinite(score):
            best_score = score
            best_predict = predict_ode
            
    except Exception:
        pass
    
    # Try polynomial fallback
    try:
        w, mse, F = _try_polynomial(X_train, y_train, max_deg=6)
        if w is not None:
            def predict_poly(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                n, d = X.shape
                cols = [np.ones(n)]
                for deg in range(1, 7):
                    for i in range(d):
                        cols.append(X[:, i] ** deg)
                F = np.column_stack(cols)
                pred = F @ w
                pred = np.where(np.isfinite(pred), pred, y_mean)
                return pred
            
            pred_train = predict_poly(X_train)
            mse_train = np.mean((pred_train - y_train)**2)
            score = mse_train / y_var
            
            if score < best_score and np.isfinite(score):
                best_score = score
                best_predict = predict_poly
                
    except Exception:
        pass
    
    # Final safety check
    def safe_predict(X):
        try:
            result = best_predict(X)
            result = np.where(np.isfinite(result), result, y_mean)
            return result
        except Exception:
            return np.full(X.shape[0], y_mean)
    
    return safe_predict
# EVOLVE-BLOCK-END

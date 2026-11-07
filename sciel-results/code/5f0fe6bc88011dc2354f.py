"""Neural-network based symbolic regression fit for 10 hidden equations."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def fit(X_train, y_train):
    """
    Fit a small fully-connected neural network to (X_train, y_train) and return
    a predict callable.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    n, d = X.shape

    # Normalize inputs and target for stable training
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0

    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std

    # Small MLP: 2 hidden layers, 16 units each, tanh activation
    hidden_sizes = [16, 16]

    def _init_weights(seed=42):
        rng = np.random.RandomState(seed)
        W = []
        b = []
        prev = d
        for h in hidden_sizes:
            W.append(rng.randn(prev, h) * np.sqrt(2.0 / prev))
            b.append(np.zeros(h))
            prev = h
        W.append(rng.randn(prev, 1) * np.sqrt(2.0 / prev))
        b.append(np.zeros(1))
        return W, b

    def _forward(X, W, b):
        h = X
        for i in range(len(W)):
            h = np.tanh(h @ W[i] + b[i])
        return h

    def _predict_raw(X_raw, W, b):
        Xn_local = (X_raw - x_mean) / x_std
        out = _forward(Xn_local, W, b)
        return out[:, 0] * y_std + y_mean

    # Train via L-BFGS-B on MSE loss (noiseless data, so exact fit possible)
    W0, b0 = _init_weights(seed=0)
    params0 = np.concatenate([w.flatten() for w in W0] + [b.flatten() for b in b0])

    def _unflatten(params):
        W = []
        b = []
        idx = 0
        prev = d
        for h in hidden_sizes:
            size = prev * h
            W.append(params[idx:idx + size].reshape(prev, h))
            idx += size
            size_b = h
            b.append(params[idx:idx + size_b].reshape(h))
            idx += size_b
            prev = h
        size = prev * 1
        W.append(params[idx:idx + size].reshape(prev, 1))
        idx += size
        size_b = 1
        b.append(params[idx:idx + size_b].reshape(1))
        idx += size_b
        return W, b

    def _loss_grad(params):
        W, b = _unflatten(params)
        # Forward
        activations = [Xn]
        h = Xn
        for i in range(len(W)):
            z = h @ W[i] + b[i]
            h = np.tanh(z)
            activations.append(h)
        pred = h[:, 0]
        # MSE loss
        loss = np.mean((pred - yn) ** 2)
        # Backward
        m = Xn.shape[0]
        dL = 2.0 * (pred - yn) / m  # shape (m,1)
        grads = []
        for i in reversed(range(len(W))):
            a_prev = activations[i]
            dz = dL * (1 - h ** 2) if i < len(W) - 1 else dL
            # Wait: for output layer, activation is also tanh, so derivative is (1 - h^2)
            # Actually all layers use tanh, so:
            dz = dL * (1.0 - h ** 2)
            dW = a_prev.T @ dz
            db = np.sum(dz, axis=0, keepdims=True)
            grads.append(dW.flatten())
            grads.append(db.flatten())
            if i > 0:
                dL = dz @ W[i].T
            h = a_prev
        grads = grads[::-1]  # reverse because we went backward
        # Actually let's redo this more carefully
        return loss, np.concatenate(grads)

    # Let me redo the gradient computation more carefully
    def _loss_grad(params):
        W, b = _unflatten(params)
        m = Xn.shape[0]

        # Forward pass, store all activations
        acts = [Xn]
        h = Xn
        for i in range(len(W)):
            z = h @ W[i] + b[i]
            h = np.tanh(z)
            acts.append(h)

        pred = h[:, 0]
        loss = np.mean((pred - yn) ** 2)

        # Backward pass
        # dL/dpred = 2*(pred - yn)/m, shape (m,1)
        delta = 2.0 * (pred - yn) / m  # (m, 1)

        grads_W = []
        grads_b = []

        for i in reversed(range(len(W))):
            h_i = acts[i + 1]  # output of layer i
            # tanh derivative
            delta = delta * (1.0 - h_i ** 2)  # (m, hidden_size_i)
            a_prev = acts[i]
            dW_i = a_prev.T @ delta  # (prev, hidden_i)
            db_i = np.sum(delta, axis=0, keepdims=True)  # (1, hidden_i)
            grads_W.insert(0, dW_i.flatten())
            grads_b.insert(0, db_i.flatten())
            if i > 0:
                delta = delta @ W[i].T  # (m, prev)

        grad = np.concatenate(grads_W + grads_b)
        return loss, grad

    # Run L-BFGS-B
    result = minimize(
        _loss_grad,
        params0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
    )

    W_final, b_final = _unflatten(result.x)

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xn_local = (X - x_mean) / x_std
        out = _forward(Xn_local, W_final, b_final)
        return out[:, 0] * y_std + y_mean

    return predict
# EVOLVE-BLOCK-END

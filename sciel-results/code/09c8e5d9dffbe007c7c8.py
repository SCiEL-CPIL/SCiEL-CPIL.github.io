"""Symbolic regression over hidden equations. Small MLP trained with Adam."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _relu(z):
    return np.maximum(0.0, z)


def _relu_grad(z):
    return (z > 0).astype(float)


def _tanh(z):
    return np.tanh(z)


def _tanh_grad(z):
    t = np.tanh(z)
    return 1.0 - t * t


class _SmallMLP:
    """2 hidden layers, 32 units each, tanh activation, scalar output."""

    def __init__(self, d_in, seed=42):
        rng = np.random.default_rng(seed)
        self.d_in = d_in
        self.h = 32
        # He-style init
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), size=(d_in, self.h))
        self.b1 = np.zeros(self.h)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.h), size=(self.h, self.h))
        self.b2 = np.zeros(self.h)
        self.W3 = rng.normal(0, np.sqrt(1.0 / self.h), size=(self.h, 1))
        self.b3 = np.zeros(1)

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = _tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _tanh(z2)
        out = a2 @ self.W3 + self.b3
        return z1, a1, z2, a2, out

    def predict(self, X):
        _, _, _, _, out = self._forward(X)
        return out.ravel()

    def train_step(self, X, y, lr):
        n = X.shape[0]
        z1, a1, z2, a2, out = self._forward(X)
        # dL/dout (MSE)
        d_out = (2.0 / n) * (out - y.reshape(-1, 1))
        # backprop
        d_W3 = a2.T @ d_out
        d_b3 = d_out.sum(axis=0)
        d_a2 = d_out @ self.W3.T
        d_z2 = d_a2 * _tanh_grad(z2)
        d_W2 = a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * _tanh_grad(z1)
        d_W1 = X.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        # Adam updates
        self._adam_update(self.W1, d_W1, lr)
        self._adam_update(self.b1, d_b1, lr)
        self._adam_update(self.W2, d_W2, lr)
        self._adam_update(self.b2, d_b2, lr)
        self._adam_update(self.W3, d_W3, lr)
        self._adam_update(self.b3, d_b3, lr)

    def _adam_update(self, param, grad, lr):
        if not hasattr(self, '_adam_state'):
            self._adam_state = {}
        key = id(param)
        if key not in self._adam_state:
            self._adam_state[key] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param),
                't': 0,
            }
        state = self._adam_state[key]
        state['t'] += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        state['m'] = beta1 * state['m'] + (1 - beta1) * grad
        state['v'] = beta2 * state['v'] + (1 - beta2) * (grad * grad)
        m_hat = state['m'] / (1 - beta1 ** state['t'])
        v_hat = state['v'] / (1 - beta2 ** state['t'])
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    n, d = X_train.shape

    # Normalize inputs to [0, 1] range for better NN training
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = np.where(X_max - X_min > 1e-12, X_max - X_min, 1.0)
    X_norm = (X_train - X_min) / X_range

    # Center and scale y
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std

    # Time budget per equation
    time_per_eq = max(1.0, _time_left() / 10.0)
    max_steps = 3000
    lr = 0.01

    best_model = None
    best_loss = float('inf')

    for seed in [42, 123, 7]:
        if _time_left() < 2.0:
            break
        t_start = time.time()
        mlp = _SmallMLP(d, seed=seed)

        # Adaptive learning rate: start high, decay
        current_lr = lr
        patience = 0
        best_epoch_loss = float('inf')

        for step in range(max_steps):
            if _time_left() < 2.0:
                break
            if time.time() - t_start > time_per_eq:
                break

            mlp.train_step(X_norm, y_norm.reshape(-1, 1), current_lr)

            # Check loss every 100 steps
            if step % 100 == 0:
                pred = mlp.predict(X_norm)
                loss = float(np.mean((pred - y_norm) ** 2))
                if loss < best_epoch_loss:
                    best_epoch_loss = loss
                    patience = 0
                    # Save best weights
                    best_model = (mlp.W1.copy(), mlp.b1.copy(), mlp.W2.copy(),
                                   mlp.b2.copy(), mlp.W3.copy(), mlp.b3.copy())
                else:
                    patience += 1
                    if patience >= 5:
                        current_lr *= 0.5
                        patience = 0
                        if current_lr < 1e-5:
                            break

        # Final evaluation
        pred = mlp.predict(X_norm)
        loss = float(np.mean((pred - y_norm) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_model = (mlp.W1.copy(), mlp.b1.copy(), mlp.W2.copy(),
                          mlp.b2.copy(), mlp.W3.copy(), mlp.b3.copy())

    if best_model is None:
        # Fallback: constant prediction
        def predict(X):
            X = np.asarray(X, dtype=float)
            return np.full(X.shape[0], y_mean)
        return predict

    W1, b1, W2, b2, W3, b3 = best_model

    def predict(X):
        X = np.asarray(X, dtype=float)
        Xn = (X - X_min) / X_range
        z1 = Xn @ W1 + b1
        a1 = _tanh(z1)
        z2 = a1 @ W2 + b2
        a2 = _tanh(z2)
        out = a2 @ W3 + b3
        return out.ravel() * y_std + y_mean

    return predict
# EVOLVE-BLOCK-END

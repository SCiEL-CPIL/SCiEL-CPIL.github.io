"""Symbolic regression via deep feedforward neural network (Adam) + KRR fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_total_start = None
_time_limit = 285.0  # safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _init_weights(fan_in, fan_out, rng):
    # He-like init scaled for tanh
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return (rng.standard_normal((fan_in, fan_out)) * scale)

def _tanh(z):
    return np.tanh(z)

def _tanh_grad(a):
    # a = tanh(z)
    return 1.0 - a * a

def _train_mlp(X, y, hidden_dims, lr=0.01, max_iter=6000, batch_size=32,
               weight_decay=1e-5, seed=0, time_budget=None):
    """Train an MLP with Adam. Returns (weights, biases) list of layers."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    layer_dims = [d] + list(hidden_dims) + [1]
    L = len(layer_dims) - 1

    W = [_init_weights(layer_dims[i], layer_dims[i+1], rng) for i in range(L)]
    b = [np.zeros((1, layer_dims[i+1])) for i in range(L)]

    # Adam state
    mW = [np.zeros_like(W[i]) for i in range(L)]
    vW = [np.zeros_like(W[i]) for i in range(L)]
    mb = [np.zeros_like(b[i]) for i in range(L)]
    vb = [np.zeros_like(b[i]) for i in range(L)]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    best_loss = np.inf
    best_W = [w.copy() for w in W]
    best_b = [bi.copy() for bi in b]

    start = time.time()
    for it in range(max_iter):
        if time_budget is not None and (time.time() - start) > time_budget:
            break
        t += 1
        bc1, bc2 = 1 - beta1**t, 1 - beta2**t

        # Mini-batch
        idx = rng.integers(0, n, size=batch_size)
        Xb = X[idx]
        yb = y[idx]

        # Forward
        acts = [Xb]
        zs = []
        for i in range(L):
            z = acts[i] @ W[i] + b[i]
            if i < L - 1:
                a = _tanh(z)
            else:
                a = z  # linear output
            acts.append(a)
            zs.append(z)

        # Loss (MSE)
        pred = acts[-1].ravel()
        loss = np.mean((pred - yb) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_W = [w.copy() for w in W]
            best_b = [bi.copy() for bi in b]

        # Backward
        # output gradient
        dL = (2.0 / len(yb)) * (pred - yb)  # (batch,)
        dL = dL[:, None]  # (batch,1)
        dW = [None] * L
        db = [None] * L
        da = dL
        for i in range(L - 1, -1, -1):
            dW[i] = acts[i].T @ da
            db[i] = da.sum(axis=0, keepdims=True)
            if i > 0:
                dz = da @ W[i].T
                if i - 1 < L - 1:
                    dz = dz * _tanh_grad(acts[i])
                da = dz
            # add weight decay
            dW[i] = dW[i] + weight_decay * W[i]

        # Adam update
        for i in range(L):
            mW[i] = beta1 * mW[i] + (1 - beta1) * dW[i]
            vW[i] = beta2 * vW[i] + (1 - beta2) * (dW[i] ** 2)
            mW_hat = mW[i] / bc1
            vW_hat = vW[i] / bc2
            W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

            mb[i] = beta1 * mb[i] + (1 - beta1) * db[i]
            vb[i] = beta2 * vb[i] + (1 - beta2) * (db[i] ** 2)
            mb_hat = mb[i] / bc1
            vb_hat = vb[i] / bc2
            b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    return best_W, best_b, best_loss

def _mlp_predict(X, W, b):
    a = X
    L = len(W)
    for i in range(L):
        z = a @ W[i] + b[i]
        if i < L - 1:
            a = _tanh(z)
        else:
            a = z
    return a.ravel()

def _krr_fit_predict(X_train, y_train, eps, alpha):
    n = X_train.shape[0]
    sq = np.sum(X_train ** 2, axis=1)
    sqd = sq[:, None] + sq[None, :] - 2 * X_train @ X_train.T
    np.maximum(sqd, 0, out=sqd)
    K = np.exp(-eps * sqd)
    try:
        w = np.linalg.solve(K + alpha * np.eye(n), y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K + alpha * np.eye(n), y_train, rcond=None)[0]
    def predict(X):
        sqt = np.sum(X ** 2, axis=1)
        sqd = sqt[:, None] + sq[None, :] - 2 * X @ X_train.T
        np.maximum(sqd, 0, out=sqd)
        Kt = np.exp(-eps * sqd)
        return Kt @ w
    return predict, w

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()

    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n, d = X.shape

    # Normalize inputs to [0,1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    Xn = (X - X_min) / X_range

    # Normalize target
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    yn = (y - y_mean) / y_std

    best_loss = np.inf
    best_W = None
    best_b = None
    best_arch = None

    # Candidate architectures (hidden dims)
    archs = [
        [64, 64],
        [128, 64],
        [64, 32, 32],
        [128, 128, 64],
        [96, 96, 64, 32],
    ]

    # Time budget per (arch, restart)
    restarts = 2
    per_budget = max(1.0, min(8.0, _remaining() / (len(archs) * restarts)))

    seed = 12345
    for arch in archs:
        if _remaining() < 2.0:
            break
        for r in range(restarts):
            if _remaining() < 1.0:
                break
            try:
                W, b, loss = _train_mlp(
                    Xn, yn, arch,
                    lr=0.01,
                    max_iter=4000,
                    batch_size=32,
                    weight_decay=1e-5,
                    seed=seed + r,
                    time_budget=per_budget
                )
                if loss < best_loss:
                    best_loss = loss
                    best_W = W
                    best_b = b
                    best_arch = arch
            except Exception:
                pass

    # KRR fallback / candidate
    krr_predict = None
    krr_score = np.inf
    eps_vals = [0.5, 1.0, 2.0, 5.0]
    alpha_vals = [1e-4, 1e-3, 1e-2]
    for eps in eps_vals:
        if _remaining() < 1.0:
            break
        for alpha in alpha_vals:
            if _remaining() < 0.5:
                break
            try:
                pred_fn, _ = _krr_fit_predict(Xn, yn, eps, alpha)
                train_pred = pred_fn(Xn)
                sc = np.mean((train_pred - yn) ** 2)
                if sc < krr_score:
                    krr_score = sc
                    krr_predict = pred_fn
            except Exception:
                pass

    # Decide: use MLP if it clearly beats KRR on train (proxy), else KRR
    use_mlp = (best_W is not None) and (best_loss <= krr_score + 0.05)

    if use_mlp:
        W = best_W
        b = best_b
        def predict(X_test):
            Xt = X_test.astype(np.float64)
            Xtn = (Xt - X_min) / X_range
            p = _mlp_predict(Xtn, W, b)
            p = p * y_std + y_mean
            p = np.where(np.isfinite(p), p, y_mean)
            return p
    else:
        if krr_predict is None:
            # last resort: mean predictor
            def predict(X_test):
                return np.full(X_test.shape[0], y_mean)
        else:
            pred_fn = krr_predict
            def predict(X_test):
                Xt = X_test.astype(np.float64)
                Xtn = (Xt - X_min) / X_range
                p = pred_fn(Xtn)
                p = p * y_std + y_mean
                p = np.where(np.isfinite(p), p, y_mean)
                return p

    def safe_predict(X_test):
        p = predict(X_test)
        p = np.asarray(p, dtype=np.float64).ravel()
        p = np.where(np.isfinite(p), p, y_mean)
        return p

    return safe_predict
# EVOLVE-BLOCK-END

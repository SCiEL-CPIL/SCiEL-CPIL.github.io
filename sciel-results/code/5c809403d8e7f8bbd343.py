import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import CubicSpline, PchipInterpolator
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0
    
    X = np.asarray(X_train).flatten().astype(float)
    y = np.asarray(y_train).flatten().astype(float)
    n = len(X)
    
    if np.var(y) < 1e-12:
        mean_y = np.mean(y)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const

    best_model = None
    best_mse = np.inf
    
    def check_and_update(mse, predict_func):
        nonlocal best_model, best_mse
        if mse < best_mse:
            best_mse = mse
            best_model = predict_func
        return mse < 1e-10  # True if essentially exact

    # 1. Polynomials (degrees 1-20) - fast, catches x^k, ax+b, etc.
    for deg in range(1, 21):
        if time.time() > deadline - 5:
            break
        try:
            if deg > n - 1:
                break
            coef = np.polyfit(X, y, deg)
            y_pred = np.polyval(coef, X)
            mse = np.mean((y_pred - y)**2)
            if check_and_update(mse, lambda X, c=coef: np.polyval(c, np.asarray(X).flatten()).reshape(np.asarray(X).shape)):
                return best_model
        except Exception:
            continue

    # 2. Rational functions P(x)/Q(x) with Q(0)=1
    def rational_func(x, *params):
        np_deg = int(len(params) / 2)
        nq_deg = int(len(params) / 2)
        a = params[:np_deg]
        b = params[np_deg:]
        num = np.zeros_like(x, dtype=float)
        for i, ai in enumerate(a):
            num = num + ai * (x ** i)
        den = np.ones_like(x, dtype=float)
        for i, bi in enumerate(b):
            den = den + bi * (x ** (i + 1))
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return num / den

    for np_deg in [1, 2, 3, 4, 5]:
        for nq_deg in [1, 2, 3, 4]:
            if time.time() > deadline - 10:
                break
            try:
                p0 = np.zeros(np_deg + nq_deg)
                p0[:np_deg] = np.linspace(0.5, 1.0, np_deg)
                p0[np_deg:] = np.linspace(0.1, 0.3, nq_deg)
                popt, _ = curve_fit(rational_func, X, y, p0=p0, maxfev=2000)
                y_pred = rational_func(X, *popt)
                if np.all(np.isfinite(y_pred)):
                    mse = np.mean((y_pred - y)**2)
                    if check_and_update(mse, lambda X, p=popt: rational_func(np.asarray(X).flatten(), *p).reshape(np.asarray(X).shape)):
                        return best_model
            except Exception:
                continue

    # 3. Exponential: a*exp(b*x) + c
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    try:
        if time.time() < deadline - 5:
            p0 = [1.0, 0.1, 0.0]
            popt, _ = curve_fit(exp_func, X, y, p0=p0, maxfev=2000)
            y_pred = exp_func(X, *popt)
            if np.all(np.isfinite(y_pred)):
                mse = np.mean((y_pred - y)**2)
                if check_and_update(mse, lambda X, p=popt: exp_func(np.asarray(X).flatten(), *p).reshape(np.asarray(X).shape)):
                    return best_model
    except Exception:
        pass

    # 4. Logarithmic: a*log(b*x) + c  (for x>0 ranges)
    def log_func(x, a, b, c):
        return a * np.log(np.maximum(b * np.abs(x), 1e-12)) + c
    try:
        if time.time() < deadline - 5 and np.min(np.abs(X)) > 1e-6:
            p0 = [1.0, 1.0, 0.0]
            popt, _ = curve_fit(log_func, X, y, p0=p0, maxfev=2000)
            y_pred = log_func(X, *popt)
            if np.all(np.isfinite(y_pred)):
                mse = np.mean((y_pred - y)**2)
                if check_and_update(mse, lambda X, p=popt: log_func(np.asarray(X).flatten(), *p).reshape(np.asarray(X).shape)):
                    return best_model
    except Exception:
        pass

    # 5. Sine/Cosine: a*sin(b*x+c) + d
    def sin_func(x, a, b, c, d):
        return a * np.sin(b * x + c) + d
    try:
        if time.time() < deadline - 5:
            p0 = [1.0, 1.0, 0.0, 0.0]
            popt, _ = curve_fit(sin_func, X, y, p0=p0, maxfev=2000)
            y_pred = sin_func(X, *popt)
            if np.all(np.isfinite(y_pred)):
                mse = np.mean((y_pred - y)**2)
                if check_and_update(mse, lambda X, p=popt: sin_func(np.asarray(X).flatten(), *p).reshape(np.asarray(X).shape)):
                    return best_model
    except Exception:
        pass

    # 6. Compact neural network as universal approximator
    def nn_forward(x, params):
        x = np.asarray(x).flatten().reshape(-1, 1).astype(float)
        h1 = 16
        w1 = params[:h1].reshape(1, h1)
        b1 = params[h1:2*h1]
        w2 = params[2*h1:2*h1+h1].reshape(h1, 1)
        b2 = params[2*h1+h1:2*h1+h1+1]
        z1 = x @ w1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ w2 + b2
        return z2.flatten()

    try:
        if time.time() < deadline - 20:
            h1 = 16
            n_params = 1*h1 + h1 + h1*1 + 1
            np.random.seed(42)
            p0 = np.random.randn(n_params) * 0.1
            def loss(params):
                try:
                    pred = nn_forward(X, params)
                    if not np.all(np.isfinite(pred)):
                        return 1e10 * np.ones(n)
                    return pred - y
                except:
                    return 1e10 * np.ones(n)
            res = least_squares(loss, p0, max_nfev=500)
            y_pred = nn_forward(X, res.x)
            if np.all(np.isfinite(y_pred)):
                mse = np.mean((y_pred - y)**2)
                if check_and_update(mse, lambda X, p=res.x: nn_forward(np.asarray(X).flatten(), p).reshape(np.asarray(X).shape)):
                    return best_model
    except Exception:
        pass

    # 7. PCHIP interpolation - safe fallback for noiseless data
    try:
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        unique_X, inverse_indices = np.unique(X_sorted, return_inverse=True)
        if len(unique_X) < n:
            unique_y = np.zeros(len(unique_X))
            counts = np.zeros(len(unique_X))
            np.add.at(unique_y, inverse_indices, y_sorted)
            np.add.at(counts, inverse_indices, 1)
            unique_y /= counts
            X_sorted = unique_X
            y_sorted = unique_y
        pchip = PchipInterpolator(X_sorted, y_sorted)
        y_pred = pchip(X)
        mse = np.mean((y_pred - y)**2)
        if mse < best_mse:
            best_mse = mse
            best_model = lambda X: pchip(np.asarray(X).flatten()).reshape(np.asarray(X).shape)
    except Exception:
        pass

    # 8. Cubic spline as another fallback
    try:
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        unique_X, inverse_indices = np.unique(X_sorted, return_inverse=True)
        if len(unique_X) < n:
            unique_y = np.zeros(len(unique_X))
            counts = np.zeros(len(unique_X))
            np.add.at(unique_y, inverse_indices, y_sorted)
            np.add.at(counts, inverse_indices, 1)
            unique_y /= counts
            X_sorted = unique_X
            y_sorted = unique_y
        cs = CubicSpline(X_sorted, y_sorted, bc_type='natural')
        y_pred = cs(X)
        mse = np.mean((y_pred - y)**2)
        if mse < best_mse:
            best_mse = mse
            best_model = lambda X: cs(np.asarray(X).flatten()).reshape(np.asarray(X).shape)
    except Exception:
        pass

    if best_model is None:
        mean_y = np.mean(y)
        best_model = lambda X: np.full(np.asarray(X).shape, mean_y)

    return best_model

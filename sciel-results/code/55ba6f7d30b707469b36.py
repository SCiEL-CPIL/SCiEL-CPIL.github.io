# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator, CubicSpline, make_interp_spline
from scipy.linalg import solve
from scipy.optimize import minimize_scalar, differential_evolution
from scipy.signal import savgol_filter

_TOTAL_DEADLINE = time.time() + 310.0


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
    """Wrap a prediction function to always return finite values."""
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict


def _prepare_1d(X, y):
    """Sort 1D data and handle duplicates."""
    x = X[:, 0]
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    unique_xs, unique_indices, counts = np.unique(xs, return_inverse=True, return_counts=True)
    if len(unique_xs) < len(xs):
        ys_sum = np.zeros_like(unique_xs)
        np.add.at(ys_sum, unique_indices, ys)
        ys = ys_sum / counts
    return unique_xs, ys


def _fit_pchip(X, y):
    try:
        xs, ys = _prepare_1d(X, y)
        if len(xs) < 4:
            return None
        s = PchipInterpolator(xs, ys)
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            return np.nan_to_num(s(Xp[:, 0]), nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_cubic_spline(X, y):
    try:
        xs, ys = _prepare_1d(X, y)
        if len(xs) < 4:
            return None
        s = CubicSpline(xs, ys)
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            return np.nan_to_num(s(Xp[:, 0]), nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_interp_spline(X, y, k=3):
    try:
        xs, ys = _prepare_1d(X, y)
        if len(xs) < k + 1:
            return None
        s = make_interp_spline(xs, ys, k=k)
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            return np.nan_to_num(s(Xp[:, 0]), nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_savgol(X, y, window=15, polyorder=3):
    """Fit a Savitzky-Golay smoothed interpolant."""
    try:
        xs, ys = _prepare_1d(X, y)
        if len(xs) < window:
            window = max(3, len(xs) // 2 * 2 + 1)
        if window < 5 or polyorder >= window:
            return None
        # Evaluate on a fine grid
        x_min, x_max = xs[0], xs[-1]
        x_fine = np.linspace(x_min, x_max, max(200, len(xs) * 5))
        y_fine = np.interp(x_fine, xs, ys)
        y_smooth = savgol_filter(y_fine, window, polyorder)
        
        # Interpolate the smoothed data
        s = PchipInterpolator(x_fine, y_smooth)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            return np.nan_to_num(s(Xp[:, 0]), nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_rational(X, y, num_deg=3, den_deg=2):
    """Fit a rational function P(x)/Q(x) using nonlinear optimization."""
    try:
        xs, ys = _prepare_1d(X, y)
        n = len(xs)
        if n < num_deg + den_deg + 2:
            return None
        
        # Normalize x to [-1, 1] for numerical stability
        x_min, x_max = xs[0], xs[-1]
        if x_max - x_min < 1e-10:
            return None
        x_norm = 2 * (xs - x_min) / (x_max - x_min) - 1
        
        # Initial guess: all coefficients 1
        init_params = np.ones(num_deg + den_deg + 1)
        
        def objective(params):
            a = params[:num_deg + 1]  # Numerator coefficients (highest degree first)
            b = params[num_deg + 1:]  # Denominator coefficients
            # Ensure denominator doesn't vanish
            denom = np.polyval(b, x_norm)
            if np.any(np.abs(denom) < 1e-10):
                return 1e15
            numer = np.polyval(a, x_norm)
            pred = numer / denom
            return np.sum((pred - ys) ** 2)
        
        # Use differential evolution for global optimization
        bounds = [(-10, 10)] * (num_deg + den_deg + 1)
        result = differential_evolution(
            objective, bounds, seed=42, maxiter=50, tol=1e-8,
            popsize=10, polish=False, init='sobol'
        )
        
        if result.fun > 1e10:
            return None
            
        best_params = result.x
        a = best_params[:num_deg + 1]
        b = best_params[num_deg + 1:]
        
        # Check if denominator is valid
        denom = np.polyval(b, x_norm)
        if np.any(np.abs(denom) < 1e-8):
            return None
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xpn = 2 * (Xp[:, 0] - x_min) / (x_max - x_min) - 1
            denom = np.polyval(b, xpn)
            numer = np.polyval(a, xpn)
            # Avoid division by zero
            denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
            return np.nan_to_num(numer / denom_safe, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_local_polynomial(X, y, order=3, bandwidth=None):
    """Fit a local polynomial regression (LOESS-like)."""
    try:
        xs, ys = _prepare_1d(X, y)
        n = len(xs)
        if n < order + 2:
            return None
        
        if bandwidth is None:
            bandwidth = 0.5  # Use 50% of data as local window
            
        x_min, x_max = xs[0], xs[-1]
        if x_max - x_min < 1e-10:
            return None
            
        # Precompute local weights and polynomial fits
        # For efficiency, we'll use a coarse grid approach
        n_grid = 200
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = np.zeros(n_grid)
        
        for i, x0 in enumerate(x_grid):
            # Compute distances
            dist = np.abs(xs - x0)
            # Local weights (tricube kernel)
            u = dist / (bandwidth * (x_max - x_min))
            weights = np.where(u < 1, (1 - u**3)**3, 0)
            
            if weights.sum() < 1e-10:
                y_grid[i] = ys[np.argmin(dist)]
                continue
                
            # Weighted polynomial fit
            x_local = xs - x0
            try:
                # Build Vandermonde-like matrix
                V = np.vander(x_local, N=order+1, increasing=True)
                W = np.diag(weights)
                # Weighted least squares
                A = V.T @ W @ V
                b = V.T @ W @ ys
                # Add small regularization for stability
                A += 1e-10 * np.eye(order+1)
                coeffs = np.linalg.solve(A, b)
                y_grid[i] = coeffs[0]  # Value at x0 is the constant term
            except Exception:
                y_grid[i] = np.average(ys, weights=weights)
        
        # Interpolate the grid
        s = PchipInterpolator(x_grid, y_grid)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            return np.nan_to_num(s(Xp[:, 0]), nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_gaussian_process(X, y, length_scale=1.0, noise=1e-6):
    """Gaussian Process regression with RBF kernel."""
    try:
        xs, ys = _prepare_1d(X, y)
        n = len(xs)
        
        # Normalize
        x_min, x_max = xs[0], xs[-1]
        if x_max - x_min < 1e-10:
            return None
        x_norm = (xs - x_min) / (x_max - x_min)
        
        # RBF kernel
        diff = x_norm[:, np.newaxis] - x_norm[np.newaxis, :]
        K = np.exp(-0.5 * (diff / length_scale) ** 2) + noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))
        except Exception:
            K_reg = K + 1e-8 * np.eye(n)
            try:
                L = np.linalg.cholesky(K_reg)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))
            except Exception:
                return None
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xpn = (Xp[:, 0] - x_min) / (x_max - x_min)
            diff = xpn[:, np.newaxis] - x_norm[np.newaxis, :]
            Ks = np.exp(-0.5 * (diff / length_scale) ** 2)
            preds = Ks @ alpha
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_fourier(X, y, n_harmonics=20):
    """Fit a Fourier series."""
    try:
        xs, ys = _prepare_1d(X, y)
        n = len(xs)
        if n < n_harmonics * 2 + 2:
            n_harmonics = max(2, n // 3)
        
        x_min, x_max = xs[0], xs[-1]
        if x_max - x_min < 1e-10:
            return None
        
        # Build Fourier basis
        cols = [np.ones(n)]
        for k in range(1, n_harmonics + 1):
            cols.append(np.sin(2 * np.pi * k * (xs - x_min) / (x_max - x_min)))
            cols.append(np.cos(2 * np.pi * k * (xs - x_min) / (x_max - x_min)))
        
        F = np.column_stack(cols)
        w, *_ = np.linalg.lstsq(F, ys, rcond=None)
        
        if not np.all(np.isfinite(w)):
            return None
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            cols = [np.ones(Xp.shape[0])]
            for k in range(1, n_harmonics + 1):
                cols.append(np.sin(2 * np.pi * k * (Xp[:, 0] - x_min) / (x_max - x_min)))
                cols.append(np.cos(2 * np.pi * k * (Xp[:, 0] - x_min) / (x_max - x_min)))
            Fp = np.column_stack(cols)
            return np.nan_to_num(Fp @ w, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_mlp(X, y, n_hidden=10, lr=0.01, n_iter=500):
    """Simple MLP with manual backprop."""
    try:
        xs, ys = _prepare_1d(X, y)
        n = len(xs)
        if n < 10:
            return None
        
        x_min, x_max = xs[0], xs[-1]
        if x_max - x_min < 1e-10:
            return None
        x_norm = (xs - x_min) / (x_max - x_min)
        
        # Initialize weights
        np.random.seed(42)
        W1 = np.random.randn(1, n_hidden) * 0.1
        b1 = np.zeros(n_hidden)
        W2 = np.random.randn(n_hidden, 1) * 0.1
        b2 = np.zeros(1)
        
        # Train
        for i in range(n_iter):
            if _remaining() < 10:
                break
            # Forward
            z1 = x_norm @ W1.T + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2.T + b2
            pred = z2.flatten()
            
            # Backward
            dL_dy = (pred - ys) / n
            dz2 = dL_dy
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2)
            da1 = dz2 @ W2
            dz1 = da1 * (1 - a1**2)
            dW1 = x_norm.T @ dz1
            db1 = np.sum(dz1)
            
            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xpn = (Xp[:, 0] - x_min) / (x_max - x_min)
            z1 = xpn @ W1.T + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2.T + b2
            return np.nan_to_num(z2.flatten(), nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_train - const) ** 2))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    if d != 1:
        return best_predict

    candidates = []

    # 1. Interpolation methods
    if _remaining() > 5.0:
        p = _fit_pchip(X_train, y_train)
        if p is not None:
            candidates.append(('pchip', p))

    if _remaining() > 5.0:
        p = _fit_cubic_spline(X_train, y_train)
        if p is not None:
            candidates.append(('cubic_spline', p))

    if _remaining() > 5.0:
        for k in [3, 4, 5]:
            if _remaining() < 3.0:
                break
            p = _fit_interp_spline(X_train, y_train, k=k)
            if p is not None:
                candidates.append((f'interp_k{k}', p))

    # 2. Savitzky-Golay smoothed
    if _remaining() > 5.0:
        for window in [11, 15, 21, 31]:
            if _remaining() < 3.0:
                break
            for polyorder in [2, 3, 4]:
                if _remaining() < 2.0:
                    break
                p = _fit_savgol(X_train, y_train, window=window, polyorder=polyorder)
                if p is not None:
                    candidates.append((f'savgol_w{window}_p{polyorder}', p))

    # 3. Rational functions (key structural addition)
    if _remaining() > 15.0:
        for num_deg in [2, 3, 4]:
            for den_deg in [1, 2, 3]:
                if _remaining() < 10.0:
                    break
                if num_deg + den_deg + 1 > 10:
                    continue
                p = _fit_rational(X_train, y_train, num_deg=num_deg, den_deg=den_deg)
                if p is not None:
                    candidates.append((f'rational_{num_deg}_{den_deg}', p))

    # 4. Local polynomial
    if _remaining() > 10.0:
        for order in [2, 3, 4]:
            for bandwidth in [0.3, 0.5, 0.7]:
                if _remaining() < 5.0:
                    break
                p = _fit_local_polynomial(X_train, y_train, order=order, bandwidth=bandwidth)
                if p is not None:
                    candidates.append((f'local_poly_o{order}_bw{bandwidth}', p))

    # 5. Gaussian Process
    if _remaining() > 10.0:
        for ls in [0.1, 0.3, 0.5, 1.0, 2.0]:
            if _remaining() < 5.0:
                break
            p = _fit_gaussian_process(X_train, y_train, length_scale=ls)
            if p is not None:
                candidates.append((f'gp_ls{ls}', p))

    # 6. Fourier
    if _remaining() > 5.0:
        for n_h in [10, 20, 30]:
            if _remaining() < 3.0:
                break
            p = _fit_fourier(X_train, y_train, n_harmonics=n_h)
            if p is not None:
                candidates.append((f'fourier_{n_h}', p))

    # 7. MLP
    if _remaining() > 15.0:
        for n_hidden in [5, 10, 20]:
            if _remaining() < 10.0:
                break
            p = _fit_mlp(X_train, y_train, n_hidden=n_hidden, lr=0.01, n_iter=300)
            if p is not None:
                candidates.append((f'mlp_{n_hidden}', p))

    # Evaluate all candidates
    for name, predict_fn in candidates:
        if _remaining() < 2.0:
            break
        try:
            pred = predict_fn(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = predict_fn
        except Exception:
            continue

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END

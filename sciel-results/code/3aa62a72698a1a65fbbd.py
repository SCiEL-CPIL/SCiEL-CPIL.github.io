# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    x = X[:, 0]
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_min = x[0] - 0.5 if n > 0 else 0.0
        x_max = x[0] + 0.5 if n > 0 else 1.0
    
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std < 1e-12:
        y_std = 1.0
    
    candidates = []
    
    def elapsed():
        return time.time() - start_time
    
    def add_candidate(name, pred_func):
        if elapsed() > deadline:
            return
        try:
            yp = np.asarray(pred_func(X), dtype=float).flatten()
            if yp.shape[0] != n or not np.all(np.isfinite(yp)):
                return
            mse = float(np.mean((yp - y) ** 2))
            if np.isfinite(mse):
                candidates.append((name, pred_func, mse))
        except Exception:
            pass
    
    # --- Strategy 1: Polynomial regression ---
    x_scaled = (x - x_min) / x_range
    for deg in range(1, 20):
        if elapsed() > deadline:
            break
        V = np.vander(x_scaled, N=deg + 1, increasing=True)
        try:
            w, *_ = np.linalg.lstsq(V, y, rcond=None)
            def pred_poly(Xp, deg=deg, w=w, x_min=x_min, x_range=x_range):
                Xp = np.asarray(Xp, dtype=float).reshape(-1)
                Xs = (Xp - x_min) / x_range
                Vp = np.vander(Xs, N=deg + 1, increasing=True)
                return Vp @ w
            add_candidate(f"poly_{deg}", pred_poly)
        except Exception:
            pass
    
    # --- Strategy 2: Fourier basis ---
    omegas = [np.pi / x_range, 2 * np.pi / x_range, np.pi, 2 * np.pi, 1.0, 0.5, 0.25, 4.0, 0.1, 5.0]
    for nf in range(1, 12):
        if elapsed() > deadline:
            break
        for omega in omegas:
            if elapsed() > deadline:
                break
            cols = [np.ones(n)]
            for k in range(1, nf + 1):
                cols.append(np.sin(k * omega * x))
                cols.append(np.cos(k * omega * x))
            V = np.column_stack(cols)
            try:
                w, *_ = np.linalg.lstsq(V, y, rcond=None)
                def pred_fourier(Xp, nf=nf, omega=omega, w=w):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    cols = [np.ones(len(Xp))]
                    for k in range(1, nf + 1):
                        cols.append(np.sin(k * omega * Xp))
                        cols.append(np.cos(k * omega * Xp))
                    Vp = np.column_stack(cols)
                    return Vp @ w
                add_candidate(f"four_nf{nf}_om{omega:.4f}", pred_fourier)
            except Exception:
                pass
    
    # --- Strategy 3: Rich basis with greedy selection ---
    basis_funcs = {}
    basis_funcs['1'] = lambda x: np.ones_like(x)
    basis_funcs['x'] = lambda x: x
    basis_funcs['x2'] = lambda x: x ** 2
    basis_funcs['x3'] = lambda x: x ** 3
    basis_funcs['x4'] = lambda x: x ** 4
    basis_funcs['x5'] = lambda x: x ** 5
    basis_funcs['x6'] = lambda x: x ** 6
    basis_funcs['abs_x'] = lambda x: np.abs(x)
    basis_funcs['sqrt_abs'] = lambda x: np.sqrt(np.abs(x))
    basis_funcs['log1p_abs'] = lambda x: np.log1p(np.abs(x))
    basis_funcs['tanh'] = lambda x: np.tanh(x)
    basis_funcs['tanh_2x'] = lambda x: np.tanh(2 * x)
    basis_funcs['tanh_5x'] = lambda x: np.tanh(5 * x)
    basis_funcs['exp_05x'] = lambda x: np.exp(0.5 * x)
    basis_funcs['exp_neg05x'] = lambda x: np.exp(-0.5 * x)
    basis_funcs['exp_x'] = lambda x: np.exp(x)
    basis_funcs['exp_negx'] = lambda x: np.exp(-x)
    basis_funcs['exp_01x'] = lambda x: np.exp(0.1 * x)
    basis_funcs['exp_neg01x'] = lambda x: np.exp(-0.1 * x)
    basis_funcs['x_over_1x2'] = lambda x: x / (1 + x ** 2)
    basis_funcs['x_over_1abx'] = lambda x: x / (1 + np.abs(x))
    basis_funcs['sinc'] = lambda x: np.sinc(x)
    basis_funcs['sin_x'] = lambda x: np.sin(x)
    basis_funcs['cos_x'] = lambda x: np.cos(x)
    basis_funcs['sin_2x'] = lambda x: np.sin(2 * x)
    basis_funcs['cos_2x'] = lambda x: np.cos(2 * x)
    basis_funcs['sin_05x'] = lambda x: np.sin(0.5 * x)
    basis_funcs['cos_05x'] = lambda x: np.cos(0.5 * x)
    basis_funcs['sin_pi_x'] = lambda x: np.sin(np.pi * x)
    basis_funcs['cos_pi_x'] = lambda x: np.cos(np.pi * x)
    basis_funcs['x_sin_x'] = lambda x: x * np.sin(x)
    basis_funcs['x_cos_x'] = lambda x: x * np.cos(x)
    basis_funcs['exp_negx2'] = lambda x: np.exp(-x ** 2)
    basis_funcs['exp_neg05x2'] = lambda x: np.exp(-0.5 * x ** 2)
    basis_funcs['exp_neg2x2'] = lambda x: np.exp(-2 * x ** 2)
    
    basis_names = list(basis_funcs.keys())
    B = np.zeros((n, len(basis_names)))
    for j, name in enumerate(basis_names):
        try:
            B[:, j] = basis_funcs[name](x)
            B[:, j] = np.where(np.isfinite(B[:, j]), B[:, j], 0.0)
        except Exception:
            pass
    
    if elapsed() < deadline:
        # Greedy forward selection
        selected = []
        remaining = list(range(len(basis_names)))
        for step in range(min(10, len(basis_names))):
            if elapsed() > deadline:
                break
            best_j = None
            best_rss = np.inf
            for j in remaining:
                try:
                    if selected:
                        V = np.column_stack([B[:, i] for i in selected] + [B[:, j]])
                    else:
                        V = B[:, j:j+1]
                    w, *_ = np.linalg.lstsq(V, y, rcond=None)
                    rss = float(np.sum((V @ w - y) ** 2))
                    if np.isfinite(rss) and rss < best_rss:
                        best_rss = rss
                        best_j = j
                except Exception:
                    pass
            if best_j is None:
                break
            selected.append(best_j)
            remaining.remove(best_j)
            
            sel_names = [basis_names[i] for i in selected]
            try:
                V = np.column_stack([B[:, i] for i in selected])
                w, *_ = np.linalg.lstsq(V, y, rcond=None)
                def pred_sel(Xp, sel_names=sel_names, w=w):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    cols = []
                    for nm in sel_names:
                        c = basis_funcs[nm](Xp)
                        c = np.where(np.isfinite(c), c, 0.0)
                        cols.append(c)
                    Vp = np.column_stack(cols)
                    return Vp @ w
                add_candidate(f"sel_{'_'.join(sel_names[:3])}", pred_sel)
            except Exception:
                pass
    
    # --- Strategy 4: Gaussian Process with Matern 5/2 ---
    if elapsed() < deadline:
        def matern52_kernel(X1, X2, ls):
            X1 = np.asarray(X1, dtype=float).reshape(-1, 1)
            X2 = np.asarray(X2, dtype=float).reshape(-1, 1)
            D = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))
            sqrt3d = np.sqrt(3.0) * D / ls
            return (1 + sqrt3d + sqrt3d ** 2 / 3.0) * np.exp(-sqrt3d)
        
        y_var = float(np.var(y))
        if y_var < 1e-12:
            y_var = 1.0
        
        for ls in [x_range / 3.0, x_range / 5.0, x_range / 8.0, x_range / 15.0, 0.2, 0.3, 0.5, 1.0, 2.0, 0.1]:
            if ls <= 0 or elapsed() > deadline:
                continue
            try:
                K_kernel = matern52_kernel(x, x, ls)
                K = y_var * K_kernel + 1e-8 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                
                def pred_gp(Xp, x=x, L=L, alpha=alpha, ls=ls, y_var=y_var, matern52_kernel=matern52_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Ks = matern52_kernel(Xp, x, ls)
                    return y_var * (Ks @ alpha)
                
                add_candidate(f"gp_ls{ls:.4f}", pred_gp)
            except Exception:
                pass
    
    # --- Strategy 5: RBF kernel regression ---
    if elapsed() < deadline:
        def rbf_kernel(X1, X2, width):
            X1 = np.asarray(X1, dtype=float).reshape(-1, 1)
            X2 = np.asarray(X2, dtype=float).reshape(-1, 1)
            D2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
            return np.exp(-D2 / (2 * width ** 2))
        
        for width in [x_range / 4.0, x_range / 6.0, x_range / 10.0, x_range / 20.0, 0.1, 0.2, 0.5, 1.0, 2.0]:
            if width <= 0 or elapsed() > deadline:
                continue
            try:
                K = rbf_kernel(x, x, width)
                K += 1e-8 * np.eye(n)
                w, *_ = np.linalg.lstsq(K, y, rcond=None)
                
                def pred_rbf(Xp, x=x, w=w, width=width, rbf_kernel=rbf_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Ks = rbf_kernel(Xp, x, width)
                    return Ks @ w
                
                add_candidate(f"rbf_w{width:.4f}", pred_rbf)
            except Exception:
                pass
    
    # --- Strategy 6: Neural network with random initialization and gradient descent ---
    if elapsed() < deadline:
        try:
            np.random.seed(42)
            x_norm = (x - x_min) / x_range
            
            for hidden in [16, 32, 64]:
                if elapsed() > deadline:
                    break
                W1 = np.random.randn(x_norm.shape[0], hidden) * 0.1
                b1 = np.zeros(hidden)
                W2 = np.random.randn(hidden, 1) * 0.1
                b2 = np.zeros(1)
                
                lr = 0.01
                for step in range(2000):
                    if elapsed() > deadline:
                        break
                    z1 = x_norm @ W1 + b1
                    a1 = np.tanh(z1)
                    z2 = a1 @ W2 + b2
                    pred = z2.flatten()
                    
                    err = pred - y
                    dz2 = err / n
                    dW2 = a1.T @ dz2
                    db2 = np.sum(dz2, axis=0)
                    da1 = dz2 @ W2.T
                    dz1 = da1 * (1 - a1 ** 2)
                    dW1 = x_norm.T @ dz1
                    db1 = np.sum(dz1, axis=0)
                    
                    W1 -= lr * dW1
                    b1 -= lr * db1
                    W2 -= lr * dW2
                    b2 -= lr * db2
                
                def pred_nn(Xp, x_min=x_min, x_range=x_range, W1=W1, b1=b1, W2=W2, b2=b2):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Xn = (Xp - x_min) / x_range
                    z1 = Xn @ W1 + b1
                    a1 = np.tanh(z1)
                    z2 = a1 @ W2 + b2
                    return z2.flatten()
                
                add_candidate(f"nn_h{hidden}", pred_nn)
        except Exception:
            pass
    
    # --- Strategy 7: Spline interpolation via piecewise polynomial ---
    if elapsed() < deadline:
        try:
            from scipy.interpolate import UnivariateSpline
            for s in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
                if elapsed() > deadline:
                    break
                try:
                    spl = UnivariateSpline(x, y, s=s)
                    def pred_spline(Xp, spl=spl):
                        Xp = np.asarray(Xp, dtype=float).reshape(-1)
                        return spl(Xp)
                    add_candidate(f"spline_s{s}", pred_spline)
                except Exception:
                    pass
        except Exception:
            pass
    
    # --- Strategy 8: Local polynomial regression ---
    if elapsed() < deadline:
        for bw in [x_range / 5.0, x_range / 10.0, x_range / 20.0, 0.5, 1.0]:
            if bw <= 0 or elapsed() > deadline:
                continue
            try:
                def pred_local(Xp, x=x, y=y, bw=bw, deg=2):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    out = np.zeros(len(Xp))
                    for i, xi in enumerate(Xp):
                        dist = np.abs(x - xi)
                        wts = np.exp(-dist ** 2 / (2 * bw ** 2))
                        wts = wts / (wts.sum() + 1e-12)
                        # Local polynomial fit
                        V = np.vander((x - xi) / bw, N=deg + 1, increasing=True)
                        W = np.diag(wts)
                        VtW = V.T @ W
                        VtWV = VtW @ V
                        VtWy = VtW @ y
                        try:
                            coeffs = np.linalg.solve(VtWV + 1e-10 * np.eye(deg + 1), VtWy)
                            out[i] = coeffs[0]  # value at xi (local)
                        except Exception:
                            out[i] = np.sum(wts * y)
                    return out
                add_candidate(f"local_bw{bw:.4f}", pred_local)
            except Exception:
                pass
    
    # Select best candidate
    if not candidates:
        # Fallback: linear regression
        A = np.column_stack([np.ones(n), x])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        def pred_lin(Xp, w=w):
            Xp = np.asarray(Xp, dtype=float).reshape(-1)
            return np.column_stack([np.ones(len(Xp)), Xp]) @ w
        return pred_lin
    
    candidates.sort(key=lambda t: t[2])
    best_name, best_pred, best_mse = candidates[0]
    
    def final_predict(Xp):
        out = best_pred(Xp)
        out = np.asarray(out, dtype=float).flatten()
        out = np.where(np.isfinite(out), out, 0.0)
        return out
    
    return final_predict
# EVOLVE-BLOCK-END

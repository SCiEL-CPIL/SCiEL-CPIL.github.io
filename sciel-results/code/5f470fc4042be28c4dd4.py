"""Symbolic + non-parametric regression ensemble for 1-D noiseless equations."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_RNG = np.random.default_rng(12345)


def _as_1d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _safe_eval_tree(tree, x):
    """x: 1-D array of input values. Returns 1-D array (finite)."""
    op = tree[0]
    if op == 'x':
        return x
    if op == 'c':
        return np.full_like(x, tree[1], dtype=float)
    if op == 'neg':
        return -_safe_eval_tree(tree[1], x)
    if op == 'sin':
        return np.sin(_safe_eval_tree(tree[1], x))
    if op == 'cos':
        return np.cos(_safe_eval_tree(tree[1], x))
    if op == 'tan':
        v = np.tan(_safe_eval_tree(tree[1], x))
        return np.clip(v, -50.0, 50.0)
    if op == 'exp':
        v = np.exp(np.clip(_safe_eval_tree(tree[1], x), -30.0, 30.0))
        return np.clip(v, -1e6, 1e6)
    if op == 'tanh':
        return np.tanh(_safe_eval_tree(tree[1], x))
    if op == 'abs':
        return np.abs(_safe_eval_tree(tree[1], x))
    if op == 'sqrt':
        return np.sqrt(np.abs(_safe_eval_tree(tree[1], x)))
    if op == 'log':
        return np.log1p(np.abs(_safe_eval_tree(tree[1], x)))
    if op == 'sq':
        return _safe_eval_tree(tree[1], x) ** 2
    if op == 'cube':
        return _safe_eval_tree(tree[1], x) ** 3
    if op == 'add':
        return _safe_eval_tree(tree[1], x) + _safe_eval_tree(tree[2], x)
    if op == 'sub':
        return _safe_eval_tree(tree[1], x) - _safe_eval_tree(tree[2], x)
    if op == 'mul':
        return _safe_eval_tree(tree[1], x) * _safe_eval_tree(tree[2], x)
    if op == 'div':
        a = _safe_eval_tree(tree[1], x)
        b = _safe_eval_tree(tree[2], x)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = a / np.where(np.abs(b) < 1e-8, 1e-8, b)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    raise ValueError('unknown op')


def _tree_depth(t):
    if len(t) == 1:
        return 0
    if len(t) == 2:
        return 1 + _tree_depth(t[1])
    return 1 + max(_tree_depth(t[1]), _tree_depth(t[2]))


def _tree_size(t):
    if len(t) == 1:
        return 1
    if len(t) == 2:
        return 1 + _tree_size(t[1])
    return 1 + _tree_size(t[1]) + _tree_size(t[2])


def _rand_tree(max_depth, rng):
    def build(d):
        if d == 0 or (d < max_depth and rng.random() < 0.15):
            return _rand_terminal(rng)
        op = rng.choice(_FUNCS)
        if op in ('neg', 'sin', 'cos', 'tan', 'exp', 'tanh', 'abs', 'sqrt', 'log', 'sq', 'cube'):
            return [op, build(d - 1)]
        else:
            return [op, build(d - 1), build(d - 1)]
    return build(max_depth)


def _rand_terminal(rng):
    if rng.random() < 0.8:
        return ['x']
    return ['c', float(rng.uniform(-2.0, 2.0))]


_FUNCS = ['add', 'sub', 'mul', 'div', 'neg', 'sin', 'cos', 'tan', 'exp', 'tanh', 'abs', 'sqrt', 'log', 'sq', 'cube']


def _crossover(a, b, rng):
    a = [a]  # placeholder to allow mutation in place? we work on copies
    def copy(t):
        return [t[0]] + [copy(c) for c in t[1:]]
    A = copy(a)
    B = copy(b)
    # pick random node in A
    def nodes(t):
        yield t
        for c in t[1:]:
            yield from nodes(c)
    cand = list(nodes(A))
    pos = rng.choice(len(cand))
    node = cand[pos]
    # pick random subtree in B
    candb = list(nodes(B))
    sub = candb[rng.choice(len(candb))]
    node[:] = copy(sub)
    return A


def _mutate(t, rng):
    def copy(tt):
        return [tt[0]] + [copy(c) for c in tt[1:]]
    A = copy(t)
    def nodes(tt):
        yield tt
        for c in tt[1:]:
            yield from nodes(c)
    cand = list(nodes(A))
    node = cand[rng.choice(len(cand))]
    r = rng.random()
    if r < 0.3:
        # replace with random terminal
        node[:] = _rand_terminal(rng)
    elif r < 0.6:
        # change function (if it has children)
        if len(node) > 1:
            node[0] = rng.choice(_FUNCS)
            # fix arity
            if node[0] in ('neg','sin','cos','tan','exp','tanh','abs','sqrt','log','sq','cube'):
                if len(node) > 2:
                    node = node[:2]
            else:
                while len(node) < 3:
                    node.append(_rand_terminal(rng))
        else:
            node[:] = _rand_terminal(rng)
    else:
        # replace subtree
        node[:] = _rand_tree(min(3, _maxd), rng)
    return A


_maxd = 6


def _gp_search(x, y, deadline, rng):
    pop_size = 100
    best = None
    best_fit = np.inf
    # init population
    pop = [_rand_tree(_maxd, rng) for _ in range(pop_size)]
    # evaluate
    def fitness(tree):
        try:
            pred = _safe_eval_tree(tree, x)
            if not np.all(np.isfinite(pred)):
                return 1e12
            err = np.mean((pred - y) ** 2)
            # parsimony
            return err + 1e-4 * _tree_size(tree)
        except Exception:
            return 1e12
    fits = [fitness(t) for t in pop]
    idx = int(np.argmin(fits))
    best = pop[idx]
    best_fit = fits[idx]
    gen = 0
    while time.time() < deadline:
        gen += 1
        new_pop = []
        # elitism
        order = np.argsort(fits)
        new_pop.append(pop[order[0]])
        for _ in range(pop_size - 1):
            # tournament
            i1 = rng.integers(pop_size)
            i2 = rng.integers(pop_size)
            parent = pop[i1] if fits[i1] < fits[i2] else pop[i2]
            if rng.random() < 0.7:
                j1 = rng.integers(pop_size)
                j2 = rng.integers(pop_size)
                parent2 = pop[j1] if fits[j1] < fits[j2] else pop[j2]
                child = _crossover(parent, parent2, rng)
            else:
                child = _mutate(parent, rng)
            if _tree_depth(child) > _maxd + 2:
                child = _rand_tree(_maxd, rng)
            new_pop.append(child)
        pop = new_pop
        fits = [fitness(t) for t in pop]
        idx = int(np.argmin(fits))
        if fits[idx] < best_fit:
            best_fit = fits[idx]
            best = pop[idx]
    return best


def _spline_predict(xtr, ytr, s):
    from scipy.interpolate import UnivariateSpline
    sp = UnivariateSpline(xtr, ytr, s=s, k=3)
    return sp


def _rbf_predict(xtr, ytr, gamma, lam):
    d2 = ((xtr[:, None] - xtr[None, :]) ** 2)
    K = np.exp(-gamma * d2)
    w = np.linalg.solve(K + lam * np.eye(len(xtr)), ytr)
    return (xtr, w, gamma)


def _poly_predict(xtr, ytr, deg, lam):
    F = np.vander(xtr, N=deg + 1, increasing=True)
    w = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ ytr)
    return (F.shape[1], w)


def fit(X_train, y_train):
    X = _as_1d(X_train)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    n, d = X.shape
    rng = np.random.default_rng(98765)
    # validation split
    perm = rng.permutation(n)
    nval = max(20, int(0.3 * n))
    val_idx = perm[:nval]
    tr_idx = perm[nval:]
    xtr, ytr = X[tr_idx, 0], y[tr_idx]
    xval, yval = X[val_idx, 0], y[val_idx]

    def val_rmse(pred_fn):
        try:
            p = pred_fn(xval)
            p = np.asarray(p, dtype=float)
            if not np.all(np.isfinite(p)):
                return np.inf
            return float(np.mean((p - yval) ** 2))
        except Exception:
            return np.inf

    candidates = []  # (val_rmse, predict_fn_on_full_X)

    if d == 1:
        # 1) Splines
        # UnivariateSpline
        for s in [0.0, 1e-6, 1e-4, 1e-2, 0.1]:
            try:
                sp = _spline_predict(xtr, ytr, s)
                vr = val_rmse(lambda xv: sp(xv))
                if np.isfinite(vr):
                    candidates.append((vr, lambda Xv, sp=sp: sp(Xv[:, 0])))
            except Exception:
                pass
        
        # make_interp_spline (Cubic and Quintic)
        try:
            from scipy.interpolate import make_interp_spline
            # Sort data for spline
            sort_idx = np.argsort(xtr)
            xtr_s = xtr[sort_idx]
            ytr_s = ytr[sort_idx]
            
            for k in [3, 5]:
                try:
                    sp = make_interp_spline(xtr_s, ytr_s, k=k)
                    vr = val_rmse(lambda xv: sp(xv))
                    if np.isfinite(vr):
                        candidates.append((vr, lambda Xv, sp=sp: sp(Xv[:, 0])))
                except Exception:
                    pass
        except Exception:
            pass
        # 2) RBF
        for gamma in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]:
            for lam in [1e-6, 1e-4, 1e-3]:
                try:
                    model = _rbf_predict(xtr, ytr, gamma, lam)
                    xtr_m, w, g = model
                    def rbf_pred(Xv, xtr_m=xtr_m, w=w, g=g):
                        K = np.exp(-g * ((Xv[:, 0][:, None] - xtr_m[None, :]) ** 2))
                        return K @ w
                    vr = val_rmse(rbf_pred)
                    if np.isfinite(vr):
                        candidates.append((vr, rbf_pred))
                except Exception:
                    pass
        # 3) Poly ridge
        for deg in [3, 4, 5, 6]:
            for lam in [1e-8, 1e-6, 1e-4]:
                try:
                    model = _poly_predict(xtr, ytr, deg, lam)
                    D, w = model
                    def poly_pred(Xv, D=D, w=w):
                        F = np.vander(Xv[:, 0], N=D, increasing=True)
                        return F @ w
                    vr = val_rmse(poly_pred)
                    if np.isfinite(vr):
                        candidates.append((vr, poly_pred))
                except Exception:
                    pass
        # 4) Fourier Series Fit (captures periodicity)
        # Estimate frequency range based on input span
        x_min, x_max = np.min(xtr), np.max(xtr)
        span = x_max - x_min
        if span > 1e-6:
            # Try different numbers of harmonics
            for n_harm in [1, 2, 3, 5, 10, 20]:
                try:
                    # Build Fourier basis
                    # Assume base frequency 2*pi/span to cover at least 1 cycle
                    base_freq = 2 * np.pi / span
                    A = np.ones((len(xtr), 1))
                    for k in range(1, n_harm + 1):
                        f = k * base_freq
                        A = np.column_stack([A, np.sin(f * xtr), np.cos(f * xtr)])
                    
                    # Ridge regression for stability
                    lam = 1e-6
                    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
                    
                    def fourier_pred(Xv, A_cols=A.shape[1], base_freq=base_freq, w=w):
                        xv = Xv[:, 0]
                        Af = np.ones((len(xv), 1))
                        for k in range(1, int(np.ceil((A_cols-1)/2)) + 1):
                            f = k * base_freq
                            Af = np.column_stack([Af, np.sin(f * xv), np.cos(f * xv)])
                        # Handle case where A_cols is odd (only sine or cos missing? No, we add pairs)
                        # If A_cols was 1 + 2*n_harm, then we need n_harm pairs.
                        # The loop above reconstructs based on A_cols count?
                        # Better to store n_harm.
                        return Af @ w[:Af.shape[1]]
                    # Wait, the reconstruction logic above is fragile. Let's store n_harm.
                    n_harm_val = n_harm
                    def fourier_pred(Xv, n_harm=n_harm_val, base_freq=base_freq, w=w):
                        xv = Xv[:, 0]
                        Af = np.ones((len(xv), 1))
                        for k in range(1, n_harm + 1):
                            f = k * base_freq
                            Af = np.column_stack([Af, np.sin(f * xv), np.cos(f * xv)])
                        return Af @ w
                    vr = val_rmse(fourier_pred)
                    if np.isfinite(vr):
                        candidates.append((vr, fourier_pred))
                except Exception:
                    pass

        # 5) Chebyshev Polynomial Fit
        try:
            from numpy.polynomial import chebyshev as C
            # Map x to [-1, 1] for Chebyshev
            x_center = (x_max + x_min) / 2
            x_scale = (x_max - x_min) / 2
            if x_scale > 1e-6:
                xtr_norm = (xtr - x_center) / x_scale
                for deg in [3, 5, 7, 9, 12, 15]:
                    try:
                        # Use least squares on Chebyshev basis
                        # C.chebfit does this, but let's do it manually for control or use C.chebfit
                        # C.chebfit(x, y, deg) returns coefficients
                        coeffs = C.chebfit(xtr_norm, ytr, deg)
                        def cheb_pred(Xv, coeffs=coeffs, x_center=x_center, x_scale=x_scale):
                            xv = Xv[:, 0]
                            xv_norm = (xv - x_center) / x_scale
                            return C.chebval(xv_norm, coeffs)
                        vr = val_rmse(cheb_pred)
                        if np.isfinite(vr):
                            candidates.append((vr, cheb_pred))
                    except Exception:
                        pass
        except Exception:
            pass

    # Fallback candidates: KNN (Interpolation-like) and Quadratic
    try:
        # KNN with k=1 (Nearest Neighbor)
        from scipy.spatial import cKDTree
        tree_knn = cKDTree(X[:, 0])
        def knn_pred(Xv, k=1):
            Xv = _as_1d(Xv)
            d, idx = tree_knn.query(Xv[:, 0], k=k)
            return y[idx]
        # We need to handle the case where k=1 returns 1D array of indices
        # cKDTree.query with k=1 returns d shape (N,), idx shape (N,)
        # If k>1, shapes are (N, k)
        # Let's use k=1 for noiseless interpolation
        vr = val_rmse(knn_pred)
        if np.isfinite(vr):
            candidates.append((vr, knn_pred))
    except Exception:
        pass

    try:
        # Quadratic Fallback
        F = np.column_stack([np.ones(n), X[:, 0], X[:, 0] ** 2])
        w = np.linalg.lstsq(F, y, rcond=None)[0]
        def base_pred(Xv):
            Xv = _as_1d(Xv)
            Fv = np.column_stack([np.ones(len(Xv)), Xv[:, 0], Xv[:, 0] ** 2])
            return Fv @ w
        candidates.append((val_rmse(base_pred), base_pred))
    except Exception:
        pass

    if not candidates:
        # ultimate fallback: constant mean
        c = float(np.mean(y))
        return lambda Xv: np.full(_as_1d(Xv).shape[0], c)

    candidates.sort(key=lambda t: t[0])
    best_pred = candidates[0][1]

    # Final safety wrapper
    ymean = float(np.mean(y))
    def predict(X):
        Xv = _as_1d(X)
        try:
            out = best_pred(Xv)
            out = np.asarray(out, dtype=float).reshape(-1)
            if out.shape[0] != Xv.shape[0]:
                out = np.resize(out, Xv.shape[0])
            out = np.nan_to_num(out, nan=ymean, posinf=1e6, neginf=-1e6)
            if not np.all(np.isfinite(out)):
                out = np.full(Xv.shape[0], ymean)
            return out
        except Exception:
            return np.full(Xv.shape[0], ymean)

    return predict
# EVOLVE-BLOCK-END

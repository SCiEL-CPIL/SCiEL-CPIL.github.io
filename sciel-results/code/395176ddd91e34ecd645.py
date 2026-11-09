"""Second autocorrelation inequality: maximize ratio via smooth optimization with log-sum-exp."""
import numpy as np
import time
import sys

N = 50
DT = 1.0 / N  # interval spacing on [-1/4, 1/4]


def _autoconv(x):
    """Linear autoconvolution of x with itself."""
    return np.convolve(x, x, mode='full')


def _ratio(x):
    """Compute the ratio R = ||x||_2^2 / (||x||_1 * ||x*x||_inf)."""
    x = np.asarray(x, dtype=np.float64)
    l2_sq = np.sum(x ** 2)
    if l2_sq < 1e-30:
        return 0.0
    l1 = np.sum(np.abs(x))
    if l1 < 1e-30:
        return 0.0
    ac = _autoconv(x)
    ac_max = np.max(np.abs(ac))
    if ac_max < 1e-30:
        return 0.0
    return l2_sq / (l1 * ac_max)


def _soft_max(v, beta):
    """Soft approximation of max(v) using log-sum-exp."""
    vmax = np.max(v)
    if vmax < 1e-30:
        return 1e-30
    # beta * log-sum-exp(v / beta) approximates max(v) as beta -> 0
    shifted = v / beta
    shifted = shifted - np.max(shifted)  # for numerical stability
    s = np.sum(np.exp(shifted))
    return beta * (np.log(s) + np.max(shifted))


def _smooth_objective(x, beta):
    """
    Smooth surrogate for -R(x) = -||x||_2^2 / (||x||_1 * ||x*x||_inf).
    We minimize this (i.e., maximize R).
    Fix ||x||_2 = 1 by normalizing, so we minimize ||x||_1 * soft_max(conv(x,x), beta).
    """
    x = np.maximum(x, 0)
    l2_sq = np.sum(x ** 2)
    if l2_sq < 1e-30:
        return 1e10
    # Normalize to unit L2
    x = x / np.sqrt(l2_sq)
    l1 = np.sum(x)
    ac = _autoconv(x)
    ac_pos = np.abs(ac)
    smax = _soft_max(ac_pos, beta)
    if l1 < 1e-30 or smax < 1e-30:
        return 1e10
    # We want to minimize l1 * smax (since R = 1/(l1*smax) when ||x||_2=1)
    return l1 * smax


def _smooth_gradient(x, beta):
    """
    Numerical gradient of the smooth objective.
    """
    eps = 1e-7
    g = np.zeros_like(x)
    f0 = _smooth_objective(x, beta)
    for i in range(N):
        xp = x.copy()
        xp[i] += eps
        fp = _smooth_objective(xp, beta)
        xm = x.copy()
        xm[i] = max(0, x[i] - eps)
        fm = _smooth_objective(xm, beta)
        if x[i] <= eps:
            g[i] = (fp - f0) / eps
        else:
            g[i] = (fp - fm) / (2 * eps)
    return g


def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_x = np.ones(N, dtype=np.float64)
    best_score = _ratio(best_x)

    # Try several initial configurations
    inits = []
    # Uniform
    inits.append(np.ones(N))
    # Triangular
    t = np.linspace(-1, 1, N)
    inits.append(np.maximum(0, 1 - np.abs(t)))
    # Gaussian-like
    t = np.linspace(-1, 1, N)
    inits.append(np.exp(-t**2))
    # Sinc-like (positive part)
    t = np.linspace(-1, 1, N)
    sinc = np.sinc(t)
    inits.append(np.maximum(0, sinc))
    # Two peaks
    x2 = np.zeros(N)
    x2[N // 4] = 1.0
    x2[3 * N // 4] = 1.0
    inits.append(x2)
    # Ramp up then down
    x3 = np.concatenate([np.linspace(0, 1, N // 2), np.linspace(1, 0, N - N // 2)])
    inits.append(x3)
    # Cosine bump
    t = np.linspace(0, np.pi, N)
    inits.append(0.5 * (1 + np.cos(t)))
    # Sparse: every other
    xs = np.zeros(N)
    xs[::2] = 1.0
    inits.append(xs)
    # Sparse: every third
    xt = np.zeros(N)
    xt[::3] = 1.0
    inits.append(xt)
    # Concentrated center
    xc = np.zeros(N)
    xc[N // 2 - 2:N // 2 + 3] = 1.0
    inits.append(xc)

    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(15):
        x_rand = rng.random(N)
        inits.append(x_rand)

    # Beta values for soft-max approximation
    betas = [0.05, 0.1, 0.2, 0.5]

    from scipy.optimize import minimize

    # Phase 1: SLSQP on original nonsmooth objective (fast, good for many starts)
    for x0 in inits:
        if time.time() > deadline - 30:
            break

        x0 = np.asarray(x0, dtype=np.float64)
        if np.sum(x0**2) < 1e-30:
            continue
        x0 = x0 / np.sqrt(np.sum(x0**2))
        x0 = np.maximum(x0, 0)
        if np.sum(x0**2) < 1e-30:
            continue
        x0 = x0 / np.sqrt(np.sum(x0**2))

        def objective_sl(x):
            return -_ratio(x)

        bounds = [(0, None)] * N
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x**2) - 1.0,
                        'jac': lambda x: 2 * x}]

        try:
            result = minimize(objective_sl, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 150, 'ftol': 1e-12})
            x_opt = np.maximum(result.x, 0)
            if np.sum(x_opt**2) > 1e-30:
                x_opt = x_opt / np.sqrt(np.sum(x_opt**2))
                score = _ratio(x_opt)
                if score > best_score:
                    best_score = score
                    best_x = x_opt
        except Exception:
            pass

    # Phase 2: L-BFGS-B with smooth surrogate (better convergence in smooth regions)
    for x0 in inits[:10]:  # Limit to save time
        if time.time() > deadline - 15:
            break

        x0 = np.asarray(x0, dtype=np.float64)
        if np.sum(x0**2) < 1e-30:
            continue
        x0 = x0 / np.sqrt(np.sum(x0**2))
        x0 = np.maximum(x0, 0)
        if np.sum(x0**2) < 1e-30:
            continue
        x0 = x0 / np.sqrt(np.sum(x0**2))

        for beta in betas:
            if time.time() > deadline - 5:
                break

            def objective_smooth(x, b=beta):
                return _smooth_objective(x, b)

            def gradient_smooth(x, b=beta):
                return _smooth_gradient(x, b)

            bounds = [(0, None)] * N
            try:
                result = minimize(objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                                  jac=gradient_smooth,
                                  options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-8})
                x_opt = np.maximum(result.x, 0)
                if np.sum(x_opt**2) > 1e-30:
                    x_opt = x_opt / np.sqrt(np.sum(x_opt**2))
                    score = _ratio(x_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
            except Exception:
                pass

    # Phase 3: Final polish with small perturbations and re-optimization
    if time.time() < deadline - 5:
        # Try perturbations around best
        for _ in range(30):
            if time.time() > deadline - 2:
                break
            x_pert = best_x + 0.01 * rng.standard_normal(N)
            x_pert = np.maximum(x_pert, 0)
            if np.sum(x_pert**2) < 1e-30:
                continue
            x_pert = x_pert / np.sqrt(np.sum(x_pert**2))
            score = _ratio(x_pert)
            if score > best_score:
                best_score = score
                best_x = x_pert

        # One more SLSQP polish from the best
        if time.time() < deadline - 3:
            x0 = best_x.copy()

            def objective_final(x):
                return -_ratio(x)

            bounds = [(0, None)] * N
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x**2) - 1.0,
                            'jac': lambda x: 2 * x}]

            try:
                result = minimize(objective_final, x0, method='SLSQP', bounds=bounds,
                                  constraints=constraints, options={'maxiter': 200, 'ftol': 1e-14})
                x_opt = np.maximum(result.x, 0)
                if np.sum(x_opt**2) > 1e-30:
                    x_opt = x_opt / np.sqrt(np.sum(x_opt**2))
                    score = _ratio(x_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
            except Exception:
                pass

    return list(best_x)


# Keep the interface exactly as required
if __name__ == '__main__':
    result = construct()
    print(f"Score: {_ratio(result)}")

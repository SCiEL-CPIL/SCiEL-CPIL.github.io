import numpy as np
import scipy.optimize as opt
import time

# Physicists' Hermite polynomials
def H0(x): return np.ones_like(x)
def H4(x): return 16*x**4 - 48*x**2 + 12
def H8(x): return 256*x**8 - 3840*x**6 + 13440*x**4 - 13440*x**2 + 1680
def H12(x):
    return (4096*x**12 - 122880*x**10 + 1290240*x**8 - 6635520*x**6
            + 14515200*x**4 - 14515200*x**2 + 4790016)

# Pre-compute on a moderately dense grid (fast vectorized eval)
X = np.linspace(-15.0, 15.0, 1500)
DX = X[1] - X[0]
GAUSS = np.exp(-X**2 / 2.0)

H0_G = H0(X) * GAUSS
H4_G = H4(X) * GAUSS
H8_G = H8(X) * GAUSS
H12_G = H12(X) * GAUSS

# Values at 0 (for g(0)=0 constraint)
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 4790016.0

def compute_d(a, b, c):
    return -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0

def _ratio(g):
    n2 = np.sum(g * g) * DX
    if n2 <= 0.0:
        return 1e30
    n4 = np.sum(g * g * g * g) * DX
    return n4 / n2

def objective(params):
    a, b, c = params
    d = compute_d(a, b, c)
    g = a * H0_G + b * H4_G + c * H8_G + d * H12_G
    return _ratio(g)

def construct():
    start = time.time()
    deadline = start + 195.0  # safety margin before 210s kill

    # Seed values
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = objective(x0)
    best_params = x0.copy()

    def try_update(p, v):
        nonlocal best, best_params
        if v < best:
            best = v
            best_params = np.asarray(p, dtype=float).copy()

    # --- Phase 1: coarse global grid to locate the best basin ---
    try:
        if time.time() < deadline - 15.0:
            grid = np.array([
                [a, b, c]
                for a in np.linspace(-2.0, 2.0, 15)
                for b in np.linspace(-0.4, 0.4, 13)
                for c in np.linspace(-0.05, 0.05, 11)
            ])
            vals = np.empty(len(grid))
            for i, p in enumerate(grid):
                vals[i] = objective(p)
            k = int(np.argmin(vals))
            try_update(grid[k], vals[k])
            # local refine around the grid best
            if time.time() < deadline - 20.0:
                res = opt.minimize(objective, grid[k], method='L-BFGS-B',
                                   options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
                try_update(res.x, res.fun)
    except Exception:
        pass

    # --- Phase 2: local refinement from the seed too ---
    try:
        if time.time() < deadline - 15.0:
            res = opt.minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            try_update(res.x, res.fun)
    except Exception:
        pass

    # --- Phase 3: high-precision Nelder-Mead polish ---
    try:
        if time.time() < deadline - 8.0:
            res = opt.minimize(objective, best_params, method='Nelder-Mead',
                               options={'xatol': 1e-14, 'fatol': 1e-16,
                                        'maxiter': 20000, 'adaptive': True})
            try_update(res.x, res.fun)
    except Exception:
        pass

    # --- Phase 4: final tight L-BFGS-B polish ---
    try:
        if time.time() < deadline - 4.0:
            res = opt.minimize(objective, best_params, method='L-BFGS-B',
                               options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18})
            try_update(res.x, res.fun)
    except Exception:
        pass

    return np.array(best_params)

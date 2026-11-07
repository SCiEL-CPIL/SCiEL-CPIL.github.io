import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize

N = 50

def construct():
    K = 40
    
    def compute_value_from_h(h):
        s = np.sum(h)
        if s == 0:
            return 0.0
        h = h / s
        g = np.convolve(h, h)
        m = len(g)
        y = np.zeros(m + 2)
        y[1:-1] = g
        w = 1.0 / (m + 1)
        y_sq = y**2
        y_prod = y[:-1] * y[1:]
        l2 = (w / 3.0) * np.sum(y_sq[:-1] + y_prod + y_sq[1:])
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        if norm_1 == 0 or norm_inf == 0:
            return 0.0
        return l2 / (norm_1 * norm_inf)

    def objective(c):
        c_full = np.zeros(N)
        c_full[:K] = c
        h_raw = idct(c_full, type=2, norm='ortho')
        h = np.sqrt(h_raw**2 + 1e-10)
        return -compute_value_from_h(h)

    best_val = -np.inf
    best_c = None

    np.random.seed(42)

    inits = []
    for i in range(20):
        c0 = np.zeros(K)
        if i == 0:
            c0[0] = 1.0
        elif i == 1:
            c0[0] = 0.8
            c0[1] = 0.3
        elif i == 2:
            c0[0] = 0.9
            c0[2] = 0.2
        elif i == 3:
            c0[0] = 1.0
            c0[1:5] = 0.1
        else:
            c0 = np.random.randn(K) * 0.1
            c0[0] = np.random.uniform(0.5, 1.0)
            if i % 2 == 0:
                c0[10:] = 0
        inits.append(c0)

    for c0 in inits:
        try:
            res = minimize(objective, c0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except Exception:
            pass

    if best_c is None:
        best_c = np.zeros(K)
        best_c[0] = 1.0

    c_full = np.zeros(N)
    c_full[:K] = best_c
    h_raw = idct(c_full, type=2, norm='ortho')
    h = np.abs(h_raw)

    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = np.ones(N) / N

    return h

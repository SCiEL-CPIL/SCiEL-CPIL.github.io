import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize

N = 50

def construct():
    K = 20

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
        h = np.abs(h_raw)
        return -compute_value_from_h(h)

    def numerical_gradient(c):
        eps = 1e-8
        grad = np.zeros(K)
        f0 = objective(c)
        for i in range(K):
            c_plus = c.copy()
            c_plus[i] += eps
            f_plus = objective(c_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    best_val = -np.inf
    best_c = None

    np.random.seed(42)

    for restart in range(8):
        if restart == 0:
            c0 = np.zeros(K)
            c0[0] = 1.0
        elif restart == 1:
            c0 = np.zeros(K)
            c0[0] = 0.9
            c0[1] = 0.2
            c0[2] = -0.1
        elif restart == 2:
            c0 = np.random.randn(K) * 0.1
            c0[0] = 1.0
        elif restart == 3:
            c0 = np.zeros(K)
            c0[0] = 1.0
            for i in range(1, min(5, K)):
                c0[i] = 0.1 * (1 - i/5)
        elif restart == 4:
            c0 = np.zeros(K)
            c0[0] = 0.8
            c0[1] = 0.3
            c0[3] = 0.1
        elif restart == 5:
            c0 = np.random.randn(K) * 0.15
            c0[0] = max(0.7, c0[0])
        elif restart == 6:
            c0 = np.zeros(K)
            c0[0] = 0.5
            c0[5:10] = 0.2
        else:
            c0 = np.random.randn(K) * 0.2
            c0[0] = max(0.6, c0[0])

        try:
            res = minimize(objective, c0, jac=numerical_gradient, method='L-BFGS-B',
                          options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-6})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except Exception:
            pass

    if best_c is None:
        c0 = np.zeros(K)
        c0[0] = 1.0
        try:
            res = minimize(objective, c0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_val:
                best_val = -res.fun
                best_c = res.x.copy()
        except:
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

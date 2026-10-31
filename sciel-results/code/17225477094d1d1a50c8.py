import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    def autocorr(h, t):
        return np.sum(h * np.roll(h, t))
    
    def objective(x):
        lam = x[N]
        return -lam
    
    def make_constraint(t_val):
        def constraint(x):
            h = x[:N]
            lam = x[N]
            return autocorr(h, t_val) - lam
        return constraint
    
    def sum_con(x):
        return np.sum(x[:N]) - N / 2.0
    
    cons = []
    for t in range(N):
        cons.append({'type': 'ineq', 'fun': make_constraint(t)})
    cons.append({'type': 'eq', 'fun': sum_con})
    
    x0 = np.zeros(N + 1)
    x0[:N] = 0.5
    x0[N] = N / 4.0
    
    bounds = [(0.0, 1.0)] * N + [(-N/2.0, N/2.0)]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
        h = np.clip(res.x[:N], 0, 1)
    except Exception:
        h = np.full(N, 0.5)
    
    # Ensure sum is exactly N/2
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0, 1)
    # Fine-tune sum
    diff = N / 2.0 - np.sum(h)
    idx = np.argsort(h)[::-1]
    for i in idx:
        if diff > 0:
            add = min(diff, 1.0 - h[i])
            h[i] += add
            diff -= add
        elif diff < 0:
            sub = min(-diff, h[i])
            h[i] -= sub
            diff += sub
        else:
            break
    
    return h

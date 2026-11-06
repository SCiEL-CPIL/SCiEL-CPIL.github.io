import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    # Step 1: Hexagonal lattice initialization
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            idx = np.argsort(radius, kind='stable')[:N]
            init = points[idx].copy()
            break
        spacing *= 0.98
    
    t0 = time.time()
    deadline = t0 + 115.0
    
    def make_obj(beta):
        iu = np.triu_indices(N, k=1)
        def neg_r_soft_and_grad(flat):
            c = flat.reshape(-1, 2)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(d, np.inf)
            d_pairs = d[iu]
            r = np.sqrt(np.sum(c**2, axis=1))
            b = 1.0 - r
            vals_pairs = d_pairs / 2.0
            vals_b = b
            exp_pairs = np.exp(-beta * vals_pairs)
            exp_b = np.exp(-beta * vals_b)
            S = np.sum(exp_pairs) + np.sum(exp_b)
            r_soft = -(1.0 / beta) * np.log(S)
            w_pairs = exp_pairs / S
            w_b = exp_b / S
            grad = np.zeros_like(c)
            ii, jj = iu
            vecs = c[ii] - c[jj]
            d_safe = np.where(d_pairs > 1e-12, d_pairs, 1.0)
            unit_vecs = vecs / d_safe[:, np.newaxis]
            np.add.at(grad, ii, w_pairs[:, np.newaxis] * unit_vecs * 0.5)
            np.add.at(grad, jj, -w_pairs[:, np.newaxis] * unit_vecs * 0.5)
            r_safe = np.where(r > 1e-12, r, 1.0)
            grad_b = -c / r_safe[:, np.newaxis]
            grad += w_b[:, np.newaxis] * grad_b
            return -r_soft, -grad
        return neg_r_soft_and_grad
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    x0 = init.flatten()
    best_c = init.copy()
    best_score = -np.inf
    
    def compute_score(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        r = np.sqrt(np.sum(c**2, axis=1))
        min_b = np.min(1.0 - r)
        return min(min_pair, min_b)
    
    for beta in [20.0, 50.0, 100.0, 200.0, 500.0]:
        if time.time() > deadline - 20.0:
            break
        func = make_obj(beta)
        try:
            result = minimize(func, x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            c = result.x.reshape(-1, 2)
            score = compute_score(c)
            if score > best_score:
                best_score = score
                best_c = c.copy()
                x0 = c.flatten()
        except Exception:
            pass
    
    # Verify validity
    r = np.sqrt(np.sum(best_c**2, axis=1))
    if np.any(r > 1.0 + 1e-9) or np.any(~np.isfinite(best_c)):
        return init
    
    return best_c

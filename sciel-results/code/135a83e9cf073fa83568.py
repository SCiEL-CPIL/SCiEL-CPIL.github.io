import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def is_valid(x, tol=1e-8):
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-10:
                return False
            if xi - ri < -tol or 1 - xi - ri < -tol or yi - ri < -tol or 1 - yi - ri < -tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(x[3*i]-x[3*j], x[3*i+1]-x[3*j+1])
                if d < x[3*i+2] + x[3*j+2] - tol:
                    return False
        return True
    
    def total_radius(x):
        return float(np.sum(x[2::3]))
    
    # Initial: 6x6 grid, take first 26
    side = 6
    init = np.array([[(i+0.5)/side, (j+0.5)/side, 0.5/side - 0.001] 
                     for i in range(side) for j in range(side)][:N]).flatten()
    
    best = init.copy()
    best_score = total_radius(init)
    
    rng = np.random.RandomState(12345)
    
    def objective(x):
        radii = x[2::3]
        val = -np.sum(radii)
        eps = 1e-8
        pen = 0.0
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        # Wall penalties (smooth)
        wall_v = np.concatenate([rs-xs, rs-(1-xs), rs-ys, rs-(1-ys)])
        pos_wall_v = np.maximum(0, wall_v)
        pen += np.sum(0.5*(pos_wall_v + np.sqrt(pos_wall_v**2 + eps)))
        # Pair penalties
        for i in range(N):
            xi, yi, ri = xs[i], ys[i], rs[i]
            for j in range(i+1, N):
                xj, yj, rj = xs[j], ys[j], rs[j]
                d = np.hypot(xi-xj, yi-yj)
                v = ri + rj - d
                if v > 0:
                    pen += 0.5*(v + np.sqrt(v*v + eps))
        return val + 100.0 * pen
    
    def optimize_from(x0):
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'gtol': 1e-6})
            if is_valid(res.x):
                return res.x
        except:
            pass
        return None
    
    # Multiple random restarts
    n_restarts = 50
    for trial in range(n_restarts):
        if time.time() > deadline - 20:
            break
        if trial == 0:
            x0 = init.copy()
        else:
            x0 = init.copy()
            x0[0::3] += rng.randn(N) * 0.05
            x0[1::3] += rng.randn(N) * 0.05
            x0[2::3] *= (1.0 + rng.randn(N) * 0.1)
            x0[2::3] = np.maximum(x0[2::3], 0.01)
            # Clip positions to valid range given radii
            for i in range(N):
                ri = x0[3*i+2]
                x0[3*i] = np.clip(x0[3*i], ri, 1-ri)
                x0[3*i+1] = np.clip(x0[3*i+1], ri, 1-ri)
        
        res = optimize_from(x0)
        if res is not None:
            s = total_radius(res)
            if s > best_score:
                best = res.copy()
                best_score = s
    
    # Final polish
    if time.time() < deadline - 5:
        def objective_polish(x):
            radii = x[2::3]
            val = -np.sum(radii)
            eps = 1e-10
            pen = 0.0
            xs = x[0::3]
            ys = x[1::3]
            rs = x[2::3]
            wall_v = np.concatenate([rs-xs, rs-(1-xs), rs-ys, rs-(1-ys)])
            pos_wall_v = np.maximum(0, wall_v)
            pen += np.sum(0.5*(pos_wall_v + np.sqrt(pos_wall_v**2 + eps)))
            for i in range(N):
                xi, yi, ri = xs[i], ys[i], rs[i]
                for j in range(i+1, N):
                    xj, yj, rj = xs[j], ys[j], rs[j]
                    d = np.hypot(xi-xj, yi-yj)
                    v = ri + rj - d
                    if v > 0:
                        pen += 0.5*(v + np.sqrt(v*v + eps))
            return val + 1000.0 * pen
        
        try:
            res = minimize(objective_polish, best, method='L-BFGS-B',
                          options={'maxiter': 1000, 'gtol': 1e-8})
            if is_valid(res.x):
                s = total_radius(res.x)
                if s > best_score:
                    best = res.x.copy()
                    best_score = s
        except:
            pass
    
    return best.reshape(N, 3)

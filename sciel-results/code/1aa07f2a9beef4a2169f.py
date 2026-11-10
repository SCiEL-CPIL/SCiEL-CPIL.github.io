import numpy as np
from scipy.optimize import differential_evolution, minimize, Powell
from math import log, pi, cos, sin
import time

def construct():
    def objective(x):
        p1 = x[0:2]
        p2 = x[2:4]
        p3 = x[4:6]
        p4 = x[6:8]
        
        P = np.array([
            p1, p2, p3, p4,
            -p1, -p2, -p3, -p4
        ])
        
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        dists = np.maximum(dists, 1e-12)
        
        iu = np.triu_indices(8, k=1)
        log_dists = np.log(dists[iu])
        sum_log_dists = np.sum(log_dists)
        
        D = np.max(dists)
        log_D = np.log(D)
        
        return sum_log_dists - 28.0 * log_D

    bounds = [(-1, 1)] * 8
    
    start_time = time.time()
    time_limit = 120.0
    
    try:
        # Run DE
        res = differential_evolution(objective, bounds, maxiter=300, popsize=20, seed=42, tol=0.01)
        
        if time.time() - start_time > time_limit:
            raise TimeoutError()

        best_x = res.x
        best_val = -res.fun
        
        # Polish with Powell
        if time.time() - start_time < time_limit - 5:
            res2 = minimize(objective, best_x, method='Powell', options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-12})
            if -res2.fun > best_val:
                best_x = res2.x
                best_val = -res2.fun
            
        # Try Nelder-Mead
        if time.time() - start_time < time_limit - 2:
            res3 = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if -res3.fun > best_val:
                best_x = res3.x
                best_val = -res3.fun
                
        p1 = best_x[0:2]
        p2 = best_x[2:4]
        p3 = best_x[4:6]
        p4 = best_x[6:8]
        
        P = np.array([
            p1, p2, p3, p4,
            -p1, -p2, -p3, -p4
        ])
        
        D = np.max(np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2))
        if D > 0:
            P = P * (2.0 / D)
            
        return P.tolist()

    except Exception as e:
        pts = []
        for k in range(8):
            angle = 2 * pi * k / 8
            pts.append([cos(angle), sin(angle)])
        return pts

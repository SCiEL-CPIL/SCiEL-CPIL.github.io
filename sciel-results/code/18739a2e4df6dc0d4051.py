import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Initial honeycomb arrangement (3 rows x 4 cols)
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * 3.0 + (1.5 if row % 2 == 1 else 0.0)
            y = row * sqrt3 / 2
            positions.append([x, y, 0.0])
    
    # Center
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx0
        p[1] -= cy0
    
    def hex_vertices(cx, cy, angle, s=1.0):
        return np.array([[cx + s*np.cos(angle + j*np.pi/3), 
                         cy + s*np.sin(angle + j*np.pi/3)] for j in range(6)])
    
    def max_violation(vars, S):
        n = N
        cx = vars[3*n]
        cy = vars[3*n+1]
        theta = vars[3*n+2]
        apothem = S * sqrt3 / 2
        max_v = 0.0
        for i in range(n):
            xi = vars[3*i]
            yi = vars[3*i+1]
            ai = vars[3*i+2]
            verts = hex_vertices(xi, yi, ai)
            for v in verts:
                for k in range(6):
                    phi = theta + k * np.pi / 3
                    val = (v[0]-cx)*np.cos(phi) + (v[1]-cy)*np.sin(phi)
                    if val > max_v:
                        max_v = val
        return max_v - apothem
    
    def objective(vars):
        S = vars[-1]
        v = max_violation(vars, S)
        if v > 0:
            return S + 500.0 * v * v
        return S
    
    x0 = []
    for p in positions:
        x0.extend([p[0], p[1], 0.0])
    x0.extend([0.0, 0.0, 0.0, 6.0])
    
    bounds = []
    for _ in range(N):
        bounds.extend([(-15, 15), (-15, 15), (-np.pi, np.pi)])
    bounds.extend([(-15, 15), (-15, 15), (-np.pi, np.pi), (1.0, 15.0)])
    
    best_S = x0[-1]
    best_vars = x0.copy()
    
    def try_optimize(x0_init, maxiter=300):
        nonlocal best_S, best_vars
        try:
            res = minimize(objective, x0_init, method='SLSQP', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-7})
            S_val = res.x[-1]
            v = max_violation(res.x, S_val)
            if v < 1e-4 and S_val < best_S:
                best_S = S_val
                best_vars = res.x
            elif v < 1e-4 and res.fun < best_S:
                best_S = res.fun
                best_vars = res.x
        except Exception:
            pass
    
    try_optimize(x0, 500)
    
    # Random restarts
    np.random.seed(42)
    for r in range(20):
        if time.time() - start_time > deadline:
            break
        x0r = best_vars.copy()
        for i in range(N):
            x0r[3*i] += np.random.uniform(-0.3, 0.3)
            x0r[3*i+1] += np.random.uniform(-0.3, 0.3)
            x0r[3*i+2] += np.random.uniform(-0.2, 0.2)
        x0r[-1] = best_S + 0.3
        try_optimize(x0r, 200)
    
    # Final polish
    if time.time() - start_time < deadline - 2:
        try_optimize(best_vars, 1000)
    
    inner = []
    for i in range(N):
        inner.append([best_vars[3*i], best_vars[3*i+1], np.degrees(best_vars[3*i+2])])
    
    cx = best_vars[3*N]
    cy = best_vars[3*N+1]
    theta = best_vars[3*N+2]
    S = best_vars[3*N+3]
    
    return inner, (cx, cy), S, np.degrees(theta)

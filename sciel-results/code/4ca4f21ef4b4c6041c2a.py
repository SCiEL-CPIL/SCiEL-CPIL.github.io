import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import time

def _compute_area(path):
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    W = np.stack([X.ravel(), Y.ravel()], axis=1)
    inside = np.ones(len(W), dtype=bool)
    for x, y, a_deg in path:
        a = np.radians(a_deg)
        ca, sa = np.cos(a), np.sin(a)
        dx = W[:, 0] - x
        dy = W[:, 1] - y
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        inA = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        inB = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        inside &= (inA | inB)
    return inside.sum() * 0.0001

def moving_sofa_2d() -> np.ndarray:
    K = 10
    angles = np.linspace(0, 90, K)
    
    cp0 = np.array([
        [0.0, 0.0],
        [-0.2, 0.1],
        [-0.4, 0.2],
        [-0.6, 0.2],
        [-0.8, 0.15],
        [-1.0, 0.1],
        [-1.2, 0.1],
        [-1.4, 0.1],
        [-1.7, 0.05],
        [-2.0, 0.0],
    ])
    
    def objective(params):
        cp = np.zeros((K, 2))
        cp[0] = [0, 0]
        cp[-1] = [-2, 0]
        cp[1:-1] = params.reshape(-1, 2)
        
        try:
            cs_x = CubicSpline(angles, cp[:, 0], bc_type='not-a-knot')
            cs_y = CubicSpline(angles, cp[:, 1], bc_type='not-a-knot')
        except Exception:
            return -1.0
        
        a_int = np.arange(0, 91)
        x_vals = cs_x(a_int)
        y_vals = cs_y(a_int)
        path = np.column_stack([x_vals, y_vals, a_int.astype(float)])
        
        return _compute_area(path)
    
    x0 = cp0[1:-1].ravel()
    best_params = x0.copy()
    best_val = objective(x0)
    
    start_time = time.time()
    timeout = 550
    
    for _ in range(20):
        if time.time() - start_time > timeout:
            break
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9,
                                'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    cp = np.zeros((K, 2))
    cp[0] = [0, 0]
    cp[-1] = [-2, 0]
    cp[1:-1] = best_params.reshape(-1, 2)
    
    cs_x = CubicSpline(angles, cp[:, 0], bc_type='not-a-knot')
    cs_y = CubicSpline(angles, cp[:, 1], bc_type='not-a-knot')
    a_int = np.arange(0, 91)
    x_vals = cs_x(a_int)
    y_vals = cs_y(a_int)
    path = np.column_stack([x_vals, y_vals, a_int.astype(float)])
    
    return path

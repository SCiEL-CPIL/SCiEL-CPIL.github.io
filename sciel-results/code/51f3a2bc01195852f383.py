import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_norm(x, y):
    sqrt3 = np.sqrt(3.0)
    val1 = np.abs(x)
    val2 = np.abs(y) / (sqrt3 / 2.0)
    val3 = np.abs(x + y / sqrt3)
    val4 = np.abs(x - y / sqrt3)
    return np.maximum(np.maximum(val1, val2), np.maximum(val3, val4))

def _objective(params):
    centers = params[:3*N].reshape(N, 3)
    S = params[3*N]
    
    thetas = centers[:, 2]
    xs = centers[:, 0]
    ys = centers[:, 1]
    
    k = np.arange(6) * np.pi / 3.0
    cos_k = np.cos(k)
    sin_k = np.sin(k)
    
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    vx = cos_t[:, np.newaxis] * cos_k[np.newaxis, :] - sin_t[:, np.newaxis] * sin_k[np.newaxis, :]
    vy = sin_t[:, np.newaxis] * cos_k[np.newaxis, :] + cos_t[:, np.newaxis] * sin_k[np.newaxis, :]
    
    vx = xs[:, np.newaxis] + vx
    vy = ys[:, np.newaxis] + vy
    
    h_vals = _hex_norm(vx.flatten(), vy.flatten())
    
    violations = np.maximum(0.0, h_vals - S)
    penalty = np.sum(violations**2)
    
    return S + penalty

def construct():
    start_time = time.time()
    
    # Initial configuration
    init_centers = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, np.sqrt(3)/2, 0.0),
        (-0.5, np.sqrt(3)/2, 0.0),
        (-1.0, 0.0, 0.0),
        (-0.5, -np.sqrt(3)/2, 0.0),
        (0.5, -np.sqrt(3)/2, 0.0),
        (1.5, np.sqrt(3)/2, 0.0),
        (1.5, -np.sqrt(3)/2, 0.0),
        (0.0, np.sqrt(3), 0.0),
        (-1.5, np.sqrt(3)/2, 0.0)
    ]
    
    x0 = [c[0] for c in init_centers]
    y0 = [c[1] for c in init_centers]
    t0 = [c[2] for c in init_centers]
    S0 = 5.0
    
    params0 = np.array(x0 + y0 + t0 + [S0])
    
    # Nelder-Mead
    res = minimize(_objective, params0, method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    params = res.x
    centers = params[:3*N].reshape(N, 3)
    S = params[3*N]
    
    # Verify and fix S if needed
    thetas = centers[:, 2]
    xs = centers[:, 0]
    ys = centers[:, 1]
    
    k = np.arange(6) * np.pi / 3.0
    cos_k = np.cos(k)
    sin_k = np.sin(k)
    
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    vx = cos_t[:, np.newaxis] * cos_k[np.newaxis, :] - sin_t[:, np.newaxis] * sin_k[np.newaxis, :]
    vy = sin_t[:, np.newaxis] * cos_k[np.newaxis, :] + cos_t[:, np.newaxis] * sin_k[np.newaxis, :]
    
    vx = xs[:, np.newaxis] + vx
    vy = ys[:, np.newaxis] + vy
    
    h_vals = _hex_norm(vx.flatten(), vy.flatten())
    max_h = np.max(h_vals)
    
    if max_h > S:
        S = max_h
        
    inner = []
    for i in range(N):
        x, y, t_deg = centers[i, 0], centers[i, 1], np.degrees(centers[i, 2])
        inner.append([x, y, t_deg])
        
    return inner, (0.0, 0.0), float(S), 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END

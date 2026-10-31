import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0

def compute_ratio(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    s = np.sum(h)
    if s <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf <= 0:
        return 0.0
    return l2_sq / (s * conv_inf)

def objective_log(x):
    h = np.exp(x)
    s = np.sum(h)
    if s <= 0:
        return 1e10
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf <= 0:
        return 1e10
    ratio = l2_sq / (s * conv_inf)
    return -ratio

def construct():
    np.random.seed(42)
    
    best_x = None
    best_val = 0.0
    
    initial_shapes = []
    
    # Uniform
    initial_shapes.append(np.ones(N) / N)
    
    # Gaussian centered
    x_grid = np.linspace(-1, 1, N)
    g = np.exp(-4*x_grid**2)
    initial_shapes.append(g / np.sum(g))
    
    # Sparse: single peak
    for peak_pos in [N//4, N//2, 3*N//4]:
        shape = np.zeros(N)
        shape[peak_pos] = 1.0
        initial_shapes.append(shape)
    
    # Sparse: two peaks
    for p1, p2 in [(N//4, 3*N//4), (N//5, 4*N//5), (N//3, 2*N//3)]:
        shape = np.zeros(N)
        shape[p1] = 0.5
        shape[p2] = 0.5
        initial_shapes.append(shape)
    
    # Sparse: three peaks
    shape = np.zeros(N)
    shape[N//6] = 1/3
    shape[N//2] = 1/3
    shape[5*N//6] = 1/3
    initial_shapes.append(shape)
    
    # Sparse: alternating
    shape = np.zeros(N)
    shape[::2] = 1.0/(N//2)
    initial_shapes.append(shape)
    
    # Random sparse
    for _ in range(8):
        shape = np.zeros(N)
        n_spikes = np.random.randint(2, 9)
        positions = np.random.choice(N, size=n_spikes, replace=False)
        weights = np.random.dirichlet(np.ones(n_spikes))
        shape[positions] = weights
        initial_shapes.append(shape)
    
    # Smooth windows
    for center in [N//4, N//2, 3*N//4]:
        for width in [4, 7, 10]:
            shape = np.zeros(N)
            for i in range(N):
                d = abs(i - center)
                if d < width:
                    shape[i] = 0.5*(1 + np.cos(np.pi*d/width))
            if np.sum(shape) > 0:
                shape /= np.sum(shape)
                initial_shapes.append(shape)
    
    n_init = len(initial_shapes)
    for idx, shape in enumerate(initial_shapes):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        shape_pos = np.maximum(shape, 1e-8)
        x0 = np.log(shape_pos)
        
        remaining = TIME_LIMIT - 10 - (time.time() - START_TIME)
        time_per = max(3.0, remaining / max(1, n_init - idx))
        
        result = minimize(objective_log, x0, method='L-BFGS-B',
                         options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12,
                                  'maxfun': 2000, 'time_limit': time_per})
        
        x_opt = result.x
        h_opt = np.exp(x_opt)
        ratio = compute_ratio(h_opt)
        
        if ratio > best_val:
            best_val = ratio
            best_x = x_opt.copy()
    
    if best_x is None:
        return [1.0/N] * N
    
    h_result = np.exp(best_x)
    result = np.maximum(0, h_result)
    return [float(x) for x in result]

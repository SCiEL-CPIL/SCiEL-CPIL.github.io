import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    def overlaps(h):
        h = np.asarray(h)
        return np.array([np.sum(h * (1 - np.roll(h, k))) for k in range(N)])
    
    def objective(x):
        ov = overlaps(x)
        beta = 200.0
        m = np.max(ov)
        return m + np.log(np.sum(np.exp(beta * (ov - m)))) / beta
    
    def constraint_sum(x):
        return np.sum(x) - N / 2
    
    # Try a few initializations
    best_x = np.full(N, 0.5)
    best_val = objective(best_x)
    
    # Binary initialization: alternating
    x_bin = np.zeros(N)
    x_bin[::2] = 1.0
    # Adjust to have exactly 50 ones
    # x_bin already has 50 ones
    val_bin = objective(x_bin)
    if val_bin < best_val:
        best_val = val_bin
        best_x = x_bin.copy()
    
    # Another init: first half 1, second half 0
    x_half = np.zeros(N)
    x_half[:N//2] = 1.0
    val_half = objective(x_half)
    if val_half < best_val:
        best_val = val_half
        best_x = x_half.copy()
    
    # Optimize from best initialization
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    bounds = [(0, 1)] * N
    
    result = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 500, 'ftol': 1e-14})
    
    if time.time() > deadline:
        return best_x
    
    x_opt = result.x
    x_opt = np.clip(x_opt, 0, 1)
    # Enforce sum constraint precisely
    diff = N/2 - np.sum(x_opt)
    x_opt += diff / N
    x_opt = np.clip(x_opt, 0, 1)
    
    return x_opt

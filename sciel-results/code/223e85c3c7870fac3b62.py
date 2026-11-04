import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    def objective(y):
        # y: array of shape (N-1,) representing log-gaps for points 1 to N-1
        # Point 0 log-gap is fixed to 0 to break symmetry
        log_gaps = np.empty(N)
        log_gaps[0] = 0.0
        log_gaps[1:] = y
        
        # Softmax to get positive gaps summing to 2*pi
        max_log = np.max(log_gaps)
        exp_gaps = np.exp(log_gaps - max_log)
        sums = np.sum(exp_gaps)
        gaps = 2 * np.pi * exp_gaps / sums
        
        # Calculate cumulative angles
        phi = np.zeros(N)
        phi[1:] = np.cumsum(gaps[:N-1])
        
        # Calculate minimum triangle area numerator
        min_val = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = phi[j] - phi[i]
                    b = phi[k] - phi[j]
                    c = 2 * np.pi - a - b
                    val = np.sin(a) + np.sin(b) + np.sin(c)
                    if val < min_val:
                        min_val = val
        
        # Hull area denominator (proportional to sum of sines of gaps)
        hull_area_num = np.sum(np.sin(gaps))
        
        if hull_area_num <= 0:
            return 0.0
            
        ratio = min_val / hull_area_num
        return -ratio

    best_result = None
    best_score = -float('inf')
    
    # Multi-start strategy
    starts = [
        np.zeros(N-1),
        np.linspace(-0.5, 0.5, N-1),
        np.array([1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1, 0.05, -0.05, 0.01, -0.01, 0.001])
    ]
    
    for start in starts:
        res = minimize(
            objective, 
            start, 
            method='Powell', 
            options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10}
        )
        
        if res.success:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_result = res
        
    if best_result is None:
        y_opt = np.zeros(N-1)
    else:
        y_opt = best_result.x
        
    # Reconstruct optimal gaps
    log_gaps = np.empty(N)
    log_gaps[0] = 0.0
    log_gaps[1:] = y_opt
    
    max_log = np.max(log_gaps)
    exp_gaps = np.exp(log_gaps - max_log)
    sums = np.sum(exp_gaps)
    gaps = 2 * np.pi * exp_gaps / sums
    
    # Construct points
    angles = np.zeros(N)
    angles[1:] = np.cumsum(gaps[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points

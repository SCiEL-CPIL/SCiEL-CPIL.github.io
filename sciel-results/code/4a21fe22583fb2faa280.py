import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _autoconv_max_continuous(h):
    """
    Calculate the supremum norm of the continuous autoconvolution of the step function defined by h.
    For step functions on a uniform grid, max_cont = DELTA * max(discrete_convolution).
    """
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    return DELTA * max_conv

def _score_raw(h):
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 == 0:
        return 0.0
    autoconv_max = _autoconv_max_continuous(h)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    def normalized_score(h):
        s = np.sum(h)
        if s == 0: return 0.0
        h_norm = h / s
        return _score_raw(h_norm)

    # Start with uniform
    best_h = np.ones(N)
    best_score = normalized_score(best_h)
    
    # Parametric families for initialization
    x = np.arange(N)
    centers = np.linspace(0, N-1, 15)
    widths = np.linspace(1, N, 15)
    
    for c in centers:
        for w in widths:
            h_gauss = np.exp(-0.5 * ((x - c) / w)**2)
            s = normalized_score(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
            h_tri = np.maximum(0, 1 - np.abs(x - c) / w)
            s = normalized_score(h_tri)
            if s > best_score:
                best_score = s
                best_h = h_tri.copy()
    
    # Use scipy.optimize L-BFGS-B in log-space for smooth optimization
    def objective(y):
        h = np.exp(y)
        return -normalized_score(h)
    
    def objective_grad(y):
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            y_plus = y.copy()
            y_plus[i] += eps
            y_minus = y.copy()
            y_minus[i] -= eps
            grad[i] = (objective(y_plus) - objective(y_minus)) / (2 * eps)
        return grad
    
    # Multiple restarts
    initializations = [np.zeros(N)]  # Uniform
    for c in [N//4, N//2, 3*N//4]:
        for w in [N//4, N//2, N]:
            h_g = np.exp(-0.5 * ((x - c) / w)**2)
            h_g = np.maximum(h_g, 1e-10)
            initializations.append(np.log(h_g))
    
    for y0 in initializations:
        if time.time() > deadline - 15:
            break
        try:
            result = minimize(objective, y0, method='L-BFGS-B',
                             jac=objective_grad,
                             bounds=[(-10, 10)] * N,
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            h_opt = np.exp(result.x)
            s_opt = normalized_score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
        except:
            pass
    
    # Adaptive coordinate descent polish
    if time.time() < deadline - 5:
        step = 0.05
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                h_trial = best_h.copy()
                h_trial[i] += step
                s = normalized_score(h_trial)
                if s > best_score:
                    best_score = s
                    best_h = h_trial
                    improved = True
                else:
                    h_trial = best_h.copy()
                    h_trial[i] -= step
                    if h_trial[i] < 0: h_trial[i] = 0
                    s = normalized_score(h_trial)
                    if s > best_score:
                        best_score = s
                        best_h = h_trial
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
    
    return list(best_h)

# EVOLVE-BLOCK-START
def construct():
    # Re-define inside to keep scope clean if needed, or use globals
    start_time = time.time()
    deadline = 110.0
    
    h = np.ones(N)
    
    def normalized_score(h):
        s = np.sum(h)
        if s == 0: return 0
        h_norm = h / s
        l1 = np.sum(h_norm)
        l2_sq = np.sum(h_norm**2)
        autoconv_max = DELTA * np.max(np.convolve(h_norm, h_norm, mode='full'))
        if autoconv_max == 0: return 0
        return (l2_sq / (l1 * autoconv_max))

    best_h = h.copy()
    best_score = normalized_score(h)
    
    # Parametric search
    centers = np.linspace(0, N-1, 20)
    widths = np.linspace(1, N, 20)
    
    for c in centers:
        for w in widths:
            x = np.arange(N)
            h_gauss = np.exp(-0.5 * ((x - c) / w)**2)
            s = normalized_score(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
            h_tri = np.maximum(0, 1 - np.abs(x - c) / w)
            s = normalized_score(h_tri)
            if s > best_score:
                best_score = s
                best_h = h_tri.copy()
                
    # Coordinate Descent
    step = 0.1
    for _ in range(100): # Max iterations
        if time.time() > deadline: break
        improved = False
        for i in range(N):
            if time.time() > deadline: break
            # Try increase
            h_trial = best_h.copy()
            h_trial[i] += step
            s = normalized_score(h_trial)
            if s > best_score:
                best_score = s
                best_h = h_trial
                improved = True
            else:
                # Try decrease
                h_trial = best_h.copy()
                h_trial[i] -= step
                if h_trial[i] < 0: h_trial[i] = 0
                s = normalized_score(h_trial)
                if s > best_score:
                    best_score = s
                    best_h = h_trial
                    improved = True
        if not improved:
            break
            
    return list(best_h)
# EVOLVE-BLOCK-END

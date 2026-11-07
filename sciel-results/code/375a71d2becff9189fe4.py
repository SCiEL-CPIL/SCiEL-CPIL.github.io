import numpy as np
import time
import scipy.optimize

N = 50

def _value_exact(h):
    h = np.maximum(h, 0)
    if np.all(h == 0):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    y1 = y[:m+1]
    y2 = y[1:m+2]
    l2 = np.sum(y1**2 + y1*y2 + y2**2) * (w / 3.0)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _value_smooth(h, p=10):
    h = np.maximum(h, 0)
    if np.all(h == 0):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    y1 = y[:m+1]
    y2 = y[1:m+2]
    l2 = np.sum(y1**2 + y1*y2 + y2**2) * (w / 3.0)
    norm_1 = np.sum(g) / (m + 1)
    # Power mean
    if np.any(g > 0):
        norm_inf_smooth = (np.sum(g**p))**(1.0/p)
    else:
        norm_inf_smooth = 0.0
    if norm_1 == 0 or norm_inf_smooth == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf_smooth)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.ones(N)
    best_val = _value_exact(best_h)
    
    rng = np.random.default_rng(50)
    
    # Generate diverse initial points
    inits = []
    # Uniform
    inits.append(np.ones(N))
    # Triangle
    t = np.linspace(-1, 1, N)
    inits.append(np.maximum(0, 1 - np.abs(t)))
    # Gaussian
    x = np.linspace(-3, 3, N)
    inits.append(np.exp(-x**2))
    # Random
    for _ in range(10):
        inits.append(rng.random(N))
        
    # Normalize
    for i in range(len(inits)):
        s = np.sum(inits[i])
        if s > 0:
            inits[i] = inits[i] / s
        else:
            inits[i] = np.ones(N) / N
            
    # Optimize using smooth surrogate
    # Use L-BFGS-B with numerical gradients? Or Powell?
    # Powell is robust. Let's try Powell on the smooth function.
    
    for h0 in inits:
        if time.time() > deadline - 5.0:
            break
            
        def obj_smooth(h):
            return -_value_smooth(h, p=10)
            
        try:
            # Powell doesn't need gradients
            result = scipy.optimize.minimize(
                obj_smooth, 
                h0, 
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            h_res = result.x
            # Evaluate exact value
            val = _value_exact(h_res)
            
            if val > best_val:
                best_val = val
                best_h = h_res.copy()
                
        except Exception:
            pass

    # Polish with exact objective using Nelder-Mead
    if time.time() < deadline - 2.0:
        def obj_exact(h):
            return -_value_exact(h)
            
        try:
            result = scipy.optimize.minimize(
                obj_exact, 
                best_h, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            h_polished = result.x
            val_polished = _value_exact(h_polished)
            if val_polished > best_val:
                best_val = val_polished
                best_h = h_polished
        except Exception:
            pass

    return best_h.tolist()

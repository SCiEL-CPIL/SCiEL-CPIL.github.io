import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h, mode='full')
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    y2 = y * y
    cross = y[:-1] * y[1:]
    l2 = (w / 3.0) * np.sum(y2[:-1] + cross + y2[1:])
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _smooth_max(g, beta=50.0):
    """Smooth approximation of max(g) using log-sum-exp."""
    g_max = np.max(g)
    if g_max < 1e-300:
        return 0.0
    shifted = g / g_max
    return g_max * np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta

def _objective_and_grad(h_vec):
    """Objective (negative value) and gradient via finite differences or analytic."""
    h = np.maximum(h_vec, 0.0)
    g = np.convolve(h, h, mode='full')
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    
    y2 = y * y
    cross = y[:-1] * y[1:]
    l2 = (w / 3.0) * np.sum(y2[:-1] + cross + y2[1:])
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 * norm_inf < 1e-300:
        return 0.0, np.zeros_like(h_vec)
    
    value = l2 / (norm_1 * norm_inf)
    return -value, np.zeros_like(h_vec)

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_h = np.ones(N)
    best_value = _compute_value(best_h)
    
    # Multiple random restarts
    np.random.seed(42)
    
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Different initializations
        if restart == 0:
            h0 = np.ones(N)
        elif restart == 1:
            h0 = np.random.exponential(1.0, size=N)
        elif restart == 2:
            h0 = np.random.uniform(0.0, 1.0, size=N)
        elif restart == 3:
            # Triangular
            h0 = np.abs(np.linspace(-1, 1, N))
        elif restart == 4:
            # Gaussian-like
            h0 = np.exp(-0.5 * (np.linspace(-2, 2, N))**2)
        elif restart == 5:
            # Two peaks
            h0 = np.zeros(N)
            h0[:N//4] = 1.0
            h0[3*N//4:] = 1.0
        elif restart == 6:
            # Center peak
            center = N // 2
            h0 = np.exp(-0.5 * (np.arange(N) - center)**2 / 10.0)
        elif restart == 7:
            # Edge peaks
            h0 = np.zeros(N)
            h0[:5] = 1.0
            h0[-5:] = 1.0
        else:
            # Random mix
            r = np.random.random(N)
            h0 = np.where(r < 0.3, np.random.exponential(0.5, size=N), np.random.uniform(0, 1, size=N))
        
        # Normalize to avoid extreme scales
        h0 = h0 / (np.sum(h0) + 1e-10)
        
        # Use L-BFGS-B with smooth objective
        # We'll use a penalty approach: fix norm_1 by scaling, optimize shape
        
        # Actually, let's directly optimize with a smoothed objective
        # Use log-sum-exp for max with adaptive beta
        
        def obj_and_grad(x):
            h = np.maximum(x, 0.0)
            g = np.convolve(h, h, mode='full')
            m = len(g)
            y = np.concatenate(([0.0], g, [0.0]))
            w = 1.0 / (m + 1)
            
            y2 = y * y
            cross = y[:-1] * y[1:]
            l2 = (w / 3.0) * np.sum(y2[:-1] + cross + y2[1:])
            
            norm_1 = np.sum(g) / (m + 1)
            
            # Smooth max with log-sum-exp
            beta = 100.0
            g_max = np.max(g)
            if g_max < 1e-300:
                return 0.0, np.zeros_like(x)
            smooth_max = g_max * np.log(np.sum(np.exp(beta * (g / g_max - 1.0)))) / beta
            
            if norm_1 * smooth_max < 1e-300:
                return 0.0, np.zeros_like(x)
            
            value = l2 / (norm_1 * smooth_max)
            return -value, np.zeros_like(x)
        
        try:
            result = minimize(
                obj_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = np.maximum(result.x, 0.0)
            val = _compute_value(h_opt)
            
            if val > best_value:
                best_value = val
                best_h = h_opt.copy()
        except Exception:
            pass
        
        # Also try with different beta values
        for beta in [50.0, 200.0, 500.0]:
            if time.time() > deadline - 5.0:
                break
            
            def obj_smooth(x, _beta=beta):
                h = np.maximum(x, 0.0)
                g = np.convolve(h, h, mode='full')
                m = len(g)
                y = np.concatenate(([0.0], g, [0.0]))
                w = 1.0 / (m + 1)
                
                y2 = y * y
                cross = y[:-1] * y[1:]
                l2 = (w / 3.0) * np.sum(y2[:-1] + cross + y2[1:])
                
                norm_1 = np.sum(g) / (m + 1)
                g_max = np.max(g)
                if g_max < 1e-300:
                    return 0.0, np.zeros_like(x)
                smooth_max = g_max * np.log(np.sum(np.exp(_beta * (g / g_max - 1.0)))) / _beta
                
                if norm_1 * smooth_max < 1e-300:
                    return 0.0, np.zeros_like(x)
                
                value = l2 / (norm_1 * smooth_max)
                return -value, np.zeros_like(x)
            
            try:
                result = minimize(
                    obj_smooth,
                    h0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, None)] * N,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_opt = np.maximum(result.x, 0.0)
                val = _compute_value(h_opt)
                if val > best_value:
                    best_value = val
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    return list(best_h)

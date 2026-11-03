"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 130.0  # safety margin before 150s kill

def _score(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    c = np.convolve(h, h, mode='full')
    cmax = np.max(c)
    if cmax <= 0:
        return 0.0
    return l2_sq / (l1 * cmax)

def _objective_and_grad(x, t_idx):
    """Objective: t - ratio, gradient w.r.t. x and t.
    x = [h (N vars), t (1 var)]
    We minimize t - ratio(h) where ratio = sum(h^2)/(sum(h)*max(c))
    
    Actually, let's use epigraph: minimize t s.t. sum(h^2) >= t * sum(h) * c_max
    But c_max is nonsmooth. Instead, minimize t subject to sum(h^2) - t*sum(h)*c[k] >= 0 for all k.
    That's too many constraints. 
    
    Alternative: directly minimize -ratio(h) using numerical or analytic grad.
    Let's use analytic gradient of ratio directly.
    """
    h = x[:N]
    t = x[N]
    
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    c = np.convolve(h, h, mode='full')
    cmax = c[t_idx]  # use active max index
    
    if l1 <= 1e-12 or cmax <= 1e-12:
        return 1e10, np.zeros_like(x)
    
    ratio = l2_sq / (l1 * cmax)
    
    # d(ratio)/d(h_i)
    # ratio = l2_sq / (l1 * cmax)
    # d(l2_sq)/dh_i = 2*h_i
    # d(l1)/dh_i = 1
    # d(cmax)/dh_i = c[i] + c[N-1-i]  (since c[k] = sum_j h[j]*h[k-j], dc[k]/dh[i] = h[k-i] + h[i] for valid indices... 
    # Actually c[k] = sum_{j=0}^{N-1} h[j]*h[k-j] where h[j]=0 outside [0,N-1]
    # dc[k]/dh[i] = h[k-i] + h[i] if 0<=k-i<N else just one term
    # More carefully: c[k] = sum_j h[j]*h[k-j], so dc[k]/dh[i] = h[k-i] + h[i] (with h out of range = 0)
    
    dh_l1 = np.ones(N)
    dh_l2 = 2.0 * h
    
    # d(c[k])/dh[i] for the active k = t_idx
    k = t_idx
    dc_dh = np.zeros(N)
    for i in range(N):
        val = 0.0
        if 0 <= k - i < N:
            val += h[k - i]
        if 0 <= i < N:
            val += h[i]
        dc_dh[i] = val
    
    # d(ratio)/dh_i = [dh_l2_i * (l1*cmax) - l2_sq * (dh_l1_i * cmax + l1 * dc_dh_i)] / (l1*cmax)^2
    denom = (l1 * cmax)**2
    grad_ratio = (dh_l2 * (l1 * cmax) - l2_sq * (dh_l1 * cmax + l1 * dc_dh)) / denom
    
    # We want to minimize -ratio, so gradient of objective = -grad_ratio
    obj = -ratio
    grad = np.zeros_like(x)
    grad[:N] = -grad_ratio
    grad[N] = 1.0  # d(t)/dt = 1, but t is separate... 
    
    # Wait, I'm confusing myself. Let me restart the formulation.
    # 
    # Let's just directly maximize ratio(h) = l2_sq / (l1 * cmax)
    # Minimize -ratio(h)
    # 
    # x = h (N variables)
    # No need for t.
    
    return obj, grad[:N]

def _objective_and_grad_v2(h):
    """Minimize -ratio(h) where ratio = sum(h^2)/(sum(h)*max(conv(h,h)))"""
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    c = np.convolve(h, h, mode='full')
    cmax = np.max(c)
    t_idx = np.argmax(c)
    
    if l1 <= 1e-12 or cmax <= 1e-12:
        return 1e10, np.zeros_like(h)
    
    ratio = l2_sq / (l1 * cmax)
    
    dh_l1 = np.ones(N)
    dh_l2 = 2.0 * h
    
    k = t_idx
    dc_dh = np.zeros(N)
    for i in range(N):
        val = 0.0
        if 0 <= k - i < N:
            val += h[k - i]
        val += h[i]
        dc_dh[i] = val
    
    denom = (l1 * cmax)**2
    grad_ratio = (dh_l2 * (l1 * cmax) - l2_sq * (dh_l1 * cmax + l1 * dc_dh)) / denom
    
    obj = -ratio
    grad = -grad_ratio
    
    return obj, grad

def construct():
    start_time = time.time()
    
    best_h = None
    best_score = 0.0
    
    # Try several random initializations
    rng = np.random.RandomState(42)
    
    # Initial guess: uniform
    h0 = np.ones(N)
    s0 = _score(h0)
    best_h = h0.copy()
    best_score = s0
    
    for trial in range(20):
        if time.time() - start_time > _DEADLINE:
            break
        
        # Random initialization
        if trial == 0:
            h_init = np.ones(N)
        else:
            # Various random starts
            style = trial % 5
            if style == 0:
                h_init = rng.rand(N) + 0.1
            elif style == 1:
                h_init = np.ones(N) * (1.0 + 0.5 * rng.randn())
                h_init = np.maximum(h_init, 0.1)
            elif style == 2:
                h_init = np.exp(rng.randn(N))
            elif style == 3:
                # Peak in middle
                h_init = np.exp(-((np.arange(N) - N/2)**2) / (N/4))
                h_init += 0.1 * rng.rand(N)
            else:
                h_init = rng.rand(N) ** 2 + 0.1
        
        h_init = np.maximum(h_init, 0.0)
        
        # SLSQP optimization
        try:
            res = minimize(
                _objective_and_grad_v2,
                h_init,
                method='SLSQP',
                jac=True,
                bounds=[(0.0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or res.fun < -best_score:
                h_opt = np.maximum(res.x, 0.0)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    if time.time() - start_time > _DEADLINE - 5:
        pass
    
    # Final check
    if best_h is None:
        return [1.0] * N
    
    return [float(max(v, 0.0)) for v in best_h]

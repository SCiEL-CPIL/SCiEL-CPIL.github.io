import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_peak(h):
    """Compute the peak of the linear autoconvolution of h."""
    conv = np.convolve(h, h, mode='full')
    return np.max(np.abs(conv))

def _logsumexp(x, axis=None):
    """Numerically stable log-sum-exp."""
    m = np.max(x, axis=axis, keepdims=True)
    return m.squeeze(axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)).squeeze(axis=axis)

def _smooth_peak(h, tau=1.0):
    """Smooth approximation of peak |conv| using log-sum-exp."""
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.abs(conv)
    # log-sum-exp of abs_conv / tau, scaled back
    m = np.max(abs_conv)
    if m == 0:
        return 0.0
    lse = tau * (np.log(np.sum(np.exp(abs_conv / tau))) - np.log(len(abs_conv)))
    return lse

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_score = _autoconv_peak(best_h)
    
    def time_ok():
        return time.time() < deadline
    
    # Phase 1: Continuous relaxation optimization
    # Use tanh parameterization: h = tanh(x) where x is continuous
    # Objective: minimize smooth peak of autoconvolution
    
    def objective_and_grad(x):
        h = np.tanh(x)
        conv = np.convolve(h, h, mode='full')
        abs_conv = np.abs(conv)
        tau = 5.0
        # log-sum-exp for smooth max
        m = np.max(abs_conv)
        if m < 1e-12:
            return 0.0, np.zeros_like(x)
        lse = tau * (np.log(np.sum(np.exp(abs_conv / tau))) - np.log(len(abs_conv)))
        
        # Gradient via finite differences is too slow; use analytic-ish approach
        # For now, use a simpler gradient approximation
        # Actually, let's just use the objective and let L-BFGS-B approximate gradients
        return lse
    
    # Try multiple initializations
    np.random.seed(42)
    inits = []
    
    # All zeros (maps to all zeros in tanh, but we want ±1... use small random)
    inits.append(np.random.randn(N) * 0.1)
    # Alternating
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    inits.append(alt * 2.0)  # large values to push tanh to ±1
    # Block patterns
    block_len = 10
    blocks = np.random.choice([-1, 1], size=N // block_len + 1)
    block_init = np.repeat(blocks, block_len)[:N] * 2.0
    inits.append(block_init)
    # Random ±1
    rand_init = np.random.choice([-1, 1], size=N) * 2.0
    inits.append(rand_init)
    
    best_continuous = None
    best_cont_score = np.inf
    
    for init in inits:
        if not time_ok():
            break
        
        try:
            result = minimize(
                objective_and_grad,
                init,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if not time_ok():
                break
            
            h_cont = np.tanh(result.x)
            # Threshold to ±1
            h_thresh = np.sign(h_cont)
            h_thresh[h_thresh == 0] = 1
            
            score = _autoconv_peak(h_thresh)
            if score < best_cont_score:
                best_cont_score = score
                best_continuous = h_thresh.copy()
                
            # Also try the continuous version's peak for tracking
            if result.fun < best_cont_score:
                # The continuous score might be lower but we need discrete
                pass
                
        except Exception:
            continue
    
    if best_continuous is not None and best_cont_score < best_score:
        best_h = best_continuous
        best_score = best_cont_score
    
    # Phase 2: Discrete hill-climbing with sign flips
    if time_ok():
        h = best_h.copy()
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 200
        
        # Use random restarts for hill climbing
        np.random.seed(123)
        
        for restart in range(10):
            if not time_ok():
                break
            if no_improve_count >= max_no_improve:
                break
            
            h = best_h.copy()
            current_score = best_score
            no_improve_count = 0
            
            # Greedy sign flips
            while time_ok() and no_improve_count < max_no_improve:
                # Try random flips
                improved = False
                for _ in range(200):  # 200 random flips per iteration
                    if not time_ok():
                        break
                    
                    idx = np.random.randint(N)
                    old_val = h[idx]
                    h[idx] = -old_val
                    
                    new_score = _autoconv_peak(h)
                    if new_score < current_score:
                        current_score = new_score
                        no_improve_count = 0
                        improved = True
                    else:
                        h[idx] = old_val
                        no_improve_count += 1
                    
                    if not time_ok():
                        break
                
                if not improved and no_improve_count >= max_no_improve:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
                no_improve_count = 0
            else:
                no_improve_count += 100  # Penalize failed restarts
    
    # Phase 3: Final polish - try all single flips
    if time_ok():
        for idx in range(N):
            if not time_ok():
                break
            old_val = best_h[idx]
            best_h[idx] = -old_val
            new_score = _autoconv_peak(best_h)
            if new_score < best_score:
                best_score = new_score
            else:
                best_h[idx] = old_val
    
    return best_h

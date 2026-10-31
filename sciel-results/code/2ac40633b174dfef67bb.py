import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = None
    best_score = float('inf')
    
    # Try various periods
    periods = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 100]
    
    # We will optimize the period values p (length P)
    # The full h is formed by repeating p to length N
    # Note: If N is not divisible by P, the last part is truncated.
    # The sum constraint applies to the full h.
    
    for P in periods:
        if time.time() > deadline - 5.0:
            break
            
        # Initialize the period
        # Try a few initializations: constant, sinusoidal, random
        inits = []
        # Constant
        inits.append(np.full(P, 0.5))
        # Sinusoidal
        t = np.linspace(0, 2*np.pi, P, endpoint=False)
        inits.append(0.5 + 0.4 * np.sin(t))
        # Random
        np.random.seed(42 + P)
        inits.append(np.random.rand(P))
        
        for h0_period in inits:
            if time.time() > deadline - 5.0:
                break
            
            def objective(params):
                # params is the period
                p = params
                # Construct full h
                # Repeat p to cover N
                # h_full = np.tile(p, int(np.ceil(N/P)))[:N]
                # But we need to be careful with the sum.
                # Let's just compute the overlap on the full array
                # However, L-BFGS-B gradient might be expensive if we rebuild every time.
                # But N=100 is small.
                
                reps = int(np.ceil(N / P))
                h_full = np.tile(p, reps)[:N]
                
                # Penalty for sum
                s = np.sum(h_full)
                penalty = 100.0 * (s - N/2.0)**2
                
                # Overlap
                shifted = np.zeros(N)
                shifted[:-1] = h_full[1:]
                overlap = h_full * (1.0 - shifted)
                
                # Smooth max
                smax = _smooth_max(overlap, alpha=50.0)
                
                return smax + penalty
            
            # Bounds for period
            bounds = [(0.0, 1.0)] * P
            
            try:
                result = minimize(
                    objective,
                    h0_period,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
                )
                
                if result.success or result.fun < best_score:
                    p_opt = result.x
                    # Construct full h
                    reps = int(np.ceil(N / P))
                    h_full = np.tile(p_opt, reps)[:N]
                    
                    # Enforce sum precisely
                    s = np.sum(h_full)
                    diff = N/2.0 - s
                    if abs(diff) > 1e-12:
                        h_full = h_full + diff / N
                        h_full = np.clip(h_full, 0.0, 1.0)
                        # Re-normalize
                        for _ in range(5):
                            s = np.sum(h_full)
                            diff = N/2.0 - s
                            if abs(diff) > 1e-12:
                                h_full = h_full + diff / N
                                h_full = np.clip(h_full, 0.0, 1.0)
                    
                    # Check validity
                    if np.all(h_full >= -1e-9) and np.all(h_full <= 1.0 + 1e-9):
                        h_full = np.clip(h_full, 0.0, 1.0)
                        current_score = _score(h_full)
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_h = h_full.copy()
                            
            except Exception:
                continue
                
    # Fallback: if no periodic solution found, use the 2-block solution
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h0 = np.zeros(N)
        half = N // 2
        h0[:half] = A
        h0[half:] = B
        best_h = h0
        best_score = _score(h0)
        
    # Final polish on the best solution found using full N variables with SLSQP or L-BFGS-B
    # Since we have a good starting point, a local polish might help
    if time.time() < deadline - 10.0:
        def objective_full(params):
            h_full = params
            s = np.sum(h_full)
            penalty = 100.0 * (s - N/2.0)**2
            shifted = np.zeros(N)
            shifted[:-1] = h_full[1:]
            overlap = h_full * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=100.0)
            return smax + penalty
            
        bounds_full = [(0.0, 1.0)] * N
        try:
            result_full = minimize(
                objective_full,
                best_h,
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if result_full.fun < best_score:
                h_opt = result_full.x
                s = np.sum(h_opt)
                diff = N/2.0 - s
                if abs(diff) > 1e-12:
                    h_opt = h_opt + diff / N
                    h_opt = np.clip(h_opt, 0.0, 1.0)
                    for _ in range(5):
                        s = np.sum(h_opt)
                        diff = N/2.0 - s
                        if abs(diff) > 1e-12:
                            h_opt = h_opt + diff / N
                            h_opt = np.clip(h_opt, 0.0, 1.0)
                
                if np.all(h_opt >= -1e-9) and np.all(h_opt <= 1.0 + 1e-9):
                    h_opt = np.clip(h_opt, 0.0, 1.0)
                    new_score = _score(h_opt)
                    if new_score < best_score:
                        best_h = h_opt
                        best_score = new_score
        except Exception:
            pass
            
    return best_h

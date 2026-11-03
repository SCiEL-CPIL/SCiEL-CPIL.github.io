import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 145.0  # 5 seconds safety margin

def construct():
    start_time = time.time()
    
    # Initial guess: alternating pattern to reduce overlap
    h0 = np.zeros(N)
    h0[::2] = 1.0
    h0[1::2] = 0.0
    # Scale to satisfy sum constraint: currently sum = N/2, which is exactly what we need
    # But this is binary; let's start from a smoothed version
    h0 = 0.5 + 0.49 * np.cos(2 * np.pi * np.arange(N) / N * 2)  # frequency 2 pattern
    h0 = np.clip(h0, 0, 1)
    # Rescale to satisfy sum = N/2
    current_sum = np.sum(h0)
    if current_sum > 0:
        h0 = h0 * (N / 2) / current_sum
        h0 = np.clip(h0, 0, 1)
        # After clipping, re-normalize
        current_sum = np.sum(h0)
        if current_sum > 0:
            h0 = h0 * (N / 2) / current_sum
    
    best_h = h0.copy()
    best_score = np.inf
    
    def overlap_score(h):
        """Compute max overlap of h with shifted 1-h"""
        h = np.clip(h, 0, 1)
        max_overlap = 0.0
        for k in range(1, N):
            # Overlap at shift k: sum_i h_i * (1 - h_{i+k}) with wrap-around
            shifted = np.roll(h, -k)
            overlap = np.sum(h * (1 - shifted))
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap
    
    current_score = overlap_score(h0)
    if current_score < best_score:
        best_score = current_score
        best_h = h0.copy()
    
    # Use SLSQP with bounds and equality constraint
    bounds = [(0, 1)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    
    # Try multiple random restarts within time budget
    np.random.seed(42)
    for restart in range(20):
        if time.time() > DEADLINE:
            break
        
        # Generate random initial point
        h_init = np.random.uniform(0, 1, N)
        # Scale to satisfy sum constraint
        s = np.sum(h_init)
        if s > 0:
            h_init = h_init * (N/2) / s
            h_init = np.clip(h_init, 0, 1)
            s = np.sum(h_init)
            if s > 0:
                h_init = h_init * (N/2) / s
        
        # Also try structured patterns
        if restart % 5 == 0:
            freq = (restart // 5) * 0.7 + 1.0
            h_init = 0.5 + 0.49 * np.cos(2 * np.pi * freq * np.arange(N) / N + np.random.uniform(0, 2*np.pi))
            h_init = np.clip(h_init, 0, 1)
            s = np.sum(h_init)
            if s > 0:
                h_init = h_init * (N/2) / s
                h_init = np.clip(h_init, 0, 1)
                s = np.sum(h_init)
                if s > 0:
                    h_init = h_init * (N/2) / s
        
        try:
            result = minimize(
                overlap_score,
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if result.success:
                h_opt = np.clip(result.x, 0, 1)
                # Ensure sum constraint
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt * (N/2) / s
                    h_opt = np.clip(h_opt, 0, 1)
                
                score = overlap_score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final check
    final_h = np.clip(best_h, 0, 1)
    s = np.sum(final_h)
    if s > 0:
        final_h = final_h * (N/2) / s
        final_h = np.clip(final_h, 0, 1)
    
    return final_h

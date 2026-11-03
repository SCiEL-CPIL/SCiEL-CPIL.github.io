import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    dw = 2.0 / N
    target = N / 2.0
    
    def compute_overlaps_vectorized(h):
        """Vectorized computation of all overlaps using broadcasting."""
        idx = np.arange(N)
        M = h[(idx[:, None] + idx[None, :]) % N]  # M[i,k] = h[(i+k)%N]
        mins = np.minimum(h[:, None], 1.0 - M)
        overlaps = np.sum(mins, axis=0) * dw
        return overlaps
    
    def project_to_feasible(h):
        """Project h to satisfy sum constraint and [0,1] bounds."""
        h = np.clip(h, 0, 1)
        diff = target - h.sum()
        if abs(diff) > 1e-10:
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
                h = np.clip(h, 0, 1)
        return h
    
    def objective_smooth(x, beta=100.0):
        """Smooth soft-max objective for better gradient-based optimization."""
        h = project_to_feasible(x)
        overlaps = compute_overlaps_vectorized(h)
        # Soft max: (1/beta) * log(sum(exp(beta * overlaps)))
        m = overlaps.max()
        return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
    
    def objective(x):
        """Objective function for optimization: max overlap."""
        h = project_to_feasible(x)
        overlaps = compute_overlaps_vectorized(h)
        return overlaps.max()
    
    # Start with uniform solution
    best_h = np.full(N, 0.5)
    best_score = compute_overlaps_vectorized(best_h).max()
    
    # Try various structured initializations
    rng = np.random.default_rng(123)
    
    # Try random binary-like patterns
    for trial in range(20):
        if time.time() > DEADLINE - 60:
            break
        # Create a pattern with some structure
        h = rng.uniform(0, 1, N)
        h = project_to_feasible(h)
        score = compute_overlaps_vectorized(h).max()
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try patterns based on number-theoretic constructions
    # Use a prime-based pattern for good distribution
    for p in [97, 89, 83, 79, 73]:
        if time.time() > DEADLINE - 50:
            break
        if p > N:
            continue
        # Create a pattern based on quadratic residues
        h = np.zeros(N)
        for i in range(N):
            # Use a hash-like function based on prime
            h[i] = 0.5 + 0.3 * np.sin(2 * np.pi * i * p / N)
        h = project_to_feasible(h)
        score = compute_overlaps_vectorized(h).max()
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # L-BFGS-B optimization with smooth objective
    if time.time() < DEADLINE - 40:
        x0 = best_h.copy()
        result = minimize(objective_smooth, x0, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        h_opt = project_to_feasible(result.x)
        score = compute_overlaps_vectorized(h_opt).max()
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Nelder-Mead on the raw max objective for fine-tuning
    if time.time() < DEADLINE - 20:
        x0 = best_h.copy()
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
        h_opt = project_to_feasible(result.x)
        score = compute_overlaps_vectorized(h_opt).max()
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Coordinate descent with binary search for optimal values
    if time.time() < DEADLINE - 10:
        for iteration in range(3):
            if time.time() > DEADLINE - 5:
                break
            h_trial = best_h.copy()
            for i in range(N):
                if time.time() > DEADLINE - 5:
                    break
                # Find optimal h[i] via ternary search on [0,1]
                # The objective as a function of h[i] is piecewise linear, so 
                # we can find the minimum by checking critical points
                
                # Get current overlaps
                h_base = h_trial.copy()
                h_base[i] = 0.5  # temporary
                h_base = project_to_feasible(h_base)
                
                # Try a grid of values
                best_val = h_trial[i]
                best_local_score = compute_overlaps_vectorized(h_trial).max()
                
                for val in np.linspace(0, 1, 21):
                    h_test = h_trial.copy()
                    h_test[i] = val
                    h_test = project_to_feasible(h_test)
                    score = compute_overlaps_vectorized(h_test).max()
                    if score < best_local_score - 1e-12:
                        best_local_score = score
                        best_val = val
                
                if abs(best_val - h_trial[i]) > 1e-6:
                    h_trial[i] = best_val
            
            h_trial = project_to_feasible(h_trial)
            score = compute_overlaps_vectorized(h_trial).max()
            if score < best_score - 1e-12:
                best_score = score
                best_h = h_trial.copy()
    
    # Final polish with high-precision Nelder-Mead
    if time.time() < DEADLINE - 3:
        x0 = best_h.copy()
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True})
        h_opt = project_to_feasible(result.x)
        score = compute_overlaps_vectorized(h_opt).max()
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Final projection to ensure exact sum
    best_h = project_to_feasible(best_h)
    
    return best_h

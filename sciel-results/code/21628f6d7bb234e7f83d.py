import numpy as np
import time
from scipy.optimize import minimize

N = 100

def calculate_max_overlap(h):
    """Calculate maximum linear overlap of h with shifted 1-h."""
    max_ov = 0.0
    for k in range(N):
        if k == 0:
            ov = np.sum(h * (1 - h))
        else:
            ov = np.sum(h[k:] * (1 - h[:-k]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def soft_max_overlaps(h, beta=20.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    overlaps = np.zeros(N)
    for k in range(N):
        if k == 0:
            overlaps[k] = np.sum(h * (1 - h))
        else:
            overlaps[k] = np.sum(h[k:] * (1 - h[:-k]))
    # Log-sum-exp for numerical stability
    m = np.max(overlaps)
    if m < 1e-10:
        return 0.0
    return m + np.log(np.sum(np.exp((overlaps - m) * beta))) / beta

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Base solution
    h_base = np.full(N, 0.5)
    base_score = calculate_max_overlap(h_base)
    
    best_h = h_base.copy()
    best_score = base_score
    
    def make_h_from_params(params, sigma, amp, center, phase):
        """Generate h from parameters with Gaussian perturbation."""
        x = np.arange(N)
        eps = np.exp(-0.5 * ((x - center) / sigma) ** 2) * np.cos(2 * np.pi * phase * x / N)
        eps -= np.mean(eps)
        eps_max = np.max(np.abs(eps))
        if eps_max < 1e-10:
            eps_max = 1.0
        eps = eps / eps_max
        h_candidate = 0.5 + amp * eps
        return h_candidate

    def correct_sum(h_candidate):
        """Correct the sum to be exactly N/2 while maintaining bounds."""
        diff = N / 2.0 - np.sum(h_candidate)
        if abs(diff) < 1e-12:
            return h_candidate
        # Distribute correction across all elements proportionally to distance from 0.5
        # This maintains smoothness better than single-point adjustment
        weights = np.abs(h_candidate - 0.5) + 1e-6
        weights /= np.sum(weights)
        h_candidate = h_candidate + diff * weights
        # Clip and re-correct
        h_candidate = np.clip(h_candidate, 0, 1)
        # Final adjustment to ensure exact sum
        residual = N / 2.0 - np.sum(h_candidate)
        if abs(residual) > 1e-12:
            # Add to the element closest to 0.5
            idx = np.argmin(np.abs(h_candidate - 0.5))
            h_candidate[idx] += residual
            h_candidate = np.clip(h_candidate, 0, 1)
        return h_candidate

    def objective(params):
        """Objective function for optimization."""
        sigma, amp, center, phase = params
        if sigma < 1.0 or sigma > 60.0:
            return 100.0
        if amp < 0.0 or amp > 0.49:
            return 100.0
        if center < 0.0 or center > N:
            return 100.0
            
        h_candidate = make_h_from_params(params, sigma, amp, center, phase)
        h_candidate = correct_sum(h_candidate)
        
        if np.min(h_candidate) < 0 or np.max(h_candidate) > 1:
            return 100.0
            
        # Use soft max for smoothness during optimization
        return soft_max_overlaps(h_candidate, beta=20.0)

    # Grid search for initial good parameters
    best_params = None
    min_obj = 100.0
    
    sigmas = np.linspace(3, 50, 10)
    amps = np.linspace(0.0, 0.45, 6)
    centers = np.linspace(N*0.2, N*0.8, 5)
    phases = np.linspace(0, 1, 4)
    
    for s in sigmas:
        for a in amps:
            for c in centers:
                for p in phases:
                    if time.time() - start_time > deadline - 20:
                        break
                    val = objective([s, a, c, p])
                    if val < min_obj:
                        min_obj = val
                        best_params = [s, a, c, p]
                if time.time() - start_time > deadline - 20:
                    break
            if time.time() - start_time > deadline - 20:
                break
        if time.time() - start_time > deadline - 20:
            break
    
    # Local optimization using scipy
    if best_params is not None:
        bounds = [(1.0, 60.0), (0.0, 0.49), (0.0, float(N)), (0.0, 1.0)]
        try:
            res = minimize(objective, best_params, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success and res.fun < min_obj:
                min_obj = res.fun
                best_params = res.x
        except Exception:
            pass
        
        # Second pass with tighter tolerance
        if best_params is not None and time.time() - start_time < deadline - 10:
            try:
                res2 = minimize(objective, best_params, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
                if res2.success and res2.fun < min_obj:
                    min_obj = res2.fun
                    best_params = res2.x
            except Exception:
                pass
    
    # Construct final best h
    if best_params is not None:
        sigma, amp, center, phase = best_params
        h_candidate = make_h_from_params(best_params, sigma, amp, center, phase)
        h_candidate = correct_sum(h_candidate)
        if np.min(h_candidate) >= 0 and np.max(h_candidate) <= 1:
            final_score = calculate_max_overlap(h_candidate)
            if final_score < best_score:
                best_h = h_candidate
                best_score = final_score
    
    # Final polish: try small perturbations around the best solution
    if time.time() - start_time < deadline - 5:
        for trial in range(20):
            if time.time() - start_time > deadline - 2:
                break
            noise = np.random.normal(0, 0.01, N)
            h_trial = best_h + noise
            h_trial = np.clip(h_trial, 0, 1)
            h_trial = correct_sum(h_trial)
            if np.min(h_trial) >= 0 and np.max(h_trial) <= 1:
                trial_score = calculate_max_overlap(h_trial)
                if trial_score < best_score:
                    best_h = h_trial
                    best_score = trial_score
    
    return best_h

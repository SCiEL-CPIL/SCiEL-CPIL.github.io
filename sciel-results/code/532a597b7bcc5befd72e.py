import numpy as np
from scipy import optimize

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = np.convolve(h, h, 'full')
    total = sum(h)
    if total == 0:
        return 0.0
    return max(conv) / (total * total)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    # Initial guess: all ones (this gives score = 1.0)
    h = np.ones(N)
    current_score = compute_autoconv_max(h)
    best_h = h.copy()
    best_score = current_score
    
    # Soft-max objective for smooth optimization
    def soft_objective(x, T=0.05):
        h = np.abs(x)
        conv = np.convolve(h, h, 'full')
        total = sum(h)
        if total < 1e-10:
            return 1e10
        conv_max = max(conv)
        # Use log-sum-exp for numerical stability
        shifted = (conv - conv_max) / T
        # Clip to avoid overflow
        shifted = np.clip(shifted, -50, 50)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)
    
    # Exact objective for final evaluation
    def exact_objective(x):
        h = np.abs(x)
        return compute_autoconv_max(h)
    
    # Try multiple strategies with different initial conditions
    np.random.seed(42)
    
    # Strategy 1: Start from uniform, perturb slightly
    for restart in range(8):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            # Uniform
            h0 = np.ones(N)
        elif restart == 1:
            # Slight cosine perturbation
            h0 = 1.0 + 0.1 * np.cos(2 * np.pi * np.linspace(0, 1, N, endpoint=False))
            h0 = np.maximum(h0, 0.01)
        elif restart == 2:
            # Slight sine perturbation
            h0 = 1.0 + 0.1 * np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False))
            h0 = np.maximum(h0, 0.01)
        elif restart == 3:
            # Two-frequency perturbation
            t = np.linspace(0, 1, N, endpoint=False)
            h0 = 1.0 + 0.05 * np.cos(2 * np.pi * t) + 0.05 * np.cos(4 * np.pi * t)
            h0 = np.maximum(h0, 0.01)
        elif restart == 4:
            # Random small perturbation
            h0 = np.ones(N) + 0.05 * np.random.randn(N)
            h0 = np.maximum(h0, 0.01)
        elif restart == 5:
            # Edge-enhanced
            h0 = 1.0 + 0.2 * np.abs(np.linspace(-1, 1, N))
            h0 = np.maximum(h0, 0.01)
        elif restart == 6:
            # Center-enhanced
            t = np.linspace(-1, 1, N)
            h0 = 1.0 + 0.2 * np.exp(-t**2 * 5)
            h0 = np.maximum(h0, 0.01)
        else:
            # Random with different seed
            np.random.seed(restart * 100 + 42)
            h0 = np.random.rand(N) + 0.1
            # Normalize to have similar scale
            h0 = h0 / np.mean(h0)
        
        h0 = np.abs(h0)
        
        # Stage 1: Coarse optimization with soft-max
        try:
            result1 = optimize.minimize(
                soft_objective, h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 15:
                break
            
            # Stage 2: Fine optimization from the result of stage 1
            if result1.fun < best_score * 1.01:  # Only proceed if close
                result2 = optimize.minimize(
                    soft_objective, result1.x,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                # Evaluate both stage 1 and stage 2 with exact objective
                for candidate in [result1.x, result2.x]:
                    h_new = np.abs(candidate)
                    actual_score = compute_autoconv_max(h_new)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_new.copy()
                        
        except Exception:
            pass
    
    # Final polish: try to improve the best solution found
    if time.time() < deadline - 10:
        try:
            # Try a few local perturbations of the best solution
            for _ in range(5):
                h_perturbed = best_h + 0.01 * np.random.randn(N)
                h_perturbed = np.maximum(h_perturbed, 0)
                
                result = optimize.minimize(
                    soft_objective, h_perturbed,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_new = np.abs(result.x)
                actual_score = compute_autoconv_max(h_new)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_new.copy()
                    
        except Exception:
            pass
    
    return list(best_h)

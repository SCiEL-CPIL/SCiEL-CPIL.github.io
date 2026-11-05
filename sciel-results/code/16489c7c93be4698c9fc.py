import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

def _evaluate(h):
    """
    Computes the objective value exactly as defined in the task.
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    """
    if not np.all(np.isfinite(h)):
        return np.inf
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    # Full autoconvolution
    # np.convolve(h, h) has length 2N-1
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (S * S))

def _parametric_h(params):
    """
    Constructs a signal h of length N from parameters.
    params: [amp, freq1, phase1, freq2, phase2, tilt, offset]
    Model: h[i] = amp * (cos(2*pi*freq1*i/N + phase1) + cos(2*pi*freq2*i/N + phase2)) 
           + tilt * (i - (N-1)/2.0) / (N/2.0) + offset
    
    We use two sinusoidal modes to allow for more complex interference patterns 
    than a single chirp, which might better suppress sidelobes.
    """
    amp = params[0]
    freq1 = params[1]
    phase1 = params[2]
    freq2 = params[3]
    phase2 = params[4]
    tilt = params[5]
    offset = params[6]
    
    i = np.arange(N)
    # Normalize index to [-1, 1] for the tilt to keep scale consistent
    x = 2.0 * i / (N - 1) - 1.0
    
    h = amp * (np.cos(2 * np.pi * freq1 * i / N + phase1) + 
               np.cos(2 * np.pi * freq2 * i / N + phase2))
    h += tilt * x
    h += offset
    
    return h

def _objective(params):
    h = _parametric_h(params)
    return _evaluate(h)

def _optimize_with_lbgfgs(start_params, deadline):
    """
    Optimizes the parametric family using L-BFGS-B.
    """
    # Define bounds for parameters to keep them in a reasonable range
    # amp: [0.1, 10.0]
    # freq: [0.1, 20.0]
    # phase: [0, 2*pi]
    # tilt: [-10.0, 10.0]
    # offset: [-10.0, 10.0]
    bounds = [
        (0.1, 10.0),   # amp
        (0.1, 20.0),   # freq1
        (0.0, 2 * np.pi), # phase1
        (0.1, 20.0),   # freq2
        (0.0, 2 * np.pi), # phase2
        (-10.0, 10.0), # tilt
        (-10.0, 10.0)  # offset
    ]
    
    # Initial guess: Try a few different starting points
    # We will do a single run here, but the caller can try multiple starts.
    
    try:
        res = minimize(
            _objective, 
            start_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x, res.fun
    except Exception:
        return start_params, _objective(start_params)

def _refine_greedy(h, max_iter=500, deadline=140.0):
    """
    Greedy local search on the raw h vector.
    Tries small perturbations to reduce the objective.
    """
    best_h = h.copy()
    best_val = _evaluate(best_h)
    
    # Determine step size based on signal magnitude
    scale = np.std(best_h)
    if scale < 1e-6:
        scale = 1.0
    step = 0.1 * scale
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Randomly pick a few indices to perturb
        # To be efficient, we don't iterate all 400 every time, but a subset
        # Or we can do a coordinate descent style. 
        # Given N=400, full coordinate descent might be slow if we evaluate from scratch each time.
        # np.convolve is O(N^2) = 160,000 ops. 400 evals = 64M ops. Very fast.
        # Let's do a simple random hill climbing with restarts of step size.
        
        # Actually, let's try a more structured refinement:
        # Randomly select 10 indices, perturb them, keep if better.
        indices = np.random.choice(N, size=10, replace=False)
        
        for idx in indices:
            # Try +step and -step
            h_test = best_h.copy()
            h_test[idx] += step
            val_plus = _evaluate(h_test)
            
            h_test[idx] = best_h[idx] - 2 * step
            val_minus = _evaluate(h_test)
            
            if val_plus < best_val:
                best_h[idx] += step
                best_val = val_plus
                improved = True
            elif val_minus < best_val:
                best_h[idx] -= step
                best_val = val_minus
                improved = True
        
        if not improved:
            # Reduce step size
            step *= 0.8
            if step < 1e-6:
                break
        else:
            # Occasionally increase step if we've been improving
            if np.random.rand() < 0.1:
                step *= 1.1

    return best_h, best_val

def construct():
    start_time = time.time()
    
    # 1. Start with the seed (constant ones) as a baseline
    best_h = np.ones(N)
    best_val = _evaluate(best_h) # Should be 2.0
    
    # 2. Try several parametric initializations
    # We want to explore asymmetric, signed signals.
    # The key is to make the max autoconvolution smaller relative to sum^2.
    # For constant vector, max_conv = N * 1^2 = N. sum^2 = N^2. val = 2N^2/N^2 = 2.
    # If we introduce negative values, sum^2 might drop (if sum->0), but max_conv might also drop.
    # However, if sum is large and max_conv is small, we win.
    
    np.random.seed(42)
    
    # Define a set of starting points for the parametric optimizer
    # We focus on frequencies that create "chirp" like structures or beat patterns
    starts = []
    
    # Start 1: Low frequency sinusoid with tilt
    starts.append([1.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0])
    
    # Start 2: Higher frequency, negative tilt
    starts.append([1.0, 5.0, np.pi/2, 7.0, 0.0, -0.5, 0.0])
    
    # Start 3: Chirp-like (approximated by two close frequencies)
    starts.append([1.0, 10.0, 0.0, 10.5, 0.0, 0.0, 0.0])
    
    # Start 4: Random initial guess
    starts.append(np.random.rand(7) * [1.0, 10.0, 2*np.pi, 10.0, 2*np.pi, 1.0, 1.0] + 
                  [0.1, 0.1, 0.0, 0.1, 0.0, -0.5, -0.5])
    
    # Start 5: Another random
    starts.append(np.random.rand(7) * [1.0, 15.0, 2*np.pi, 15.0, 2*np.pi, 1.0, 1.0] + 
                  [0.1, 0.1, 0.0, 0.1, 0.0, 0.5, 0.5])

    for start in starts:
        if time.time() > DEADLINE - 20: # Leave time for refinement
            break
            
        # Optimize the parametric family
        opt_params, opt_val = _optimize_with_lbgfgs(start, DEADLINE - 10)
        
        # Convert optimized params to h
        h_candidate = _parametric_h(opt_params)
        
        # Evaluate
        val = _evaluate(h_candidate)
        
        if val < best_val:
            best_val = val
            best_h = h_candidate
            
        # Check time
        if time.time() > DEADLINE - 10:
            break

    # 3. Greedy refinement on the best candidate found
    if time.time() < DEADLINE - 5:
        refined_h, refined_val = _refine_greedy(best_h, max_iter=200, deadline=DEADLINE - 2)
        if refined_val < best_val:
            best_val = refined_val
            best_h = refined_h

    # Final check
    final_val = _evaluate(best_h)
    if final_val != np.inf and np.all(np.isfinite(best_h)):
        return best_h
    else:
        return np.ones(N)

if __name__ == "__main__":
    h = construct()
    val = _evaluate(h)
    print(f"Final Value: {val}")
    print(f"Score: {2.0 / val}")

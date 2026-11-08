"""
Simulated Annealing with Nelder-Mead refinement.
This approach explores the solution space globally using stochastic acceptance
of worse solutions, allowing it to escape local minima common in non-convex
autocorrelation optimization. It uses adaptive step sizes and periodic local
polishing with Nelder-Mead.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1
W = 1.0 / (M + 1)
TIME_LIMIT = 110.0
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _smooth_value_for_opt(h):
    """
    Objective function for local optimization (Nelder-Mead).
    We minimize the negative of the value.
    To handle the max operator in norm_inf smoothly, we use a soft-max approximation
    or simply rely on the fact that Nelder-Mead is derivative-free and robust enough
    for small regions. However, standard compute_value is non-smooth due to max.
    We can use a log-sum-exp trick for the denominator's max term to make it smooth.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    
    # Smooth approximation for max(g)
    # Using log-sum-exp with large beta
    beta = 100.0
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0
    val = l2 / (norm_1 * lse)
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    # Initialize with a diverse set of candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Gaussian peak
    x = np.arange(N)
    candidates.append(np.exp(-0.5 * ((x - N/2) / (N/6)) ** 2))
    
    # 3. Triangle
    candidates.append(1 - 2 * np.abs(x - (N-1)/2) / (N-1))
    
    # 4. Random smooth
    r = RNG.randn(N)
    # Simple low-pass filter
    kernel = np.ones(5)/5
    r = np.convolve(r, kernel, mode='same')
    r = (r - np.min(r)) / (np.max(r) - np.min(r) + 1e-10)
    candidates.append(r)
    
    # 5. Random binary-like (smoothed)
    r2 = RNG.rand(N)
    r2 = np.where(r2 > 0.5, 1.0, 0.0)
    r2 = np.convolve(r2, np.ones(3)/3, mode='same')
    candidates.append(r2)
    
    # Evaluate initial candidates
    for c in candidates:
        c = np.maximum(0.0, c)
        v = compute_value(c)
        if v > best_val:
            best_val = v
            best_h = c.copy()
    
    current_h = best_h.copy()
    current_val = best_val
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.999 # Geometric cooling
    step_size = 0.1
    min_step_size = 0.001
    max_step_size = 0.5
    
    # Adaptive step size logic
    no_improve_count = 0
    improve_threshold = 50 # Increase step if no improvement for this many steps
    
    while time.time() < deadline - 10: # Leave 10s for final polish
        
        # Generate perturbation
        # Mix of small and large jumps
        if RNG.rand() < 0.1:
            # Large jump: randomize a subset of indices
            mask = RNG.rand(N) < 0.2
            perturbation = RNG.normal(0, step_size * 2, N) * mask
        else:
            # Small jump: smooth noise
            perturbation = RNG.normal(0, step_size, N)
            # Apply simple smoothing to keep profile somewhat regular
            # perturbation = np.convolve(perturbation, np.ones(3)/3, mode='same')
            
        new_h = current_h + perturbation
        new_h = np.maximum(0.0, new_h)
        
        new_val = compute_value(new_h)
        
        # Acceptance criterion
        if new_val > current_val:
            current_h = new_h
            current_val = new_val
            no_improve_count = 0
            
            if new_val > best_val:
                best_val = new_val
                best_h = new_h.copy()
                # Slow down cooling if we found a new best
                # T = max(T_min, T * 0.99) # Optional: reheat slightly? No, standard SA cools.
        else:
            # Metropolis criterion
            delta = current_val - new_val
            if T > 0:
                prob = np.exp(-delta / T)
                if RNG.rand() < prob:
                    current_h = new_h
                    current_val = new_val
            no_improve_count += 1
            
        # Cooling
        T *= cooling_rate
        
        # Adaptive step size
        if no_improve_count > improve_threshold:
            # Stuck, try larger steps
            step_size = min(max_step_size, step_size * 1.5)
            no_improve_count = 0
        elif new_val > current_val:
            # Improving, try smaller steps
            step_size = max(min_step_size, step_size * 0.95)
            
        # Periodic local refinement
        # Every 1000 iterations or so, or if time permits
        if (int((time.time() - start_time) * 10) % 100 == 0) and time.time() < deadline - 20:
            # Use best_h for refinement
            try:
                res = opt.minimize(
                    _smooth_value_for_opt,
                    best_h.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                refined_h = np.maximum(0.0, res.x)
                refined_val = compute_value(refined_h)
                if refined_val > best_val:
                    best_val = refined_val
                    best_h = refined_h.copy()
                    # Update current state to best to continue from there
                    current_h = best_h.copy()
                    current_val = best_val
            except Exception:
                pass

    # Final polish on the best candidate
    if time.time() < deadline - 2:
        try:
            res = opt.minimize(
                _smooth_value_for_opt,
                best_h.copy(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            refined_h = np.maximum(0.0, res.x)
            refined_val = compute_value(refined_h)
            if refined_val > best_val:
                best_val = refined_val
                best_h = refined_h.copy()
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))

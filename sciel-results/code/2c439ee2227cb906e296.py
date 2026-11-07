"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _compute_value(h):
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negative
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    norm_1 = np.sum(g) / (m + 1)  # g is non-negative
    norm_inf = np.max(g)
    if norm_1 * norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Seed
    seed_h = np.ones(N)
    seed_val = _compute_value(seed_h)
    best_h = seed_h.copy()
    best_val = seed_val
    
    # Analytical shapes (Symmetric)
    # We will generate symmetric candidates
    def make_symmetric(x_half):
        x_half = np.maximum(x_half, 0.0)
        h = np.zeros(N)
        h[:N//2] = x_half
        h[N//2:] = x_half[::-1]
        # If N is even, middle two are x_half[-1] and x_half[-1]
        # If N is odd, middle is x_half[N//2]
        # N=50 is even.
        # h[0..24] = x_half[0..24]
        # h[25..49] = x_half[24..0]
        return h

    # Generate initial half-vectors
    half_N = N // 2
    inits = []
    
    # Uniform
    inits.append(np.ones(half_N))
    
    # Triangular
    tri = np.linspace(1.0, 0.1, half_N)
    inits.append(tri)
    
    # Parabolic
    x = np.linspace(0, 1, half_N)
    par = 1.0 - (2*x - 1)**2
    inits.append(np.maximum(par, 0.0))
    
    # Gaussian (centered at end of half vector to make full vector centered)
    # Full vector center is between 24 and 25.
    # Half vector center for symmetry is at index 24 (last element).
    # So peak should be at the end of the half vector.
    gauss = np.exp(-((np.arange(half_N) - (half_N-1)) / 5.0)**2)
    inits.append(gauss)
    
    gauss2 = np.exp(-((np.arange(half_N) - (half_N-1)) / 10.0)**2)
    inits.append(gauss2)
    
    # Ramp
    ramp = np.linspace(0.5, 1.0, half_N)
    inits.append(ramp)

    # Evaluate initial shapes
    for x_half in inits:
        h_full = make_symmetric(x_half)
        v = _compute_value(h_full)
        if v > best_val:
            best_val = v
            best_h = h_full.copy()
    
    # Optimization using Symmetric Subspace
    # We optimize x_half (size 25)
    def neg_value_sym(x_half):
        x_half = np.maximum(x_half, 0.0)
        h_full = make_symmetric(x_half)
        v = _compute_value(h_full)
        return -v

    # Use Nelder-Mead or Powell on the symmetric subspace
    # Powell is often better for higher dimensions and non-smooth constraints (handled by max(0,.) inside)
    # Nelder-Mead is good for low dimensions. 25 is moderate.
    
    best_x_half = None
    if best_h is not None:
        # Extract best_x_half from best_h
        best_x_half = best_h[:half_N].copy()
    else:
        best_x_half = inits[0].copy()

    # Run optimization from multiple starting points
    # We have a time budget. Let's allocate it to several restarts.
    # Each optimization should take a few seconds.
    
    # We will use a wrapper to track time
    # scipy.optimize.minimize doesn't have a direct timeout, so we rely on maxiter or just letting it run if it's fast enough.
    # For N=25, Nelder-Mead should be fast.
    
    candidates = inits + [best_x_half]
    
    # Add some random candidates
    rng = np.random.default_rng(42)
    for _ in range(5):
        rand_x = rng.uniform(0.1, 1.0, size=half_N)
        candidates.append(rand_x)
    
    for x0 in candidates:
        if time.time() > deadline - 5:
            break
            
        try:
            # Use Powell for robustness, or Nelder-Mead
            # Powell tends to be slower per iteration but more robust for this shape.
            # Let's try Nelder-Mead first as it's often faster for this size.
            result = opt.minimize(neg_value_sym, x0, method='Nelder-Mead', 
                                  options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 1000, 'maxfev': 1000})
            
            if result.success or result.fun < -best_val:
                v = -result.fun
                if v > best_val:
                    best_val = v
                    best_x_half = result.x
                    best_h = make_symmetric(best_x_half)
        except Exception:
            pass

    # Final polish: Coordinate descent on full vector to break symmetry if beneficial
    # Check if breaking symmetry helps
    if time.time() < deadline - 2:
        h = best_h.copy()
        step_size = 0.05
        for _ in range(500):
            if time.time() > deadline - 1:
                break
            idx = rng.integers(0, N)
            old_val = best_val
            h[idx] += rng.uniform(-step_size, step_size)
            h[idx] = max(0.0, h[idx])
            v = _compute_value(h)
            if v > best_val:
                best_val = v
                best_h = h.copy()
            else:
                h[idx] = best_h[idx]
            if v > old_val:
                step_size = min(step_size * 1.1, 0.5)
            else:
                step_size = max(step_size * 0.9, 0.01)

    return [max(0.0, x) for x in best_h]
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _max_autocorr_sidelobe(h):
    """Compute max |autocorr| at non-zero lags using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return np.max(np.abs(acf[1:]))

def _local_search(h, deadline):
    """Greedy local search: flip bits to reduce max sidelobe."""
    best_h = h.copy()
    best_val = _max_autocorr_sidelobe(best_h)
    
    while time.time() < deadline:
        improved = False
        # Try flipping each bit
        for i in range(N):
            h[i] = -h[i]
            val = _max_autocorr_sidelobe(h)
            h[i] = -h[i]  # revert
            
            if val < best_val:
                h[i] = -h[i]  # keep flip
                best_val = val
                best_h = h.copy()
                improved = True
        
        if not improved:
            break
    
    return best_h, best_val

def _random_restart(h, deadline):
    """Perform random restarts with local search."""
    best_h = h.copy()
    best_val = _max_autocorr_sidelobe(best_h)
    
    while time.time() < deadline:
        # Create a new random signed step function
        h_new = np.random.choice([-1, 1], size=N)
        
        # Run local search on the new starting point
        h_local, val_local = _local_search(h_new, deadline)
        
        if val_local < best_val:
            best_h = h_local
            best_val = val_local
    
    return best_h, best_val

def construct():
    """Construct a signed step function that minimizes the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Set a safety margin: aim to finish well before 120s
    np.random.seed(42)
    
    # Strategy: Continuous relaxation via L-BFGS-B on spectral amplitudes,
    # followed by discrete refinement.
    
    best_h = np.random.choice([-1, 1], size=N)
    best_val = _max_autocorr_sidelobe(best_h)
    
    # 1. Continuous Optimization Phase
    # We optimize the magnitudes of the FFT coefficients to minimize the sidelobes.
    # The phase is kept fixed (or randomized) and we adjust amplitudes.
    # Actually, a better continuous relaxation is to optimize the signal itself in a continuous space
    # and then project to {-1, 1}.
    
    # Let's try optimizing the continuous signal x in R^N to minimize max |autocorr(x)[1:]|
    # This is non-convex, but L-BFGS-B can find good basins.
    
    def objective(x):
        # x is continuous
        val = _max_autocorr_sidelobe(x)
        # Add a small penalty to keep values bounded or encourage binary-ness?
        # Just minimizing the autocorr sidelobe is the goal.
        return val
    
    # Initialize x with random values in [-1, 1]
    x0 = np.random.uniform(-1, 1, size=N)
    
    # Run L-BFGS-B
    # We need to be careful with the time budget.
    # L-BFGS-B might take a while. Let's limit iterations or time.
    
    # Since the objective is not differentiable (max of abs), we use a smooth approximation
    # or rely on the optimizer's robustness. L-BFGS-B uses gradients.
    # The gradient of max |f(x)| is tricky.
    # Alternatively, we can minimize the sum of squares of sidelobes, which is differentiable.
    
    def objective_smooth(x):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(x, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft).real[:N]
        # Sum of squares of sidelobes
        return np.sum(np.abs(acf[1:])**2)
    
    # Minimize sum of squares
    res = minimize(objective_smooth, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 5000})
    
    if time.time() < deadline:
        x_opt = res.x
        # Project to {-1, 1}
        h_proj = np.sign(x_opt)
        # Ensure no zeros (sign(0)=0)
        h_proj[h_proj == 0] = 1
        
        val_proj = _max_autocorr_sidelobe(h_proj)
        if val_proj < best_val:
            best_h = h_proj
            best_val = val_proj
            
        # Refine with local search
        if time.time() < deadline - 5.0:
            h_local, val_local = _local_search(h_proj, deadline)
            if val_local < best_val:
                best_h = h_local
                best_val = val_local
    
    # 2. Random Restart Phase with Local Search
    if time.time() < deadline - 5.0:
        # Try a few random restarts with local search to see if we can beat the continuous result
        for _ in range(3):
            if time.time() >= deadline - 5.0:
                break
            h_new = np.random.choice([-1, 1], size=N)
            h_local, val_local = _local_search(h_new, deadline)
            if val_local < best_val:
                best_h = h_local
                best_val = val_local
    
    return best_h

# EVOLVE-BLOCK-END

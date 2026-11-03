"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    max_ov = 0.0
    for k in range(1, N):
        ov = np.sum(h * (1.0 - np.roll(h, k)))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _chirp_h(sweep_rate, amplitude, phase):
    """
    Generate step heights from a linear chirp with quadratic phase.
    h[i] = 0.5 + amplitude * cos(2*pi*(f0*i + 0.5*sweep_rate*i^2)/N + phase)
    Projected to [0,1] with sum constraint.
    """
    i = np.arange(N, dtype=np.float64)
    # Quadratic phase: phi[i] = 2*pi*(f0*i + 0.5*mu*i^2)/N + phase
    # For a pure chirp, f0=0, so phi[i] = pi*mu*i^2/N + phase
    phi = np.pi * sweep_rate * i**2 / N + phase
    x = amplitude * np.cos(phi)
    # Center to have zero mean
    x -= np.mean(x)
    h = 0.5 + x
    # Project to [0,1]
    h = np.clip(h, 0.0, 1.0)
    # Adjust sum to be exactly N/2
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute difference, being careful about bounds
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        # Second pass if needed
        diff2 = N / 2.0 - np.sum(h)
        if abs(diff2) > 1e-15:
            h += diff2 / N
            h = np.clip(h, 0.0, 1.0)
    return h

def _pgd_polish(h0, max_iter=500, lr=0.01, deadline=None):
    """
    Projected gradient descent to polish the solution.
    Minimize max overlap using smooth approximation.
    """
    h = h0.copy()
    best_h = h.copy()
    best_ov = _max_overlap(h)
    
    beta = 10.0  # Smoothing parameter for log-sum-exp
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all overlaps
        overlaps = np.zeros(N - 1)
        for k in range(1, N):
            overlaps[k-1] = np.sum(h * (1.0 - np.roll(h, k)))
        
        # Smooth max via log-sum-exp
        # grad of smooth max wrt h
        # d/dh_k [sum h[i]*(1-h[i+k])] = (1-h[k]) - sum_j h[j-k]... 
        # Actually let's compute gradient numerically or analytically
        
        # Analytic gradient of overlap(k) wrt h[i]:
        # overlap(k) = sum_j h[j]*(1-h[(j+k)%N])
        # d overlap(k)/d h[i] = (1 - h[(i+k)%N]) - h[(i-k)%N]
        
        # For smooth max: softmax(overlaps / beta) weighted sum of gradients
        shifted = overlaps - np.max(overlaps)
        weights = np.exp(shifted * beta) / np.sum(np.exp(shifted * beta))
        
        grad = np.zeros(N)
        for k in range(1, N):
            w = weights[k-1]
            for i in range(N):
                grad[i] += w * ((1.0 - h[(i+k) % N]) - h[(i-k) % N])
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project to [0,1] with sum constraint
        h_new = np.clip(h_new, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h_new)
        h_new += diff / N
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Check improvement
        ov_new = _max_overlap(h_new)
        if ov_new < best_ov:
            best_ov = ov_new
            best_h = h_new.copy()
        
        h = h_new
    
    return best_h, best_ov

def construct():
    """
    Construct step heights h[0..N-1] in [0,1] with sum exactly N/2.
    Minimize the maximum overlap of h with shifted 1-h.
    
    Strategy: Use linear chirps (quadratic phase) which have flat autocorrelation
    properties, then polish with projected gradient descent.
    """
    deadline = time.time() + 130  # 20s safety margin
    
    best_ov = np.inf
    best_h = np.full(N, 0.5)
    
    # Phase 1: Search over chirp parameters
    np.random.seed(42)
    
    # Try various sweep rates and amplitudes
    for trial in range(3000):
        if time.time() > deadline:
            break
        
        # Sweep rate controls the chirp rate
        # Small sweep rates give near-constant signals
        # Larger sweep rates give more complex patterns
        if trial < 500:
            # Fine grid over small sweep rates
            mu = (trial / 500.0) * 2.0 - 1.0  # [-1, 1]
            amp = 0.3
        elif trial < 1000:
            # Medium sweep rates
            mu = np.random.uniform(-3.0, 3.0)
            amp = np.random.uniform(0.2, 0.5)
        else:
            # Random exploration
            mu = np.random.uniform(-5.0, 5.0)
            amp = np.random.uniform(0.1, 0.6)
        
        phase = np.random.uniform(0, 2 * np.pi)
        
        h = _chirp_h(mu, amp, phase)
        ov = _max_overlap(h)
        
        if ov < best_ov:
            best_ov = ov
            best_h = h.copy()
    
    # Phase 2: Polish the best candidate with PGD
    if time.time() < deadline - 30:
        polished_h, polished_ov = _pgd_polish(best_h, max_iter=300, lr=0.005, deadline=deadline - 10)
        if polished_ov < best_ov:
            best_ov = polished_ov
            best_h = polished_h
    
    # Phase 3: Try a few more chirp initializations and polish
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        for extra_trial in range(5):
            if time.time() > deadline - 15:
                break
            mu = np.random.uniform(-3.0, 3.0)
            amp = np.random.uniform(0.2, 0.5)
            phase = np.random.uniform(0, 2 * np.pi)
            h = _chirp_h(mu, amp, phase)
            ov = _max_overlap(h)
            if ov < best_ov + 2.0:  # Only polish if close to best
                polished_h, polished_ov = _pgd_polish(h, max_iter=200, lr=0.005, deadline=deadline - 10)
                if polished_ov < best_ov:
                    best_ov = polished_ov
                    best_h = polished_h
    
    # Final: ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # Verify sum
        diff2 = N / 2.0 - np.sum(best_h)
        if abs(diff2) > 1e-12:
            best_h += diff2 / N
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END

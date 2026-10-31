"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    N = len(h)
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        overlap = np.sum(h * (1.0 - shifted))
        if overlap > max_ov:
            max_ov = overlap
    return max_ov

def _param_to_h(params):
    """Convert parameters to step heights.
    
    params = [low_val, high_val, transition_center, transition_width, phase]
    Creates a binary-like sequence with smooth transitions.
    """
    low_val, high_val, center, width, phase = params
    idx = np.arange(N)
    
    # Create a periodic pattern: low for first half, high for second half
    # with smooth transitions at the boundaries
    # Pattern: [low ... transition ... high ... transition ... low ...]
    
    # Define two transition regions
    # Transition 1: from low to high around center
    # Transition 2: from high to low around center + N/2
    
    def sigmoid(x, center, width):
        if width < 1e-6:
            return np.where(x >= center, 1.0, 0.0)
        return 1.0 / (1.0 + np.exp(-(x - center) / width))
    
    # Create the base pattern
    # We want: low on [0, center), transition to high, high on (center, center+N/2), transition to low
    # Use periodic extension
    
    # Compute distance from each point to the nearest transition center
    # Transition centers are at 'center' and 'center + N/2'
    
    # For each index, determine which region it's in
    # Normalize to [0, N)
    idx_mod = idx % N
    
    # Distance to first transition (going from low to high)
    d1 = np.abs(idx_mod - center)
    # Wrap around
    d1 = np.minimum(d1, N - d1)
    
    # Distance to second transition (going from high to low)
    d2 = np.abs(idx_mod - (center + N/2))
    d2 = np.minimum(d2, N - d2)
    
    # The value should be:
    # - low when far from both transitions and in the "low" region
    # - high when far from both transitions and in the "high" region
    # - transition near the boundaries
    
    # Better approach: use a smooth step function
    # Define t(x) = 0 in low region, 1 in high region
    # t is a periodic function with two transitions
    
    # Use the phase to shift the pattern
    shifted_idx = (idx_mod - phase) % N
    
    # t = 0.5 + 0.5 * sin(2*pi*(shifted_idx - center)/N) would give smooth but not binary
    # Instead, use a sharper transition
    
    # Create a "square wave" with smooth edges
    # Base square wave: 0 for x in [0, N/2), 1 for x in [N/2, N)
    base = np.where(shifted_idx < N/2, 0.0, 1.0)
    
    # Smooth the transitions using sigmoid
    # Transition at shifted_idx = 0 (or N): from 1 to 0
    # Transition at shifted_idx = N/2: from 0 to 1
    
    # Distance to transition at 0
    d_to_0 = np.minimum(shifted_idx, N - shifted_idx)
    # Distance to transition at N/2
    d_to_half = np.minimum(np.abs(shifted_idx - N/2), N - np.abs(shifted_idx - N/2))
    
    # Smooth transition: use sigmoid of distance / width
    smooth_0 = sigmoid(d_to_0, 0, width)  # 1 when far from 0, 0 when near 0
    smooth_half = sigmoid(d_to_half, 0, width)  # 1 when far from N/2, 0 when near N/2
    
    # Combine: we want low (0) near 0, high (1) near N/2
    # t = smooth_0 * (1 - smooth_half) gives 0 near 0, 0 near N/2, 1 in between
    # Actually, let's think differently
    
    # Simpler: create a smooth periodic function that's ~0 on first half, ~1 on second half
    # Use: t(x) = sigmoid((x - N/4) / width) - sigmoid((x - 3N/4) / width)
    # This gives: ~0 for x < N/4, transitions up, ~1 for N/4 < x < 3N/4, transitions down, ~0 for x > 3N/4
    
    # But we need it periodic. Let's use a different approach.
    
    # Define a smooth step that goes from 0 to 1 over width W
    # t(x) = 0 for x <= 0, sigmoid for 0 < x < W, 1 for x >= W
    
    # For our periodic case with period N:
    # We want a function that is 0 on [0, a), transitions to 1 on [a, a+W],
    # stays 1 on [a+W, b), transitions to 0 on [b, b+W], stays 0 on [b+W, N)
    # where a and b are the transition centers, and b - a = N/2 (for integral = 1)
    
    # Let's parameterize by the transition centers
    # t1 = center of first transition (0 to 1)
    # t2 = center of second transition (1 to 0) = t1 + N/2
    
    # Value at x:
    # If x is in [t2+W, t1+W) (mod N): low region
    # If x is in [t1+W, t2-W) (mod N): high region
    # Transitions in between
    
    # Compute the "smooth square wave"
    # Use: s(x) = sigmoid((x - t1) / width) - sigmoid((x - t2) / width)
    # This is 0 when x << t1, 1 when t1 << x << t2, 0 when x >> t2
    
    t1 = center
    t2 = (center + N/2) % N
    
    # Handle the wrap-around for t2
    # If t2 < t1, we need to handle the discontinuity
    
    # Compute s(x) for each x
    s = np.zeros(N)
    for i in range(N):
        x = (idx[i] - phase) % N
        
        # Compute sigmoid((x - t1) / width)
        if width < 1e-6:
            sig1 = 1.0 if x >= t1 else 0.0
        else:
            sig1 = 1.0 / (1.0 + np.exp(-(x - t1) / width))
        
        # Compute sigmoid((x - t2) / width)
        if width < 1e-6:
            sig2 = 1.0 if x >= t2 else 0.0
        else:
            sig2 = 1.0 / (1.0 + np.exp(-(x - t2) / width))
        
        s[i] = sig1 - sig2
    
    # s should be approximately 0 in low region, 1 in high region
    # Map to [low_val, high_val]
    h = low_val + (high_val - low_val) * s
    
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Normalize to ensure sum = N/2
    total = np.sum(h)
    if total > 0:
        h = h * (N / 2.0) / total
        h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective_params(params):
    """Objective function for parameter optimization."""
    h = _param_to_h(params)
    return _max_overlap(h)

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Start with the seed solution
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Define parameter bounds
    # [low_val, high_val, center, width, phase]
    bounds = [
        (0.0, 0.5),      # low_val: in [0, 0.5]
        (0.5, 1.0),      # high_val: in [0.5, 1.0]
        (0.0, float(N)), # center: transition center
        (0.5, 10.0),     # width: transition width
        (0.0, float(N)), # phase: phase shift
    ]
    
    # Try differential evolution for global search
    try:
        result = differential_evolution(
            _objective_params,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            timeout=60
        )
        
        if time.time() > deadline:
            return best_h
        
        h_candidate = _param_to_h(result.x)
        score_candidate = _max_overlap(h_candidate)
        if score_candidate < best_score:
            best_h = h_candidate
            best_score = score_candidate
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_h
    
    # Local polish with SLSQP on the full h vector
    try:
        x0 = best_h.copy()
        
        def obj_full(h):
            return _max_overlap(h)
        
        def constraint_sum(h):
            return np.sum(h) - N / 2.0
        
        result = minimize(
            obj_full,
            x0,
            method='SLSQP',
            bounds=[(0.0, 1.0)] * N,
            constraints=[{'type': 'eq', 'fun': constraint_sum}],
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if time.time() > deadline:
            return best_h
        
        if result.fun < best_score:
            h_candidate = np.clip(result.x, 0.0, 1.0)
            s = np.sum(h_candidate)
            if s > 0:
                h_candidate = h_candidate * (N / 2.0) / s
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            score_candidate = _max_overlap(h_candidate)
            if score_candidate < best_score:
                best_h = h_candidate
                best_score = score_candidate
    except Exception:
        pass
    
    return best_h
# EVOLVE-BLOCK-END

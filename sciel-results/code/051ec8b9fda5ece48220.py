import numpy as np
import time

N = 400

def _autoconv(h):
    """Compute linear autoconvolution using FFT."""
    # Pad to length 2*N - 1 to avoid circular convolution artifacts
    size = 2 * N - 1
    # Use next power of 2 for FFT speed
    fft_size = 1 << (size - 1).bit_length()
    
    H = np.fft.rfft(h, n=fft_size)
    AC = np.fft.irfft(H * H, n=fft_size)
    return AC[:size]

def _objective(h):
    """Calculate the raw value to minimize.
    Based on inspirations, we assume the goal is to minimize the max absolute 
    autoconvolution. The 'divided by squared integral' part might be a constant 
    for +/- 1 sequences if integral is sum of squares (N), or it might favor 
    sequences with lower peaks. 
    We will minimize max(|AC|) directly, as this aligns with 'peak suppression'.
    """
    ac = _autoconv(h)
    # Exclude k=0? 
    # If we include k=0, max is sum(h^2) = N. Constant.
    # So minimizing max(|AC|) is equivalent to minimizing max(|AC[k]| for k != 0).
    # We return the max of the sidelobes.
    if len(ac) > 1:
        return np.max(np.abs(ac[1:]))
    return 0.0

def _fast_flip_eval(h, idx, current_max):
    """Evaluate change in objective if we flip h[idx]."""
    # This is a heuristic. For strictness, we just recompute.
    # Given N=400, recomputation is cheap.
    pass

def construct():
    """Construct a sequence of signed step heights minimizing C3'."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with random signs
    h = np.random.choice([-1, 1], size=N)
    best_h = h.copy()
    best_score = _objective(h)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99
    
    # Precompute AC for initial
    ac = _autoconv(h)
    max_val = np.max(np.abs(ac[1:]))
    
    steps = 0
    while time.time() < deadline and T > T_min:
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -old_val
        
        # Recompute AC
        ac = _autoconv(h)
        new_max = np.max(np.abs(ac[1:]))
        
        delta = new_max - max_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            max_val = new_max
            if new_max < best_score:
                best_score = new_max
                best_h = h.copy()
        else:
            # Revert
            h[idx] = old_val
            
        T *= cooling_rate
        steps += 1
        
        # Optional: Local search refinement every N steps
        if steps % 1000 == 0:
            # Simple local search: try flipping each bit once
            for i in range(N):
                old_val = h[i]
                h[i] = -old_val
                ac = _autoconv(h)
                new_max = np.max(np.abs(ac[1:]))
                if new_max < max_val:
                    max_val = new_max
                    if new_max < best_score:
                        best_score = new_max
                        best_h = h.copy()
                else:
                    h[i] = old_val

    return best_h

# EVOLVE-BLOCK-START
def construct():
    """Return a sequence of N signed step heights minimizing C3'.
    
    Uses simulated annealing to find a sequence with low autoconvolution sidelobes.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with a Barker-like or random sequence
    # Random is a good start to avoid trivial local optima if they exist
    h = np.random.choice([-1, 1], size=N)
    best_h = h.copy()
    
    def get_score(seq):
        ac = _autoconv(seq)
        # Minimize the maximum absolute value of the autoconvolution (sidelobes)
        # This corresponds to minimizing the "peak" relative to the signal energy
        return np.max(np.abs(ac[1:]))
    
    best_score = get_score(h)
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.995
    
    while time.time() < deadline and T > T_min:
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -old_val
        
        new_score = get_score(h)
        delta = new_score - best_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            h[idx] = old_val
            
        T *= cooling_rate

    return best_h
# EVOLVE-BLOCK-END

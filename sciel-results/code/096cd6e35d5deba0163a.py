"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute autoconvolution of x using FFT.
    For real x, autoconv = ifft(fft(x)**2) but we need linear convolution.
    Actually, autoconvolution of a sequence x is (x * x)[k] = sum_i x[i] x[k-i].
    Using FFT: pad to size >= 2N-1, compute FFT, square, IFFT, take real part.
    """
    size = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * f, size)[:2 * N - 1]
    return ac

def _score(x):
    """Return the C3' value: max |autoconv| / (integral)^2.
    Lower is better for the evaluator (raw value).
    """
    ac = _autoconv(x)
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (integral ** 2)

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with random signs
    best_x = np.ones(N, dtype=np.float64)
    best_x[np.random.RandomState(42).randint(0, 2, N) == 1] = -1.0
    best_score = _score(best_x)
    
    # Also try some structured initial guesses
    candidates = []
    
    # All ones
    x1 = np.ones(N)
    candidates.append(x1)
    
    # Alternating
    x2 = np.ones(N)
    x2[1::2] = -1.0
    candidates.append(x2)
    
    # Block patterns
    for block_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        x = np.ones(N)
        for start in range(0, N, 2 * block_size):
            end = min(start + block_size, N)
            x[start:end] = -1.0
        candidates.append(x.copy())
    
    # Random candidates
    rng = np.random.RandomState(0)
    for _ in range(50):
        x = rng.choice([-1.0, 1.0], size=N)
        candidates.append(x)
    
    # Evaluate candidates
    for x in candidates:
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    # Local search: flip individual signs
    # Use a steepest descent approach: find the flip that gives the biggest improvement
    x = best_x.copy()
    current_score = best_score
    improved = True
    iterations = 0
    max_iterations = 50000
    
    while improved and time.time() < deadline and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try flipping each position and see if it improves
        # For efficiency, we can use the fact that flipping x[i] changes the autoconv
        # in a structured way, but for now let's do a simple greedy search
        
        # To speed up, we'll only try a subset of positions each iteration
        # or use a smarter approach
        
        # Actually, let's do a full pass but break early if no improvement found
        # Since N=400, each flip requires recomputing autoconv which is O(N log N)
        # 400 flips per iteration * 50000 iterations could be too slow
        
        # Let's limit: only do local search for a limited number of full passes
        # and use a faster evaluation
        
        # Alternative: use gradient-based approach with soft-max
        # For now, do limited local search
        
        # Try flipping positions in random order, stop after first improvement
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline:
                break
            
            old_val = x[i]
            x[i] = -old_val
            new_score = _score(x)
            if new_score < current_score - 1e-10:
                current_score = new_score
                improved = True
                break
            else:
                x[i] = old_val
        
        if improved:
            best_x = x.copy()
            best_score = current_score
        else:
            # Try random restarts if stuck
            for _ in range(5):
                x_restart = x.copy()
                # Flip a random subset
                n_flips = rng.randint(1, 20)
                flip_indices = rng.choice(N, size=n_flips, replace=False)
                x_restart[flip_indices] *= -1
                s = _score(x_restart)
                if s < current_score - 1e-10:
                    x = x_restart
                    current_score = s
                    improved = True
                    break
    
    return best_x
# EVOLVE-BLOCK-END

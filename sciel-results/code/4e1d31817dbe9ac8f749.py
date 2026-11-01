# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _compute_c3_bound_fast(heights):
    """Compute the C3 upper bound using FFT for speed."""
    n = len(heights)
    # Zero-pad to avoid circular convolution effects
    size = 2 * n
    h_padded = np.zeros(size)
    h_padded[:n] = heights
    
    # FFT
    H = np.fft.fft(h_padded)
    # Autocorrelation is inverse FFT of |H|^2
    # The aperiodic autocorrelation c(k) for k>=0 is the first n elements of the inverse FFT
    # Note: np.fft.ifft returns complex numbers, take real part
    auto = np.fft.ifft(H * np.conj(H)).real[:n]
    
    # C3 bound: (sum |c(k)|^3)^(1/3) / N
    # The problem statement says "score is the upper bound on C3" and "raw value (lower is better)"
    # The seed program divides by N. We should keep this normalization to be comparable.
    c3 = (np.sum(np.abs(auto)**3))**(1.0/3.0) / n
    return c3

def construct():
    start_time = time.time()
    deadline = 150.0  # Evaluator kills at 150s, keep margin
    safety_margin = 5.0
    
    # Initialize with random or simple pattern
    rng = np.random.RandomState(42)
    best_h = rng.choice([-1, 1], size=N)
    best_score = _compute_c3_bound_fast(best_h)
    
    # Local Search with Hill Climbing
    # We can afford many more evaluations with FFT
    current_h = best_h.copy()
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 10000 # Reset if no improvement for this many steps
    
    while time.time() - start_time < deadline - safety_margin:
        # Pick a random index to flip
        idx = rng.randint(0, N)
        old_val = current_h[idx]
        current_h[idx] = -old_val
        
        new_score = _compute_c3_bound_fast(current_h)
        
        if new_score < current_score - 1e-12:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
            no_improve_count = 0
        else:
            current_h[idx] = old_val
            no_improve_count += 1
            
        # If stuck, restart with a new random configuration or a perturbation of best
        if no_improve_count > max_no_improve:
            # Try a few random restarts or large perturbations
            for _ in range(10):
                new_h = rng.choice([-1, 1], size=N)
                new_score = _compute_c3_bound_fast(new_h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = new_h
                    current_h = new_h.copy()
                    current_score = new_score
                    no_improve_count = 0
                    break
            else:
                # If no random restart helped, try perturbing best_h
                new_h = best_h.copy()
                # Flip 10% of the bits
                num_flips = N // 10
                flip_indices = rng.choice(N, size=num_flips, replace=False)
                new_h[flip_indices] *= -1
                new_score = _compute_c3_bound_fast(new_h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = new_h
                    current_h = new_h.copy()
                    current_score = new_score
                    no_improve_count = 0
                else:
                    no_improve_count = 0 # Reset counter to allow more search from current local min

    return best_h
# EVOLVE-BLOCK-END

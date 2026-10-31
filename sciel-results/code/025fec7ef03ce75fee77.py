"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_metric(h):
    """
    Computes the sum of squared discrete autocorrelation coefficients.
    This serves as a proxy for the autoconvolution upper bound.
    Lower is better.
    """
    h_float = h.astype(np.float64)
    # Zero padding to avoid circular convolution
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.fft(h_float, n=n_fft)
    P = np.abs(F) ** 2
    A = np.fft.ifft(P).real
    # A[0] is lag 0, A[1] is lag 1, etc.
    # We take the first N lags.
    return np.sum(A[:N] ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial random sequence
    h = np.random.choice([-1, 1], size=N)
    best_h = h.copy()
    best_score = _compute_metric(best_h)
    
    best_global_h = best_h.copy()
    best_global_score = best_score
    
    # Local search with random restarts
    while time.time() < deadline:
        # Hill climb from current best_h
        h_curr = best_h.copy()
        current_score = best_score
        no_improve_count = 0
        
        while time.time() < deadline and no_improve_count < 1000:
            # Pick random index to flip
            idx = np.random.randint(N)
            
            # Flip
            h_curr[idx] *= -1
            
            new_score = _compute_metric(h_curr)
            
            if new_score < current_score:
                current_score = new_score
                best_h = h_curr.copy()
                no_improve_count = 0
            else:
                # Revert
                h_curr[idx] *= -1
                no_improve_count += 1
        
        # Update global best
        if current_score < best_global_score:
            best_global_score = current_score
            best_global_h = best_h.copy()
            
        # Restart with a new random sequence
        h_curr = np.random.choice([-1, 1], size=N)
        best_h = h_curr.copy()
        current_score = _compute_metric(best_h)
        
        # If this new random start is better, update local best
        if current_score < best_global_score:
            best_global_score = current_score
            best_global_h = best_h.copy()

    return best_global_h
# EVOLVE-BLOCK-END

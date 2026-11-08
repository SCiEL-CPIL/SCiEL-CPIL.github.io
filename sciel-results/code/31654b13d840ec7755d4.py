import numpy as np
import time

N = 400

def _autocorr(h):
    """Compute circular autocorrelation via FFT. Returns array of length N."""
    H = np.fft.rfft(h)
    R = np.fft.irfft(np.abs(H) ** 2)
    return R

def construct():
    start_time = time.time()
    # Strict safety margin: 115s to ensure we finish before 120s limit
    deadline = start_time + 115.0

    best_h = None
    best_score = np.inf

    np.random.seed(42)
    
    # Pre-compute indices for fast delta calculation
    k_idx = np.arange(N)
    
    def get_delta_R(h, i):
        """Calculate the change in autocorrelation vector if h[i] is flipped."""
        val = h[i]
        idx_minus = (i - k_idx) % N
        idx_plus = (i + k_idx) % N
        delta = -2.0 * val * (h[idx_minus] + h[idx_plus])
        delta[0] = 0.0 # R[0] does not change for +/-1 flips
        return delta

    # Greedy Local Search with Random Restarts
    # Structural change: Instead of Simulated Annealing (which accepts worse moves),
    # we use a Greedy approach that only accepts improvements, with multiple restarts.
    
    while time.time() < deadline:
        # Start a new random sequence
        h = np.random.choice([-1.0, 1.0], size=N)
        R = _autocorr(h)
        current_score = np.max(np.abs(R[1:]))
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

        # Local search phase
        # We perform passes over the indices. In each pass, we try to flip each bit.
        # If a flip improves the score, we keep it.
        # We stop the pass if no flips improve the score.
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Randomize the order of indices to break symmetry and escape local minima faster
            for i in np.random.permutation(N):
                # Check time frequently inside the inner loop to avoid overshooting
                if time.time() > deadline - 1.0:
                    break
                
                delta_R = get_delta_R(h, i)
                R_new = R + delta_R
                new_score = np.max(np.abs(R_new[1:]))
                
                if new_score < current_score - 1e-9:
                    # Accept the flip
                    h[i] *= -1.0
                    R = R_new
                    current_score = new_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()

    if best_h is None:
        best_h = np.ones(N)

    return best_h

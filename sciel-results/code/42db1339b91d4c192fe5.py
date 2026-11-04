"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def construct():
    """
    Construct a signed step function (400 intervals) to minimize the upper bound on C3.
    The score is based on the L4 norm of the DFT (proxy for autoconvolution energy).
    We use a simulated annealing style local search with fast DFT updates.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    rng = np.random.RandomState(42)

    # Initialize with random signs
    best_h = rng.choice([-1.0, 1.0], size=N)
    best_H = np.fft.fft(best_h)
    
    # Precompute phase factors for fast update: phases[i, k] = exp(-i 2 pi k i / N)
    # H_k = sum_i h_i * exp(-i 2 pi k i / N)
    # Flipping h_i changes H_k by -2 * h_i * exp(-i 2 pi k i / N)
    idx = np.arange(N)
    k = np.arange(N)
    phases = np.exp(-1j * 2 * np.pi * np.outer(idx, k) / N)  # shape (N, N)

    def get_score(H):
        # Minimize sum |H_k|^4 (L4 norm of spectrum)
        # This is proportional to the L2 norm of the autocorrelation
        power = np.abs(H) ** 2
        return np.sum(power ** 2)

    best_score = get_score(best_H)

    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.99999
    min_T = 1e-6

    # We will try multiple passes
    # Each pass tries to flip each element once in random order
    
    while time.time() < deadline - 2.0 and T > min_T:
        # Shuffle order of flips for this pass
        order = rng.permutation(N)
        
        for i in order:
            if time.time() > deadline - 2.0:
                break
            
            # Try flipping element i
            delta_H = -2.0 * best_h[i] * phases[i]
            trial_H = best_H + delta_H
            trial_score = get_score(trial_H)
            
            delta = trial_score - best_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or (T > 0 and np.exp(-delta / T) > rng.random()):
                best_h[i] *= -1.0
                best_H = trial_H
                best_score = trial_score
        
        T *= cooling_rate

    # Final polish: local search with only improving moves
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            
            delta_H = -2.0 * best_h[i] * phases[i]
            trial_H = best_H + delta_H
            trial_score = get_score(trial_H)
            
            if trial_score < best_score:
                best_h[i] *= -1.0
                best_H = trial_H
                best_score = trial_score
                improved = True

    return best_h
# EVOLVE-BLOCK-END

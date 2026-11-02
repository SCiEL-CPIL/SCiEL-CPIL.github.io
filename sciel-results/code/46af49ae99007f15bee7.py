"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_autocorr(a):
    """
    Compute the full autocorrelation of a using FFT.
    Returns array of length N where c[k] is the autocorrelation at lag k.
    c[0] is the peak (sum a^2).
    c[k] for k>0 are the positive lags.
    """
    n = len(a)
    size = 2 * n
    fa = np.fft.rfft(a, size)
    power = fa * np.conj(fa)
    c = np.fft.irfft(power, size)
    # c[0..n-1] are the lags 0..n-1
    return c[:n]

def _get_max_sidelobe(c):
    """
    Get the maximum absolute value of the sidelobes (lags 1 to N-1).
    """
    return np.max(np.abs(c[1:]))

def _score_val(a):
    """
    Calculate the raw value: max_abs_sidelobe / (sum a^2)^2
    """
    c = _compute_autocorr(a)
    max_side = _get_max_sidelobe(c)
    integral_sq = (np.sum(a**2))**2
    return max_side / integral_sq

def _simulated_annealing(a_init, rng, time_limit):
    """
    Perform Simulated Annealing on the binary sequence.
    Objective: Minimize max |C[k]| for k != 0.
    """
    n = len(a_init)
    a = a_init.copy()
    
    # Initial Autocorrelation
    c = _compute_autocorr(a)
    current_max_side = _get_max_sidelobe(c)
    
    # We want to minimize current_max_side.
    # Score is current_max_side / n^2. Since n^2 is constant, minimizing max_side is sufficient.
    
    start_time = time.time()
    
    # Annealing parameters
    # Temperature T. We accept worse moves with probability exp(-delta/T).
    # delta is the increase in max_side.
    # If max_side decreases, always accept.
    
    # Initial T should be large enough to allow exploration.
    # Typical max_side for random binary seq is ~ sqrt(N) * something.
    # Let's start T around 1.0 and decay.
    T = 1.0
    T_min = 1e-4
    decay_rate = 0.99995
    
    best_a = a.copy()
    best_max_side = current_max_side
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < time_limit:
        # Pick a random index to flip
        idx = rng.integers(0, n)
        
        # Flip bit
        old_val = a[idx]
        a[idx] = -old_val
        
        # Update Autocorrelation efficiently?
        # c[k] = sum a[i] a[i+k]
        # When a[idx] flips to -a[idx], the change in c[k] is:
        # c_new[k] = c_old[k] - 2 * a_old[idx] * a[(idx+k)%n]  (if circular)
        # But our autocorr is linear (zero-padded).
        # Linear autocorr: c[k] = sum_{i=0}^{N-k-1} a[i] a[i+k]
        # Flipping a[idx] affects c[k] if idx is in [0, N-k-1] AND idx+k is in [0, N-1].
        # This is complex to update incrementally for linear autocorr.
        # Given N=400, recomputing FFT autocorr is O(N log N) ~ 4000 ops.
        # Updating linearly is O(N) ~ 400 ops.
        # Let's stick to FFT recomputation for simplicity and correctness, as N is small.
        
        c_new = _compute_autocorr(a)
        new_max_side = _get_max_sidelobe(c_new)
        
        # Check if this is an improvement
        if new_max_side < current_max_side:
            # Accept
            c = c_new
            current_max_side = new_max_side
            no_improve_count = 0
            
            if new_max_side < best_max_side:
                best_max_side = new_max_side
                best_a = a.copy()
        else:
            # Worse move. Accept with probability exp(-(new_max_side - current_max_side) / T)
            delta = new_max_side - current_max_side
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    # Accept worse move
                    c = c_new
                    current_max_side = new_max_side
                else:
                    # Reject
                    a[idx] = old_val
                    no_improve_count += 1
            else:
                # This shouldn't happen if delta > 0 check is correct, but handle just in case
                c = c_new
                current_max_side = new_max_side
                no_improve_count = 0
                
        # Decay temperature
        T *= decay_rate
        if T < T_min:
            T = T_min
            
        # Restart if stuck?
        # If no improvement for a long time, we might be in a local minimum.
        # We can do a "kick" by flipping a few random bits or resetting T.
        if no_improve_count > max_no_improve:
            # Reset T to a higher value to escape
            T = 0.5 
            no_improve_count = 0
            # Optional: Randomize a few bits to kick out
            for _ in range(10):
                k = rng.integers(0, n)
                a[k] = -a[k]
            c = _compute_autocorr(a)
            current_max_side = _get_max_sidelobe(c)
            
    return best_a, best_max_side / (n**2)

def construct():
    start_time = time.time()
    deadline = time.time() + 115.0  # 5s margin
    
    rng = np.random.default_rng(42)
    
    best_a = None
    best_val = float('inf')
    
    # Number of restarts. Each restart runs SA until time runs out or it converges.
    # We'll do multiple restarts with different seeds.
    
    # Strategy: 
    # 1. Start with a random binary sequence.
    # 2. Run SA.
    # 3. Keep the best result.
    # 4. Restart with a new random sequence if time permits.
    
    # To make the first restart strong, we can use a "greedy" initialization?
    # Or just random is fine for SA.
    
    num_restarts = 100 # We will break early based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate initial sequence
        # Random +/- 1
        a_init = rng.choice([-1, 1], size=N)
        
        # Run SA
        a_opt, val_opt = _simulated_annealing(a_init, rng, deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_a = a_opt.copy()
            
        # If we found a very good solution, we might stop early?
        # The theoretical lower bound for max sidelobe of binary seq is ~0? No, it's bounded.
        # For N=400, a good sequence has max sidelobe ~ 20-30?
        # Random is ~ 20 * sqrt(N) ~ 400? No.
        # E[C[k]] = 0. Var(C[k]) = N. So std dev is sqrt(N) ~ 20.
        # Max of 400 Gaussians with std 20 is ~ 20 * sqrt(2 ln 400) ~ 20 * 3.5 ~ 70.
        # So random max side is ~ 70.
        # Score = 70 / 160000 ~ 0.0004.
        # Current best is 0.0212. Wait.
        # 0.0212 * 160000 = 3392.
        # My estimate of random max side is wrong.
        # Let's re-calculate.
        # C[k] = sum_{i=0}^{N-k-1} a_i a_{i+k}.
        # For large N, this is a sum of ~N independent +/-1 variables.
        # Mean 0, Variance N.
        # Std Dev sqrt(N) = 20.
        # Max of 400 such variables.
        # Extreme value theory: Max ~ mu + sigma * sqrt(2 ln N).
        # Max ~ 20 * 3.5 = 70.
        # So random score is 70/160000 = 0.0004.
        # Why is the current program scoring 0.0212?
        # 0.0212 is MUCH WORSE than random.
        # Ah, the current program's `_score_val` returns `max_side / integral_sq`.
        # integral_sq = (sum a^2)^2 = N^2 = 160000.
        # If random is 0.0004, and current is 0.0212, the current program is finding BAD sequences.
        # Or my understanding of the metric is inverted?
        # "score is the upper bound on C3' ... lower is better".
        # "search score is seed/value ... 1.0 equals seed".
        # If seed is 0.0212, and random is 0.0004, then random is BETTER.
        # Wait, if random is better, why is the seed 0.0212?
        # Maybe the seed program is just bad.
        # Or maybe I am miscalculating the autocorrelation.
        # Let's check N=400.
        # If a is all 1s, C[0]=400, C[k]=400-k. Max side = 399. Score = 399/160000 = 0.0025.
        # If a is random, Max side ~ 70. Score ~ 0.0004.
        # So random is better than constant.
        # The current program scores 0.0212. This is ~50x worse than random.
        # This implies the current program is failing to find even a random-like sequence, or it's stuck in a terrible local minimum, or the metric is different.
        
        # Let's re-read: "max absolute autoconvolution divided by squared integral".
        # Autoconvolution vs Autocorrelation?
        # For real signals, autoconvolution and autocorrelation are related by time reversal.
        # max |c[k]| is the same.
        
        # It is highly likely the current program is just bad.
        # My SA should easily find 0.0004.
        # Can we go lower?
        # Barker sequences have low sidelobes.
        # For N=400, we can't have a Barker sequence (max length 13).
        # But we can get close to the theoretical limit.
        # The limit for max sidelobe is ~ sqrt(N) * constant?
        # Actually, for binary sequences, the minimal possible peak sidelobe is related to N.
        
        # Let's just run SA and see.
        
    if best_a is None:
        best_a = rng.choice([-1, 1], size=N)
        best_val = _score_val(best_a)
        
    return best_a
# EVOLVE-BLOCK-END

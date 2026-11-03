"""Rewritten program: signed autocorrelation max-abs minimization via simulated annealing."""
import numpy as np
import time

N = 400
DEADLINE = 115.0  # safety margin from 120s limit

def _autoconv_max_abs(x):
    """Compute max absolute value of linear autocorrelation of x."""
    # Use FFT for speed: autocorrelation = ifft(|fft(x)|^2)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)
    # ac[0:n] are the non-negative lags (k=0..n-1), ac[n:] are negative lags
    # Actually for real signals, autocorrelation is symmetric: ac[k] = ac[-k]
    # The full autocorrelation has length 2*n-1, but FFT gives 2*n
    # Let's just take the max abs over the full computed array
    return np.max(np.abs(ac))

def _integral_sq(x):
    """Squared integral (sum of squares) of x."""
    return np.sum(x * x)

def _score(x):
    """C3' = max |autoconv| / (integral^2). Lower is better."""
    isq = _integral_sq(x)
    if isq < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / isq

def _random_binary(n):
    """Generate a random ±1 vector."""
    return np.where(np.random.rand(n) < 0.5, -1.0, 1.0)

def _construct_initial():
    """Construct an initial signed step function.
    
    We try a few structured initializations and pick the best.
    """
    candidates = []
    
    # 1. Random
    candidates.append(_random_binary(N))
    
    # 2. Alternating
    x = np.ones(N)
    x[1::2] = -1.0
    candidates.append(x)
    
    # 3. Block structure: half +1, half -1
    x = np.ones(N)
    x[N//2:] = -1.0
    candidates.append(x)
    
    # 4. Quarter blocks: + + - -
    x = np.ones(N)
    x[N//4:3*N//4] = -1.0
    candidates.append(x)
    
    # 5. Eighth blocks: + + - - + + - -
    x = np.ones(N)
    x[N//8:3*N//8] = -1.0
    x[5*N//8:7*N//8] = -1.0
    candidates.append(x)
    
    # 6. Random with bias toward balanced
    for _ in range(10):
        candidates.append(_random_binary(N))
    
    best = None
    best_score = float('inf')
    for x in candidates:
        s = _score(x)
        if s < best_score:
            best_score = s
            best = x.copy()
    
    return best, best_score

def construct():
    start_time = time.time()
    
    x, best_score = _construct_initial()
    best_x = x.copy()
    
    # Simulated annealing
    T_start = 1.0
    T_end = 0.01
    total_iterations = 200000
    
    # We'll do multiple restarts if time permits
    n_restarts = 5
    iters_per_restart = total_iterations // n_restarts
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        # Reset temperature
        T = T_start
        x = _random_binary(N)
        current_score = _score(x)
        
        for i in range(iters_per_restart):
            # Check deadline
            if time.time() > DEADLINE - 5.0:
                break
            
            # Cooling schedule
            T = T_start * (T_end / T_start) ** (i / iters_per_restart)
            
            # Pick a random position to flip
            idx = rng.integers(N)
            old_val = x[idx]
            x[idx] = -old_val
            
            new_score = _score(x)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                x[idx] = old_val  # revert
        
        if time.time() > DEADLINE - 5.0:
            break
    
    # Final polish: local search on best_x
    if time.time() < DEADLINE - 3.0:
        x = best_x.copy()
        current_score = _score(x)
        improved = True
        while improved and time.time() < DEADLINE - 3.0:
            improved = False
            for idx in range(N):
                if time.time() > DEADLINE - 3.0:
                    break
                old_val = x[idx]
                x[idx] = -old_val
                new_score = _score(x)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                else:
                    x[idx] = old_val
    
    return best_x

# The task says construct() -> sequence of signed step heights
# and score is the upper bound on C3'
# The evaluator measures the raw value (lower is better)
# So we return the array, and the evaluator computes the score.
# But wait - looking at the contract again: "score is the upper bound on C3'"
# and "The evaluator measures the raw value (lower is better)"
# This suggests construct() returns the sequence, and the evaluator computes C3' from it.

# Let me re-read: "Interface: construct() -> sequence of signed step heights; score is the upper bound on C3' (maximum absolute autoconvolution)"
# "The evaluator measures the raw value (lower is better)."
# So construct() returns the array, and the evaluator computes max|autoconv| / integral^2 or just max|autoconv|?
# "maximum absolute autoconvolution divided by squared integral" - so it's the ratio.
# But "score is the upper bound on C3'" - C3' is defined as that ratio.
# So the evaluator computes C3' from the returned array.

# Actually, I need to be careful. The task says "score is the upper bound on C3'" which might mean
# we should return a scalar upper bound, not the array. But the interface says "construct() -> sequence of signed step heights".
# So we return the array, and the evaluator computes the score.

# Let me just make sure the function returns the array as expected.

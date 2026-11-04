"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_fft(x):
    """Compute autocorrelation using FFT for all lags 0..n-1."""
    n = len(x)
    size = 1
    while size < 2*n:
        size *= 2
    fx = np.fft.rfft(x, size)
    ac = np.fft.irfft(fx * np.conj(fx), size)[:n]
    return ac

def _l3_score(acs):
    """Compute L3 norm of autocorrelation excluding lag 0."""
    return np.sum(np.abs(acs[1:])**3)

def _delta_l3(acs, i, old_val, new_val):
    """Compute change in L3 score when flipping x[i] from old_val to new_val.
    ac[lag] = sum_j x[j]*x[j+lag]
    Flipping x[i]: for each lag, the terms involving x[i] change.
    Term at position j=i: x[i]*x[i+lag] if i < n-lag
    Term at position j=i-lag: x[i-lag]*x[i] if i >= lag
    """
    n = len(acs)
    delta = 0.0
    for lag in range(1, n):
        delta_ac = 0.0
        # x[i] appears as first factor: j=i, need i < n-lag
        if i < n - lag:
            delta_ac += (new_val - old_val) * acs[lag - i]  # wait, this isn't right
        # Let me think differently.
        # ac[lag] = sum_{j=0}^{n-1-lag} x[j] * x[j+lag]
        # If we change x[i] to x[i]+d, then:
        #   ac[lag] changes by: d * (sum over terms where x[i] is first factor + sum where x[i] is second factor)
        #   x[i] is first factor when j=i: contributes d * x[i+lag] if i < n-lag
        #   x[i] is second factor when j+lag=i, i.e., j=i-lag: contributes d * x[i-lag] if i >= lag and i-lag >= 0
        # So delta_ac = d * (x[i+lag] if i < n-lag else 0) + d * (x[i-lag] if i >= lag else 0)
        # But we need x values, not just acs. This approach won't work with just acs.
        pass
    return delta  # placeholder

def _try_improve(x, max_time=100.0):
    """Local search to minimize L3 norm of autocorrelation."""
    start_time = time.time()
    n = len(x)
    x = x.copy()
    
    acs = _autocorr_fft(x)
    current_score = _l3_score(acs)
    best_score = current_score
    best_x = x.copy()
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > max_time:
                break
            # Try flipping position i
            old_val = x[i]
            x[i] = -old_val
            
            # Recompute autocorrelation via FFT
            new_acs = _autocorr_fft(x)
            new_score = _l3_score(new_acs)
            
            if new_score < current_score:
                current_score = new_score
                acs = new_acs
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                x[i] = old_val  # Revert
    
    return best_x, best_score

def construct():
    """Construct a signed step function to minimize the upper bound on C3."""
    start_time = time.time()
    max_time = 130.0  # Safety margin before 150s limit
    
    # Try multiple random initializations
    best_x = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Start with a few different initial patterns
    initializations = [
        np.ones(N),  # All ones
        np.random.choice([-1, 1], size=N),  # Random
        np.array([1 if i % 2 == 0 else -1 for i in range(N)]),  # Alternating
        np.array([1 if (i // 10) % 2 == 0 else -1 for i in range(N)]),  # Blocks of 10
    ]
    
    for init in initializations:
        if (time.time() - start_time) > max_time:
            break
        
        remaining = max_time - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        x, score = _try_improve(init.astype(float), max_time=remaining)
        
        if score < best_score:
            best_score = score
            best_x = x
    
    # If we didn't find anything, fall back to all ones
    if best_x is None:
        best_x = np.ones(N)
    
    # Ensure we return a sequence of signed step heights
    return best_x
# EVOLVE-BLOCK-END

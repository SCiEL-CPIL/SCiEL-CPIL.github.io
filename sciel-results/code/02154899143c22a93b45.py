"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_initial_autocorr(x):
    """Compute autocorrelation for lags 0..N-1."""
    auto = np.correlate(x, x, mode='full')
    # auto[N-1] is lag 0. auto[N-1+k] is lag k.
    return auto[N-1 : 2*N-1]

def _update_autocorr_after_flip(gamma, x, i):
    """
    Update gamma in-place when x[i] is flipped.
    gamma[k] = sum_{j=0}^{N-1-k} x[j] x[j+k]
    Delta gamma[k] = -2 * old_xi * (x[i+k] + x[i-k])
    """
    old_xi = x[i]
    
    # Vectorized update is complex with boundaries, loop is O(N) which is fast for N=400
    for k in range(N):
        val = 0
        if i + k < N:
            val += x[i + k]
        if i - k >= 0:
            val += x[i - k]
        gamma[k] += -2 * old_xi * val
        
    x[i] = -old_xi
    return gamma

def _get_max_sidelobe(gamma):
    if len(gamma) <= 1:
        return 0.0
    return np.max(np.abs(gamma[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_x = None
    best_score = float('inf')
    
    # Multiple restarts with random initialization
    # We use hill climbing with incremental updates
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random
        x = rng.choice([-1, 1], size=N)
        
        # Compute initial autocorrelation
        gamma = _compute_initial_autocorr(x)
        current_score = _get_max_sidelobe(gamma)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
        # Local Search (Hill Climbing)
        # We try to flip bits to reduce the max sidelobe
        improved = True
        while time.time() < deadline and improved:
            improved = False
            # Try flipping a random set of indices
            # To be efficient, we pick a few random candidates
            candidates = rng.choice(N, size=20, replace=False)
            
            for i in candidates:
                if time.time() > deadline:
                    break
                
                # Flip bit
                _update_autocorr_after_flip(gamma, x, i)
                new_score = _get_max_sidelobe(gamma)
                
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                    # Keep the flip
                else:
                    # Revert
                    _update_autocorr_after_flip(gamma, x, i)
                    
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
    return best_x

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_sidelobe(s):
    """Compute max |autocorrelation| for non-zero lags."""
    N = len(s)
    max_val = 0.0
    for k in range(1, N):
        c = np.dot(s[:N-k], s[k:])
        if abs(c) > max_val:
            max_val = abs(c)
    return max_val


def _autocorrelation_sidelobes(s):
    """Compute all non-zero lag autocorrelation values."""
    N = len(s)
    lags = np.arange(1, N)
    corrs = np.array([np.dot(s[:N-k], s[k:]) for k in lags])
    return corrs


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_s = None
    best_val = float('inf')
    
    np.random.seed(42)
    
    # Multiple restarts with local search
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial sequence
        s = np.random.choice([-1.0, 1.0], size=N)
        current_val = _max_sidelobe(s)
        
        if current_val < best_val:
            best_val = current_val
            best_s = s.copy()
        
        # Local search: try flipping each bit
        improved = True
        while improved:
            if time.time() > deadline:
                break
            
            improved = False
            # Try random subset of positions each iteration for speed
            n_trials = min(N, 200)
            indices = np.random.choice(N, size=n_trials, replace=False)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Flip bit i
                old = s[i]
                s[i] = -s[i]
                new_val = _max_sidelobe(s)
                
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_s = s.copy()
                else:
                    s[i] = old
        
        # If we found a very good solution, do more careful search
        if best_val < 20:
            # More thorough local search on best solution
            s = best_s.copy()
            current_val = best_val
            for _ in range(100):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(N):
                    if time.time() > deadline:
                        break
                    old = s[i]
                    s[i] = -s[i]
                    new_val = _max_sidelobe(s)
                    if new_val < current_val:
                        current_val = new_val
                        improved = True
                        if current_val < best_val:
                            best_val = current_val
                            best_s = s.copy()
                    else:
                        s[i] = old
                if not improved:
                    break
    
    if best_s is None:
        return np.ones(N)
    
    return best_s
# EVOLVE-BLOCK-END

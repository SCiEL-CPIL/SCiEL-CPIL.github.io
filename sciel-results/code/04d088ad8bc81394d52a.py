"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a signed step sequence (±1) that minimizes:
        max_k |autoconv[k]| / (sum(x))^2
    where autoconv[k] = sum_i x[i]*x[i+k]
    
    Uses simulated annealing with a deadline guard.
    """
    deadline = time.time() + 110  # 110s safety margin
    
    def compute_score(x):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return np.inf
        # Compute autoconvolution
        ac = np.correlate(x, x, mode='full')
        # autoconv at lag k corresponds to ac[N-1+k] for k=0..N-1
        # Actually, np.correlate(x, x, 'full') gives lags from -(N-1) to N-1
        # The positive lags (k>=0) are ac[N-1:]
        ac_pos = ac[N-1:]  # lags 0 to N-1
        max_abs_ac = np.max(np.abs(ac_pos))
        return max_abs_ac / (s * s)
    
    # Start with all ones (known to give 1/N)
    x = np.ones(N, dtype=np.float64)
    best_x = x.copy()
    best_score = compute_score(x)
    
    # Simulated annealing
    T = 1.0
    T_min = 1e-6
    cooling = 0.9995
    
    while time.time() < deadline:
        # Random flip
        idx = np.random.randint(N)
        x[idx] *= -1
        
        new_score = compute_score(x)
        delta = new_score - best_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            best_x = x.copy()
            best_score = new_score
        else:
            x[idx] *= -1  # revert
        
        T = max(T * cooling, T_min)
        
        # Periodic restart if stuck
        if T < T_min:
            # Try a random restart with bias toward positive sum
            x = np.ones(N, dtype=np.float64)
            # Flip some fraction randomly
            n_flips = np.random.randint(0, N//4)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            x[flip_idx] *= -1
            best_x = x.copy()
            best_score = compute_score(x)
            T = 1.0
    
    return best_x

# EVOLVE-BLOCK-END

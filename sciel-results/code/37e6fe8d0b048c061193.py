import numpy as np
import time

N = 400

def _compute_autoconv(s):
    """Compute autoconvolution of s."""
    n = len(s)
    ac = np.zeros(n)
    for k in range(n):
        ac[k] = np.dot(s[:n-k], s[k:])
    return ac

def _score(s):
    """Compute the objective: max|autoconv| / (sum s)^2."""
    S = np.sum(s)
    if S == 0:
        return np.inf
    ac = _compute_autoconv(s)
    return np.max(np.abs(ac)) / (S * S)

def construct():
    deadline = time.time() + 115
    best_s = np.ones(N, dtype=np.float64)
    best_val = _score(best_s)
    
    # Incremental autoconv
    ac = _compute_autoconv(best_s)
    S = np.sum(best_s)
    
    # Local search: try flipping each element
    for iteration in range(1000000):
        if time.time() > deadline:
            break
        
        improved = False
        for j in range(N):
            if time.time() > deadline:
                break
            
            # Try flipping s[j]
            delta = -2.0 * best_s[j]
            new_S = S + delta
            if new_S == 0:
                continue
            
            # Compute change in autoconv
            # delta_ac[k] = delta * (s[j+k] if j+k<N else 0 + s[j-k] if j-k>=0 else 0)
            new_ac = ac.copy()
            for k in range(N):
                contrib = 0.0
                if j + k < N:
                    contrib += best_s[j + k]
                if j - k >= 0:
                    contrib += best_s[j - k]
                new_ac[k] = ac[k] + delta * contrib
            
            new_val = np.max(np.abs(new_ac)) / (new_S * new_S)
            
            if new_val < best_val - 1e-12:
                # Accept the flip
                best_s[j] += delta
                S = new_S
                ac = new_ac
                best_val = new_val
                improved = True
        
        if not improved:
            # Try random restarts or accept small changes
            # Try a few random flips to escape local minima
            for _ in range(10):
                if time.time() > deadline:
                    break
                j = np.random.randint(N)
                delta = -2.0 * best_s[j]
                new_S = S + delta
                if new_S == 0:
                    continue
                new_ac = ac.copy()
                for k in range(N):
                    contrib = 0.0
                    if j + k < N:
                        contrib += best_s[j + k]
                    if j - k >= 0:
                        contrib += best_s[j - k]
                    new_ac[k] = ac[k] + delta * contrib
                new_val = np.max(np.abs(new_ac)) / (new_S * new_S)
                if new_val < best_val + 0.01 * best_val:  # Allow some worsening
                    best_s[j] += delta
                    S = new_S
                    ac = new_ac
                    best_val = new_val
    
    return best_s

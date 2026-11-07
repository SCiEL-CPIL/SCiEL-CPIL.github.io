import time
import numpy as np

N = 100

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(100):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            # Need to decrease sum. Decrease elements that are > 0.
            mask = h > 1e-12
            if not np.any(mask):
                break
            # Decrease proportionally to current value to keep them in [0,1]
            # Or just distribute evenly among those > 0
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # Need to increase sum. Increase elements that are < 1.
            mask = h < 1.0 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _smooth_max_objective(h, gamma=100.0):
    """
    Differentiable approximation of max(corr).
    Uses softmax of the correlation values.
    """
    corr = np.correlate(h, 1 - h, mode="full")
    # Softmax approximation
    # max(x) ~ (1/gamma) * log(sum(exp(gamma * x)))
    # To avoid overflow, subtract max
    m = np.max(corr)
    return m + (1.0 / gamma) * np.log(np.sum(np.exp(gamma * (corr - m))))

def _local_polish(h, time_limit):
    """
    Fast local search on the continuous domain.
    Uses gradient-free pair moves.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    start_time = time.time()
    
    # Precompute correlation for speed? No, N=100 is small.
    # But we can speed up the evaluation by only checking if it improves.
    
    while time.time() < time_limit:
        improved = False
        
        # Compute current correlation
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        max_val = corr[max_idx]
        
        # Identify the lag corresponding to max_idx
        # mode="full" length is 2N-1. Index max_idx corresponds to lag max_idx - (N-1)
        lag = max_idx - (N - 1)
        
        # We want to reduce the correlation at this lag.
        # corr_lag = sum_{i} h_i * (1 - h_{i+lag})  (with appropriate bounds)
        # Actually np.correlate(a, v, mode='full')[k] = sum a[n] * v[n-k] ?
        # Let's just use the gradient intuition:
        # We want to move mass from places where it contributes to high correlation
        # to places where it doesn't.
        
        # Simple strategy: random pair swap with hill climbing.
        # Pick random i, j.
        # Try to move delta from i to j or j to i.
        
        # To make it faster, let's try a few random moves per iteration.
        for _ in range(50):
            if time.time() > time_limit:
                return best_h
            
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            # Try moving mass from i to j
            max_move = min(best_h[i], 1.0 - best_h[j])
            if max_move < 1e-6:
                continue
            
            # Try a fraction of the max move
            # Use a small step size or full move?
            # Let's try a random fraction
            frac = np.random.uniform(0.1, 1.0)
            d = max_move * frac
            
            new_h = best_h.copy()
            new_h[i] -= d
            new_h[j] += d
            
            # Check validity (should be guaranteed by construction, but check just in case)
            if np.any(new_h < -1e-9) or np.any(new_h > 1 + 1e-9):
                continue
            
            val = _correlation_value(new_h)
            
            if val < best_val:
                best_val = val
                best_h = new_h
                improved = True
        
        if not improved:
            # If we didn't improve in 50 random tries, try a larger kick or break?
            # Let's try a slightly larger random kick to escape local minima
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i != j:
                d = 0.1 * np.random.choice([-1, 1])
                new_h = best_h.copy()
                new_h[i] += d
                new_h[j] -= d
                if np.all((new_h >= -1e-9) & (new_h <= 1 + 1e-9)):
                    new_h = np.clip(new_h, 0, 1)
                    # Re-project to maintain sum?
                    # The random kick might break the sum constraint.
                    # Let's re-project.
                    new_h = _project_to_feasible(new_h)
                    val = _correlation_value(new_h)
                    if val < best_val:
                        best_val = val
                        best_h = new_h

    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial seed
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Strategy:
    # 1. Use a binary-like initialization to break symmetry.
    # The problem suggests binary permutations are good.
    # Let's start with a random binary permutation of 50 ones and 50 zeros.
    
    # Generate random binary sequence
    h_bin = np.zeros(N)
    h_bin[:50] = 1.0
    np.random.shuffle(h_bin)
    
    # Project (sum is already 50, bounds are satisfied)
    h_bin = _project_to_feasible(h_bin)
    val_bin = _correlation_value(h_bin)
    
    if val_bin < best_val:
        best_val = val_bin
        best_h = h_bin.copy()
    
    # 2. Local polish from the binary start
    t_limit = deadline - 5.0
    best_h = _local_polish(best_h, t_limit)
    best_val = _correlation_value(best_h)
    
    # 3. Try a few more random binary starts if time permits
    # Since local polish is the bottleneck, let's see how much time is left.
    # If we have > 10s left, try another restart.
    
    if time.time() < deadline - 10.0:
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            h_bin = np.zeros(N)
            h_bin[:50] = 1.0
            np.random.shuffle(h_bin)
            h_bin = _project_to_feasible(h_bin)
            
            # Quick polish
            t_limit_restart = min(deadline - 5.0, time.time() + 5.0)
            h_new = _local_polish(h_bin, t_limit_restart)
            val_new = _correlation_value(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
    
    # Final projection to ensure constraints are met strictly
    best_h = _project_to_feasible(best_h)
    
    return best_h

import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(50):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _local_search_binary(h, max_iters=1000, deadline=None):
    """
    Local search on binary sequence by flipping bits to reduce max correlation.
    """
    h = (h > 0.5).astype(float)
    # Ensure sum is exactly 50
    ones = np.sum(h)
    if ones != 50:
        # Adjust to have exactly 50 ones
        if ones > 50:
            idx = np.where(h == 1)[0]
            np.random.shuffle(idx)
            for i in idx[:int(ones - 50)]:
                h[i] = 0
        else:
            idx = np.where(h == 0)[0]
            np.random.shuffle(idx)
            for i in idx[:int(50 - ones)]:
                h[i] = 1
                
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Find the lag with maximum correlation
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_lag = np.argmax(corr)
        
        # The correlation at max_lag involves pairs (i, i+max_lag)
        # We want to reduce this.
        # Try flipping a bit involved in the high correlation.
        # If h[i]=1 and h[i+lag]=0, flipping h[i] to 0 reduces the term.
        # If h[i]=0 and h[i+lag]=1, flipping h[i] to 1 reduces the term (becomes 0).
        
        # Get indices contributing to the max correlation
        # corr[k] = sum_{i} h[i] (1-h[i+k])
        # For a given lag k, the terms are h[i] where h[i+k]=0.
        
        # Simple heuristic: pick a random index i, flip it, check if max decreases.
        # To be more targeted, pick i from the range of the max lag.
        
        improved = False
        for _ in range(20): # Try a few random flips
            if deadline is not None and time.time() > deadline:
                break
            i = np.random.randint(N)
            new_h = best_h.copy()
            new_h[i] = 1.0 - new_h[i]
            
            # Re-evaluate
            val = _correlation_value(new_h)
            if val < best_val - 1e-12:
                best_val = val
                best_h = new_h
                improved = True
                break
                
        if not improved:
            # Try a pair swap to maintain sum? No, flipping changes sum.
            # We need to maintain sum=50.
            # So we must swap a 1 and a 0.
            ones = np.where(best_h == 1)[0]
            zeros = np.where(best_h == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                i = np.random.choice(ones)
                j = np.random.choice(zeros)
                new_h = best_h.copy()
                new_h[i] = 0
                new_h[j] = 1
                val = _correlation_value(new_h)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = new_h
                    improved = True
                    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Binary Local Search from multiple random starts
    rng = np.random.default_rng(42)
    
    for start_idx in range(50):
        if time.time() > deadline - 60:
            break
            
        # Random binary start
        h_rand = (rng.random(N) > 0.5).astype(float)
        h_rand, val_rand = _local_search_binary(h_rand, max_iters=200, deadline=deadline - 60)
        
        if val_rand < best_val:
            best_val = val_rand
            best_h = h_rand.copy()
            
    # Phase 2: Continuous Optimization
    # Use the best binary solution as a start for continuous optimization
    # We can also try starting from 0.5
    
    starts = [best_h.copy(), np.full(N, 0.5)]
    
    # Add some structured starts
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(h_alt)
    
    for h_start in starts:
        if time.time() > deadline - 10:
            break
            
        h_start = _project_to_feasible(h_start)
        
        def obj(x):
            return _correlation_value(x)
        
        def grad(x):
            # Numerical gradient is expensive, let SLSQP estimate it
            return None
            
        try:
            res = minimize(
                obj,
                h_start,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Final Polish with Local Search on Continuous Values
    # Try small perturbations
    if time.time() < deadline - 5:
        current_h = best_h.copy()
        current_val = best_val
        step_size = 0.05
        for _ in range(1000):
            if time.time() > deadline - 1:
                break
            i, j = rng.choice(N, size=2, replace=False)
            # Move mass from j to i
            d = step_size * rng.uniform(0.1, 1.0)
            if current_h[i] + d <= 1.0 and current_h[j] - d >= 0.0:
                new_h = current_h.copy()
                new_h[i] += d
                new_h[j] -= d
                val = _correlation_value(new_h)
                if val < current_val - 1e-12:
                    current_val = val
                    current_h = new_h
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
                        
    return best_h

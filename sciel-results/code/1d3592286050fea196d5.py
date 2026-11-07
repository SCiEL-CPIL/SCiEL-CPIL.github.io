import time
import numpy as np
from scipy.optimize import minimize
import sys

N = 100
DEADLINE_OFFSET = 145.0  # Safety margin for 150s limit
CORR_LEN = 2 * N - 1  # Length of full correlation

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.array(h, dtype=float)
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    
    # Simple iterative projection
    for _ in range(50):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            # Need to decrease sum
            mask = h > 1e-9
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # Need to increase sum
            mask = h < 1 - 1e-9
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _generate_starts():
    """Generate a diverse set of starting candidates."""
    starts = []
    rng = np.random.default_rng(100)
    
    # 1. Seed
    starts.append(np.full(N, 0.5))
    
    # 2. Random uniform
    for _ in range(30):
        h = rng.uniform(0, 1, N)
        starts.append(_project_to_feasible(h))
        
    # 3. Sine waves (various frequencies and phases)
    for freq in range(1, 15):
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            starts.append(_project_to_feasible(h))
            
    # 4. Cosine waves
    for freq in range(1, 15):
        for phase in [0, np.pi/4, np.pi/2]:
            h = 0.5 + 0.4 * np.cos(2 * np.pi * freq * np.arange(N) / N + phase)
            starts.append(_project_to_feasible(h))
            
    # 5. Square waves / Blocks
    for block_size in range(2, 11):
        h = np.zeros(N)
        for i in range(N):
            h[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h))
        
    # 6. Sawtooth
    for period in range(5, 21, 5):
        h = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            h[i] = t
        starts.append(_project_to_feasible(h))
        
    # 7. High frequency noise
    for _ in range(10):
        h = rng.normal(0.5, 0.2, N)
        starts.append(_project_to_feasible(h))
        
    # 8. Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    return starts

def _soft_max(corr, beta=100.0):
    """
    Compute a smooth approximation of max(corr) using log-sum-exp.
    beta controls the sharpness: larger beta -> closer to true max.
    """
    # Numerically stable log-sum-exp
    c_max = np.max(corr)
    if c_max > 0:
        log_sum = c_max + np.log(np.sum(np.exp(beta * (corr - c_max))))
        return (log_sum / beta)
    else:
        # If all values are very negative, just return the max
        return c_max

def _optimize_candidate(h_start, time_budget=0.5):
    """Optimize a single candidate using SLSQP with epigraph formulation."""
    start_time = time.time()
    h_start = _project_to_feasible(h_start)
    
    # Pre-compute correlation structure for efficiency
    # corr[k] = sum_i h[i] * (1-h[i+k-N+1]) for k=0..2N-2
    # We'll use the soft-max approximation of max(corr)
    
    def objective(x):
        """
        x = [h_0, ..., h_{N-1}, t]
        Minimize t subject to:
        - soft_max(corr(h)) <= t (approximated)
        - sum(h) = N/2
        - 0 <= h_i <= 1
        """
        h = x[:N]
        t = x[N]
        
        # Compute correlation
        corr = np.correlate(h, 1 - h, mode="full")
        
        # Use soft-max as smooth surrogate for max
        smax = _soft_max(corr, beta=50.0)
        
        # Objective: minimize t + penalty for violating soft_max <= t
        # We use a barrier-like approach: t + max(0, smax - t)^2
        violation = max(0, smax - t)
        return t + 100.0 * violation * violation
    
    def constraint_sum(x):
        return np.sum(x[:N]) - N / 2.0
    
    try:
        # Initial t: estimate from current h
        corr_init = np.correlate(h_start, 1 - h_start, mode="full")
        t_init = np.max(corr_init) + 0.1
        
        x0 = np.concatenate([h_start, [t_init]])
        
        # Bounds: h in [0,1], t in [0, 2] (max possible corr is N, so t up to ~2*N)
        bounds = [(0.0, 1.0) for _ in range(N)] + [(0.0, 2.0)]
        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if res.success and res.fun is not None:
            h_opt = _project_to_feasible(res.x[:N])
            val = _correlation_value(h_opt)
            return h_opt, val
        else:
            return h_start, _correlation_value(h_start)
            
    except Exception:
        return h_start, _correlation_value(h_start)

def _local_polish(h, max_iters=100):
    """
    Coordinate descent polish with gradient-aware step selection.
    """
    best_h = _project_to_feasible(h)
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    for it in range(max_iters):
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        current_max_val = np.max(corr)
        
        improved = False
        
        # Find which h indices contribute to the max correlation
        # corr[max_idx] = sum_i h[i] * (1-h[i + max_idx - (N-1)])
        # Let's try adjusting elements near the peak
        
        # Get the indices that contribute to corr[max_idx]
        offset = max_idx - (N - 1)
        # h[i] multiplies (1-h[i+offset]) for i where both are in range
        valid_i = []
        for i in range(N):
            j = i + offset
            if 0 <= j < N:
                valid_i.append(i)
        
        if not valid_i:
            valid_i = list(range(n))
        
        # Try adjusting elements that contribute to the peak
        indices_to_try = valid_i[:min(15, len(valid_i))]
        
        for i in indices_to_try:
            # Try small adjustments
            for delta in [-0.005, -0.01, -0.02, 0.005, 0.01, 0.02]:
                new_h = best_h.copy()
                new_h[i] += delta
                if new_h[i] < 0 or new_h[i] > 1:
                    continue
                
                # Find a compensating element j to keep sum
                # Prefer j that doesn't affect the peak correlation much
                candidates = [x for x in range(n) if x != i]
                if not candidates:
                    continue
                
                # Score candidates by how much they affect the peak
                best_j = None
                best_score = float('inf')
                for j in candidates:
                    new_h[j] -= delta
                    if new_h[j] < 0 or new_h[j] > 1:
                        continue
                    
                    # Quick estimate: compute correlation change
                    # Just use random for speed
                    new_h_proj = _project_to_feasible(new_h)
                    val = _correlation_value(new_h_proj)
                    if val < best_score:
                        best_score = val
                        best_j = j
                        best_h_candidate = new_h_proj.copy()
                
                if best_j is not None and best_score < best_val - 1e-10:
                    best_val = best_score
                    best_h = best_h_candidate
                    improved = True
                    break
            if improved:
                break
                
        if not improved:
            break
            
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_OFFSET
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    starts = _generate_starts()
    
    # Evaluate initial values to sort
    start_vals = []
    for h in starts:
        v = _correlation_value(h)
        start_vals.append((v, h))
        
    start_vals.sort(key=lambda x: x[0])
    
    # Process candidates with time guard
    for val, h in start_vals:
        if time.time() > deadline - 15.0:
            break
        
        h_opt, val_opt = _optimize_candidate(h)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
            
    # Final polish pass
    if time.time() < deadline - 5.0:
        try:
            h_polished, val_polished = _local_polish(best_h, max_iters=100)
            if val_polished < best_val:
                best_val = val_polished
                best_h = h_polished
        except Exception:
            pass
            
    # Final feasibility check
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
        
    return best_h

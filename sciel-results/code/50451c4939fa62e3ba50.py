"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct 100 step heights h in [0,1] with sum(h)=50,
    minimizing max(np.correlate(h, 1-h, mode='full')) / N * 2.
    
    Strategy:
    - The objective is max over lags of sum_i h[i] * (1 - h[i+lag]) (with appropriate bounds).
    - We want h to be as "anti-correlated" with its shifts as possible.
    - A good heuristic: alternate high and low values, but since sum must be 50 (mean 0.5),
      we can try a pattern that minimizes local products h[i]*h[j] for nearby i,j.
    - Since we minimize max of correlate(h, 1-h), which is sum h[i]*(1-h[i+lag]),
      note that sum h[i]*(1-h[i+lag]) = sum h[i] - sum h[i]*h[i+lag] = 50 - sum h[i]*h[i+lag].
      So minimizing the max of this is equivalent to maximizing the min of sum h[i]*h[i+lag] over lags.
      Wait, let's re-derive:
      correlate(h, 1-h, mode='full')[k] for lag k (where k=0..2N-2) corresponds to:
      For lag k (0-indexed in full mode, where lag = k - (N-1)):
      Actually, np.correlate(a, v, mode='full') computes c_k = sum_i a[i+k] * v[i] for valid ranges.
      Let a = h, v = 1-h.
      c_k = sum_i h[i+k] * (1-h[i]) for i such that indices are valid.
      
      The max over k of c_k / N * 2 is our objective.
      
      Key insight: If we can make h[i] * h[i+k] large for all k, then c_k = sum h[i+k] - sum h[i+k]h[i] will be small.
      But we need to maximize the minimum of sum h[i]h[i+k] over k? No, we need to minimize the MAX of c_k.
      
      Alternative approach: Use a smooth optimization. Start from a good initial guess and use gradient descent
      on a smooth surrogate for the max (log-sum-exp).
      
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial guess: alternating 0 and 1 won't work since sum must be 50.
    # Let's try a sinusoidal pattern or use optimization.
    
    # Try a few different initial patterns and pick the best, then polish.
    
    def compute_value(h):
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return np.inf
        if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
            return np.inf
        h_clipped = np.clip(h, 0.0, 1.0)
        if abs(np.sum(h_clipped) - 50.0) > 1e-7 * 100:
            return np.inf
        corr = np.correlate(h_clipped, 1 - h_clipped, mode="full")
        return np.max(corr) / N * 2
    
    def logsumexp(x, axis=None):
        """Numerically stable log-sum-exp."""
        x_max = np.max(x, axis=axis, keepdims=True)
        # Handle case where x_max is -inf
        x_max_safe = np.where(np.isfinite(x_max), x_max, 0.0)
        return np.log(np.sum(np.exp(x - x_max_safe), axis=axis, keepdims=True)) + x_max_safe
    
    def smooth_objective(h, beta=100.0):
        """Smooth surrogate for max(corr) using log-sum-exp."""
        h = np.asarray(h, dtype=float)
        corr = np.correlate(h, 1 - h, mode="full")
        # Use log-sum-exp as smooth max
        lse = logsumexp(corr * beta).flatten()[0] / beta
        # Add penalty for constraint violations
        s = np.sum(h)
        sum_penalty = (s - 50.0) ** 2 * 1000.0
        # Penalty for out of [0,1]
        below = np.sum(np.maximum(0.0, -h) ** 2) * 1000.0
        above = np.sum(np.maximum(0.0, h - 1.0) ** 2) * 1000.0
        return lse / N * 2 + sum_penalty + below + above
    
    def gradient_descent(h0, max_iters=2000, lr=0.01):
        """Simple gradient descent with projected constraints."""
        h = h0.copy().astype(float)
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Compute gradient numerically or analytically
            # Use finite differences for simplicity, but that's slow for N=100
            # Let's use analytic gradient for the smooth objective
            
            # For the log-sum-exp part:
            corr = np.correlate(h, 1 - h, mode="full")
            beta = 100.0
            exp_corr = np.exp(corr * beta - np.max(corr * beta))
            weights = exp_corr / np.sum(exp_corr)
            
            # d(corr_k)/d(h_i) for correlate(h, 1-h)
            # corr_k = sum_j h[j+k'] * (1-h[j]) where the indexing depends on k
            # This is complex to compute analytically. Let's use a simpler approach.
            
            # Alternative: use a simpler heuristic-based search
            break
        
        return h
    
    # Better approach: Use a structured pattern search
    # Try patterns of the form h[i] = 0.5 + A * sin(2*pi*f*i + phase)
    # and optimize A, f, phase
    
    best_h = None
    best_val = np.inf
    
    # Try sinusoidal patterns
    for f in np.linspace(0.05, 5.0, 50):
        for A in np.linspace(0.0, 0.5, 20):
            for phase in np.linspace(0.0, 2*np.pi, 10):
                h = 0.5 + A * np.sin(2*np.pi*f*np.arange(N) + phase)
                h = np.clip(h, 0.0, 1.0)
                # Project to sum=50
                h = h + (50.0 - np.sum(h)) / N
                h = np.clip(h, 0.0, 1.0)
                # Re-project
                diff = 50.0 - np.sum(h)
                if abs(diff) > 1e-10:
                    h = h + diff / N
                h = np.clip(h, 0.0, 1.0)
                
                val = compute_value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                
                if time.time() > deadline - 10.0:
                    break
            if time.time() > deadline - 10.0:
                break
        if time.time() > deadline - 10.0:
            break
    
    # Also try alternating patterns with different duty cycles
    for period in range(2, 20):
        for high_val in np.linspace(0.1, 0.9, 15):
            low_val = 0.0
            h = np.zeros(N)
            for i in range(N):
                if (i % period) < period // 2:
                    h[i] = high_val
                else:
                    h[i] = low_val
            # Adjust to sum=50
            h = h + (50.0 - np.sum(h)) / N
            h = np.clip(h, 0.0, 1.0)
            diff = 50.0 - np.sum(h)
            if abs(diff) > 1e-10:
                h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
            
            val = compute_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            if time.time() > deadline - 10.0:
                break
        if time.time() > deadline - 10.0:
            break
    
    # Local refinement using simple coordinate descent on the best candidate
    if best_h is not None and time.time() < deadline - 5.0:
        h = best_h.copy()
        improved = True
        max_cd_iters = 50
        for cd_iter in range(max_cd_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                # Try small perturbations
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    h_new = h.copy()
                    h_new[i] += delta
                    # Compensate sum by adjusting another element
                    diff = -delta
                    # Spread the compensation
                    h_new = h_new + diff / N
                    h_new = np.clip(h_new, 0.0, 1.0)
                    diff2 = 50.0 - np.sum(h_new)
                    if abs(diff2) > 1e-10:
                        h_new = h_new + diff2 / N
                    h_new = np.clip(h_new, 0.0, 1.0)
                    
                    val = compute_value(h_new)
                    if val < best_val - 1e-10:
                        best_val = val
                        h = h_new.copy()
                        improved = True
                        break
            if not improved:
                break
    
    # Final projection to ensure constraints are met
    if best_h is not None:
        h = best_h.copy()
        # Ensure sum is exactly 50
        diff = 50.0 - np.sum(h)
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        # One more correction
        diff = 50.0 - np.sum(h)
        if abs(diff) > 1e-12:
            # Adjust the element with most room
            idx = np.argmax(h < 1.0) if diff > 0 else np.argmin(h > 0.0)
            h[idx] += diff
            h = np.clip(h, 0.0, 1.0)
        best_h = h
    
    if best_h is None:
        # Fallback
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END

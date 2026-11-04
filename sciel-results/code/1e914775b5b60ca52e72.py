"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_corr(h):
    """
    Compute non-zero autocorrelation coefficients c[1..N-1].
    c[k] = sum_{j=0}^{N-1-k} h[j] h[j+k]
    Returns an array of length N, with c[0] = 0.
    """
    c = np.zeros(N)
    for k in range(1, N):
        c[k] = np.dot(h[:N-k], h[k:])
    return c

def _score_from_corr(c):
    """
    Compute sum of squares of non-zero autocorrelations.
    """
    return np.sum(c[1:]**2)

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Try multiple random starts
    rng = np.random.RandomState(42)
    
    for seed in range(50):
        if time.time() - start_time > deadline:
            break
            
        h = rng.choice([-1.0, 1.0], size=N)
        c = _compute_corr(h)
        current_score = _score_from_corr(c)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # Local search
        improved = True
        while improved and (time.time() - start_time < deadline):
            improved = False
            best_i = -1
            best_delta = 0.0
            best_delta_c = None
            
            # Evaluate all possible flips
            for i in range(N):
                # Determine range of k affected
                # k must satisfy 1 <= k < N-i and 1 <= k <= i
                # So k is in [1, min(i, N-1-i)]
                max_k = min(i, N - 1 - i)
                if max_k < 1:
                    continue
                
                # Calculate delta_c for k in 1..max_k
                # Delta c[k] = -2 * h[i] * (h[i+k] + h[i-k])
                # Note: h[i] is the value BEFORE flip
                hi = h[i]
                # Vectorized calculation for speed
                # Indices for h[i+k]: i+1 ... i+max_k
                # Indices for h[i-k]: i-1 ... i-max_k
                # Be careful with boundaries
                k_range = np.arange(1, max_k + 1)
                idx_plus = i + k_range
                idx_minus = i - k_range
                
                # Ensure indices are valid (they are by definition of max_k)
                delta_c_vals = -2.0 * hi * (h[idx_plus] + h[idx_minus])
                
                # Calculate change in score
                # J' - J = 2 * sum(c[k] * delta_c[k]) + sum(delta_c[k]^2)
                # c is the current correlation
                c_subset = c[1:max_k+1]
                
                dot_prod = np.dot(c_subset, delta_c_vals)
                sq_sum = np.dot(delta_c_vals, delta_c_vals)
                
                delta_J = 2.0 * dot_prod + sq_sum
                
                if delta_J > best_delta:
                    best_delta = delta_J
                    best_i = i
                    best_delta_c = delta_c_vals
                    best_max_k = max_k
            
            if best_i != -1 and best_delta > 1e-9:
                # Apply flip
                h[best_i] *= -1.0
                # Update c
                c[1:best_max_k+1] += best_delta_c
                current_score += best_delta
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Local minimum reached
                # Try a small random perturbation to escape
                if time.time() - start_time < deadline - 5:
                    num_flips = rng.randint(1, 10)
                    flip_indices = rng.choice(N, size=num_flips, replace=False)
                    h[flip_indices] *= -1.0
                    # Recompute c and score from scratch to avoid accumulation errors
                    c = _compute_corr(h)
                    current_score = _score_from_corr(c)
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
                    # Continue local search from this new point
                    improved = True # Force another pass

    return best_h
# EVOLVE-BLOCK-END

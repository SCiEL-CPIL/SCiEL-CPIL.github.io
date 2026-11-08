import time
import numpy as np

N = 400

def _evaluate(h):
    c = np.convolve(h, h, mode='full')
    max_abs = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    s2 = s * s
    if s2 < 1e-12:
        return max_abs
    return max_abs / s2

def _flip_delta_binary(c, h, idx, N):
    """Compute delta in autoconvolution when flipping h[idx] (binary ±1)."""
    s = h[idx]  # ±1
    delta = np.zeros(2 * N - 1)
    
    # For binary sequences, flipping sign changes h[idx] from s to -s
    # Delta in convolution: for each pair (i,j) where i==idx or j==idx
    # c_new[k] = c[k] - 2*s*h[idx+k-(N-1)] for appropriate ranges
    
    # Contribution from h[idx] * h[j] for all j
    # In full convolution, c[k] = sum_i h[i]*h[k-i]
    # When we flip h[idx], the change is:
    # -2*s * sum over j of h[j] at positions where idx+j = k or j+idx = k
    
    # Position idx contributes to c at lags: for each j, position idx+j
    # In full conv indexing: c_full[p] where p ranges 0..2N-2
    # h[i]*h[j] contributes to c_full[i+j]
    
    # Flipping h[idx]: change = -2*s*h[idx] * (sum of h[j] at positions where i+j involves idx)
    # Actually: new_c[p] = old_c[p] - 2*s*(h[idx]*h[p-idx] + h[p-idx]*h[idx] - 2*h[idx]*h[idx] if p-idx==idx)
    # Simpler: for each p, the terms involving h[idx] are h[idx]*h[p-idx] and h[p-idx]*h[idx]
    # If p-idx == idx, it's h[idx]^2 which becomes h[idx]^2 (no change since (-s)^2 = s^2)
    # Otherwise, h[idx]*h[p-idx] becomes -h[idx]*h[p-idx], change = -2*h[idx]*h[p-idx]
    
    # Range of p where h[p-idx] is valid: 0 <= p-idx <= N-1, so idx <= p <= idx+N-1
    p_start = idx
    p_end = min(2*N-2, idx + N - 1)
    
    if p_start <= p_end:
        j_vals = np.arange(p_start, p_end + 1) - idx  # j values: 0..N-1
        # Exclude j == idx (self-term doesn't change)
        mask = (j_vals != idx)
        if np.any(mask):
            delta[p_start:p_end+1][mask] = -2 * s * h[j_vals[mask]]
    
    # Also need to handle the symmetric part: h[p-idx]*h[idx] where p-idx != idx
    # This is already covered above since convolution is symmetric in the sum
    # Actually no - we need both h[idx]*h[j] AND h[j]*h[idx] for j != idx
    # But in the sum c[p] = sum_i h[i]*h[p-i], the term with i=idx is h[idx]*h[p-idx]
    # and the term with i=p-idx is h[p-idx]*h[idx] - these are the SAME term counted once
    # Wait, no. c[p] = sum_{i=0}^{N-1} h[i]*h[p-i]
    # The term where i=idx: h[idx]*h[p-idx]
    # The term where i=p-idx: h[p-idx]*h[idx] - but this is the same as above only if we're summing over all i
    # Actually in the full sum, each pair (i, p-i) is counted once. So flipping h[idx] only
    # affects terms where i=idx OR p-i=idx, i.e., i=idx or i=p-idx.
    # If idx != p-idx, there are TWO terms: h[idx]*h[p-idx] and h[p-idx]*h[idx]
    # Each changes by -2*s*h[other], so total change = -4*s*h[other]
    # If idx == p-idx, one term h[idx]^2 which doesn't change.
    
    # Let me redo this more carefully.
    # c[p] = sum_i h[i] * h[p-i]
    # After flip: c'[p] = sum_i h'[i] * h[p-i] where h'[idx] = -h[idx]
    # c'[p] - c[p] = h'[idx]*h[p-idx] + h[p-idx]*h'[idx] - h[idx]*h[p-idx] - h[p-idx]*h[idx]
    # Wait, this double counts. Let me be precise.
    
    # c[p] = sum_{i=0}^{N-1} h[i] * h[p-i] (with h out of range = 0)
    # The only term that changes is when i=idx: h[idx]*h[p-idx] -> (-h[idx])*h[p-idx]
    # AND when p-i=idx, i.e., i=p-idx: h[p-idx]*h[idx] -> h[p-idx]*(-h[idx])
    # If idx != p-idx: two terms change, each by -2*h[idx]*h[p-idx], total -4*h[idx]*h[p-idx]
    # If idx == p-idx: one term h[idx]^2 -> h[idx]^2, no change
    
    # So: for p where 0 <= p-idx <= N-1 and p-idx != idx:
    #   delta[p] = -4 * s * h[p-idx]
    # For p where p-idx == idx: delta[p] = 0
    
    delta = np.zeros(2 * N - 1)
    p_start = idx
    p_end = min(2*N-2, idx + N - 1)
    if p_start <= p_end:
        j_vals = np.arange(p_start, p_end + 1) - idx
        mask = (j_vals != idx)
        if np.any(mask):
            delta[p_start:p_end+1][mask] = -4 * s * h[j_vals[mask]]
    
    return delta

def _hill_climb_binary(h, deadline):
    """Coordinate descent on binary ±1 sequence."""
    c = np.convolve(h, h, mode='full')
    current_max = float(np.max(np.abs(c)))
    current_sum = float(np.sum(h))
    current_s2 = current_sum * current_sum
    current_val = current_max / current_s2 if current_s2 > 1e-12 else current_max
    
    rng = np.random.default_rng(42)
    stall_count = 0
    
    while time.time() < deadline:
        idx = rng.integers(0, N)
        delta = _flip_delta_binary(c, h, idx, N)
        c_new = c + delta
        
        new_max = float(np.max(np.abs(c_new)))
        new_sum = current_sum - 2 * h[idx]
        new_s2 = new_sum * new_sum
        new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max
        
        if new_val < current_val:
            h[idx] = -h[idx]
            c = c_new
            current_max = new_max
            current_sum = new_sum
            current_s2 = new_s2
            current_val = new_val
            stall_count = 0
        else:
            stall_count += 1
    
    return h, current_val, c

def _build_random_binary(n, rng):
    """Generate random ±1 sequence."""
    return rng.choice([-1.0, 1.0], size=n)

def construct():
    deadline = 135.0  # Safety margin
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(123)
    
    # Multiple random restarts with coordinate descent
    n_restarts = 8
    for i in range(n_restarts):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time per restart
        per_restart = remaining / (n_restarts - i)
        alloc_deadline = time.time() + per_restart - 0.5
        
        h = _build_random_binary(N, rng)
        h_opt, val, _ = _hill_climb_binary(h.copy(), alloc_deadline)
        
        if val < best_val:
            best_val = val
            best_h = h_opt
    
    return best_h

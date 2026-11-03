"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Simulated Annealing with stochastic local moves.
This avoids gradient computation and uses random perturbations to explore the landscape,
potentially escaping local minima that trap gradient-based methods.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115  # Safety margin before 120s kill


def _compute_score(h):
    """
    Compute the maximum linear overlap.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # Initial guess: Cosine wave
    i = np.arange(N)
    h_init = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_init = _fix_sum(h_init)
    
    _update_best(h_init)
    
    # Simulated Annealing parameters
    T_start = 0.05
    T_end = 1e-6
    n_iterations = 50000
    
    # We will run multiple short SA runs or one long one.
    # Given N=100, one long run is better.
    
    h = h_init.copy()
    curr_score = best_score
    
    for it in range(n_iterations):
        if time.time() - t0 > _DEADLINE - 5:
            break
            
        # Annealing schedule: geometric decay
        progress = it / n_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Generate a random move: pick two distinct indices i, j
        # and shift delta from i to j
        # To ensure we explore well, we can pick i, j uniformly.
        # Delta magnitude should be adaptive or fixed small.
        # Let's use a random delta from a uniform distribution scaled by T or fixed.
        # Fixed small delta is safer for constraints.
        
        i_idx = np.random.randint(0, N)
        j_idx = np.random.randint(0, N)
        if i_idx == j_idx:
            continue
            
        # Try to move mass from i to j
        # Max possible shift is min(h[i], 1-h[j])
        max_shift = min(h[i_idx], 1.0 - h[j_idx])
        if max_shift <= 1e-12:
            continue
            
        # Random shift amount, biased towards smaller shifts?
        # Or uniform? Uniform in [0, max_shift] might be too large.
        # Let's use a random sign and magnitude.
        # Actually, just pick a small random delta.
        delta = np.random.uniform(0.0, 0.1) * (1.0 if np.random.rand() < 0.5 else -1.0)
        
        # Check if move is valid
        new_hi = h[i_idx] + delta
        new_hj = h[j_idx] - delta
        
        if new_hi < 0 or new_hi > 1 or new_hj < 0 or new_hj > 1:
            # Try smaller delta or reject
            # Let's just reject this move
            continue
            
        # Apply move
        h[i_idx] = new_hi
        h[j_idx] = new_hj
        
        # Calculate new score
        new_score = _compute_score(h)
        
        # Acceptance criterion
        if new_score < curr_score:
            curr_score = new_score
            _update_best(h)
        else:
            # Metropolis criterion
            # Prob = exp(-(new_score - curr_score) / T)
            # If new_score > curr_score, exponent is negative.
            delta_score = new_score - curr_score
            if delta_score > 0:
                prob = np.exp(-delta_score / T)
                if np.random.rand() < prob:
                    curr_score = new_score
                else:
                    # Revert move
                    h[i_idx] -= delta
                    h[j_idx] += delta
            else:
                # This shouldn't happen if new_score >= curr_score check above was strict
                curr_score = new_score
                
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h

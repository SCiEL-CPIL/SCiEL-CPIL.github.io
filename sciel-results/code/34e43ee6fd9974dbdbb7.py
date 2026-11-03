# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_fft(h):
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H, _FFT_SIZE)
    return ac[:2 * N - 1]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _cd_step(h, i, deadline):
    """
    Perform one coordinate descent step for index i.
    Finds h_i_new minimizing score.
    """
    s_old = np.sum(h)
    h_i_old = h[i]
    
    # Precompute A_k = h_{k-i} + h_{i-k}
    # c_k = sum_j h_j h_{k-j}
    # c_k = h_i h_{k-i} + h_{k-i} h_i + ... ? No.
    # c_k = sum_{j=0}^{N-1} h_j h_{k-j}
    # Term with h_i: j=i -> h_i h_{k-i}.
    # Term with h_{k-i} (if k-i != i): j=k-i -> h_{k-i} h_i.
    # So c_k = h_i (h_{k-i} + h_{i-k}) + c_k_rest
    # Note: h_{x} is 0 if x < 0 or x >= N.
    
    A = np.zeros(2 * N - 1)
    # Indices k such that k-i is in [0, N-1]
    k1 = np.arange(i, i + N)
    # h_{k-i} is h[0..N-1]
    # A[k] gets h[k-i]
    A[k1] += h
    
    # Indices k such that i-k is in [0, N-1] => k-i in [-N+1, 0] => k in [i-N+1, i]
    # But k must be in [0, 2N-2]
    k2_start = max(0, i - N + 1)
    k2_end = min(2 * N - 2, i)
    if k2_end >= k2_start:
        k2 = np.arange(k2_start, k2_end + 1)
        # h_{i-k}
        # If k = i - j, then i-k = j.
        # j ranges from i-k2_end to i-k2_start.
        # j is in [0, N-1].
        # h_{i-k} corresponds to h[j].
        # k = i - j => j = i - k.
        # So we add h[i-k] to A[k].
        # Vectorized:
        # k2 = [k_start, ..., k_end]
        # indices in h: i - k2
        idx_h = i - k2
        # Filter valid indices
        valid = (idx_h >= 0) & (idx_h < N)
        if np.any(valid):
            A[k2[valid]] += h[idx_h[valid]]
            
    # c_k_rest = c_k - h_i * A_k
    # We don't need c_k explicitly if we recompute from scratch?
    # No, we need c_k to evaluate the function.
    # Let's compute c_k once outside the loop?
    # But h changes.
    # We can maintain c_k incrementally.
    # For now, let's just compute c_k inside the search?
    # That would be O(N) per eval.
    
    # Better: Compute c_k once for current h.
    c = _autoconv_fft(h)
    
    # Function to evaluate score for new h_i
    def obj(delta):
        h_i_new = h_i_old + delta
        s_new = s_old + delta
        if abs(s_new) < 1e-9:
            return 1e10
        # c_new = c + delta * A
        c_new = c + delta * A
        max_abs = np.max(np.abs(c_new))
        return max_abs / (s_new * s_new)
        
    # Golden Section Search for delta
    # Range for delta: [-2*abs(h_i_old), 2*abs(h_i_old)]?
    # Or wider?
    # Let's try [-10, 10] relative to scale.
    # Scale of h is roughly 1/sqrt(N) if sum is 1?
    # Let's use a range based on current h_i.
    if abs(h_i_old) < 1e-6:
        a, b = -1.0, 1.0
    else:
        a, b = -2.0 * h_i_old, 2.0 * h_i_old
        
    # Ensure a < b
    if a > b: a, b = b, a
    
    # Golden ratio
    gr = (np.sqrt(5) - 1) / 2
    c_pt = b - gr * (b - a)
    d_pt = a + gr * (b - a)
    
    fc = obj(c_pt - h_i_old) # delta = c_pt - h_i_old
    fd = obj(d_pt - h_i_old)
    
    # Iterative refinement
    for _ in range(50):
        if time.time() > deadline:
            break
        if fc < fd:
            b = d_pt
            d_pt = c_pt
            fd = fc
            c_pt = b - gr * (b - a)
            fc = obj(c_pt - h_i_old)
        else:
            a = c_pt
            c_pt = d_pt
            fc = fd
            d_pt = a + gr * (b - a)
            fd = obj(d_pt - h_i_old)
            
    # Best delta is closer to the one with lower value
    if fc < fd:
        best_pt = c_pt
    else:
        best_pt = d_pt
        
    h_i_new = best_pt
    h[i] = h_i_new
    return h

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Multiple restarts with different inits
    inits = []
    inits.append(np.ones(N))
    
    # Random
    rng = np.random.default_rng(42)
    for seed in [1, 2, 3, 4, 5]:
        r = np.random.default_rng(seed)
        h_r = r.standard_normal(N)
        inits.append(h_r)
        
    # Chirp
    i = np.arange(N)
    inits.append(np.cos(np.pi * i**2 / N))
    
    # Gaussian
    x = np.arange(N)
    inits.append(np.exp(-0.5 * ((x - N/2) / (N/4))**2))
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
            
        h = h_init.copy()
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s # Normalize sum to 1
        
        # Coordinate Descent
        # Passes
        for p in range(20):
            if time.time() > deadline - 5:
                break
            
            # Shuffle order to avoid bias
            order = np.random.permutation(N)
            for i in order:
                if time.time() > deadline - 5:
                    break
                h = _cd_step(h, i, deadline)
                
            # Renormalize to sum 1?
            # The CD step minimizes the ratio, so sum might drift.
            # But the ratio is scale invariant.
            # However, if sum goes to 0, score explodes.
            # We should probably constrain sum != 0.
            
            current_score = _score(h)
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
                
    return best_h

# EVOLVE-BLOCK-END

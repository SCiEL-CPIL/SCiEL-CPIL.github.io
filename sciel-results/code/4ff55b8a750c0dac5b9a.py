"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlap_max(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    
    We interpret "overlap of h with shifted 1-h" as:
    For a shift k, overlap = sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize the maximum over all shifts k.
    
    Actually, let me reconsider. The task says "minimize the maximum overlap of h with shifted 1-h".
    
    If h is a step function on [0,2] with N steps, each of width 2/N, and h[i] is the height.
    The integral constraint: sum(h[i]) * (2/N) = 1, so sum(h[i]) = N/2.
    
    "Overlap of h with shifted 1-h" likely means: for a shift, compute the pointwise product
    of h(x) and (1-h)(x+shift), then integrate. Since these are step functions, the overlap
    for shift k (in units of step width) is:
    sum_i h[i] * (1 - h[(i+k) % N]) * (2/N)
    
    But we want to minimize the maximum over shifts. The factor 2/N is constant, so we can
    minimize max_k sum_i h[i] * (1 - h[(i+k) % N]).
    
    Note: sum_i h[i] * (1 - h[(i+k)%N]) = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    = N/2 - sum_i h[i]*h[(i+k)%N]
    
    So minimizing the max overlap is equivalent to maximizing the min of sum_i h[i]*h[(i+k)%N]
    over k. Or equivalently, we want to make the autocorrelation of h as uniform as possible,
    with all values equal to (N/2)^2 / N = N/4 (if h were constant 0.5).
    
    Wait, if h[i] = 0.5 for all i, then sum_i h[i]*h[(i+k)%N] = N * 0.25 = N/4.
    And overlap = N/2 - N/4 = N/4.
    
    The score is described as "the upper bound on C5". Let me just compute what the evaluator
    likely measures and try to minimize it.
    
    Let me define: for each shift k in 0..N-1, compute overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
    The objective is max_k overlap_k. We want to minimize this.
    
    With h = 0.5 everywhere: overlap_k = N * 0.5 * 0.5 = N/4 = 25 for N=100.
    """
    N = len(h)
    # Compute all circular cross-correlations of h with (1-h)
    # overlap_k = sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i+k)%N]
    # = N/2 - autocorr_k where autocorr_k = sum_i h[i]*h[(i+k)%N]
    
    h1 = 1.0 - h
    # Use FFT for efficient computation
    Np = 1 << (2 * N - 1).bit_length()  # power of 2 >= 2N-1
    H = np.fft.rfft(h, Np)
    H1 = np.fft.rfft(h1, Np)
    # Circular correlation: sum_i h[i] * h1[(i+k) % N]
    # This is the circular cross-correlation
    corr = np.fft.irfft(H * np.conj(H1), Np)[:N]
    # corr[k] = sum_i h[i] * h1[(i+k) % N] (need to verify indexing)
    # Actually, let's be more careful.
    # np.fft.irfft(H * conj(H1)) gives the circular cross-correlation
    # corr[k] = sum_i h[(i-k) % N] * h1[i] = sum_i h[i] * h1[(i+k) % N]
    # Yes, that's what we want.
    max_overlap = np.max(corr)
    return max_overlap

def _score(h):
    """Return the score (lower is better)."""
    return _overlap_max(h)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second deadline
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Try some structured patterns first
    # Pattern 1: alternating 0 and 1
    h_alt = np.zeros(N)
    h_alt[0::2] = 1.0
    # Check sum
    if np.sum(h_alt) != N/2:
        h_alt = np.full(N, 0.5)
    else:
        s = _score(h_alt)
        if s < best_score:
            best_score = s
            best_h = h_alt.copy()
    
    # Pattern 2: blocks
    # Try various block patterns
    for block_size in range(1, N//2 + 1):
        h_block = np.zeros(N)
        # Place 1s in blocks
        num_ones = N // 2
        # Simple: first N/2 are 1, rest are 0
        h_block[:N//2] = 1.0
        if np.sum(h_block) == N/2:
            s = _score(h_block)
            if s < best_score:
                best_score = s
                best_h = h_block.copy()
        break  # Just try one block pattern for now
    
    # Pattern 3: periodic patterns with period p
    for p in [2, 3, 4, 5, 7, 10, 11, 13, 17, 20, 25]:
        if p > N:
            continue
        if N % p != 0:
            continue
        # For each period, try all possible patterns of 0s and 1s that give correct sum
        # This is combinatorial, so just try a few
        # Pattern with ceil(p/2) ones per period
        num_ones_per_period = p // 2
        if num_ones_per_period * (N // p) == N // 2:
            h_pat = np.zeros(N)
            for i in range(N // p):
                for j in range(num_ones_per_period):
                    h_pat[i*p + j] = 1.0
            if np.sum(h_pat) == N/2:
                s = _score(h_pat)
                if s < best_score:
                    best_score = s
                    best_h = h_pat.copy()
    
    # Pattern 4: use a smooth function
    # h(x) = 0.5 + a * sin(2*pi*x/2) = 0.5 + a * sin(pi*x)
    # For step function: h[i] = 0.5 + a * sin(pi * i * 2/N)
    # Need sum = N/2, which is automatic since sin has zero mean over full period
    for a in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
        x = np.arange(N) * 2.0 / N
        h_sin = 0.5 + a * np.sin(np.pi * x)
        h_sin = np.clip(h_sin, 0, 1)
        # Adjust to ensure exact sum
        diff = N/2 - np.sum(h_sin)
        h_sin += diff / N
        h_sin = np.clip(h_sin, 0, 1)
        # Re-adjust
        diff = N/2 - np.sum(h_sin)
        if abs(diff) > 1e-10:
            # Distribute the difference
            h_sin += diff / N
            h_sin = np.clip(h_sin, 0, 1)
        if abs(np.sum(h_sin) - N/2) < 1e-6:
            s = _score(h_sin)
            if s < best_score:
                best_score = s
                best_h = h_sin.copy()
    
    # Try cosine with different frequencies
    for freq in [1, 2, 3, 4, 5]:
        for a in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
            x = np.arange(N) * 2.0 / N
            h_cos = 0.5 + a * np.cos(2 * np.pi * freq * x / 2)
            h_cos = np.clip(h_cos, 0, 1)
            diff = N/2 - np.sum(h_cos)
            h_cos += diff / N
            h_cos = np.clip(h_cos, 0, 1)
            diff = N/2 - np.sum(h_cos)
            if abs(diff) > 1e-10:
                h_cos += diff / N
                h_cos = np.clip(h_cos, 0, 1)
            if abs(np.sum(h_cos) - N/2) < 1e-6:
                s = _score(h_cos)
                if s < best_score:
                    best_score = s
                    best_h = h_cos.copy()
    
    # Gradient descent with smooth surrogate
    # Use log-sum-exp for smooth max
    def smooth_max(vals, beta=100):
        m = np.max(vals)
        return m + np.log(np.sum(np.exp((vals - m) * beta)) / len(vals)) / beta
    
    def smooth_score(h):
        N = len(h)
        h1 = 1.0 - h
        Np = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, Np)
        H1 = np.fft.rfft(h1, Np)
        corr = np.fft.irfft(H * np.conj(H1), Np)[:N]
        return smooth_max(corr, beta=100)
    
    # Do gradient descent
    h_current = best_h.copy()
    lr = 0.01
    for it in range(5000):
        if time.time() > deadline - 10:
            break
        # Compute gradient numerically or analytically
        # For simplicity, use finite differences on a few coordinates
        # Actually, let's use a simple projected gradient
        # Compute gradient of smooth_score
        grad = np.zeros(N)
        eps = 1e-4
        base_score = smooth_score(h_current)
        for i in range(N):
            if time.time() > deadline - 10:
                break
            h_plus = h_current.copy()
            h_plus[i] += eps
            h_plus = np.clip(h_plus, 0, 1)
            # Adjust to maintain sum constraint
            diff = N/2 - np.sum(h_plus)
            if abs(diff) > 1e-12:
                h_plus += diff / N
                h_plus = np.clip(h_plus, 0, 1)
            score_plus = smooth_score(h_plus)
            grad[i] = (score_plus - base_score) / eps
        # Project gradient descent step
        h_new = h_current - lr * grad
        # Project to constraints: sum = N/2, 0 <= h <= 1
        # First, adjust sum
        diff = N/2 - np.sum(h_new)
        h_new += diff / N
        h_new = np.clip(h_new, 0, 1)
        # Re-adjust sum after clipping
        diff = N/2 - np.sum(h_new)
        if abs(diff) > 1e-12:
            # Distribute remaining difference to entries not at bounds
            free_mask = (h_new > 0) & (h_new < 1)
            if np.sum(free_mask) > 0:
                h_new[free_mask] += diff / np.sum(free_mask)
                h_new = np.clip(h_new, 0, 1)
        
        s_current = _score(h_new)
        if s_current < best_score:
            best_score = s_current
            best_h = h_new.copy()
        h_current = h_new
        # Decay learning rate
        lr *= 0.999
    
    # Final check
    if time.time() > deadline - 5:
        pass
    
    # Ensure exact sum
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Adjust the largest free entry
        sorted_idx = np.argsort(best_h)
        for idx in sorted_idx:
            if best_h[idx] + diff >= 0 and best_h[idx] + diff <= 1:
                best_h[idx] += diff
                break
        else:
            # Distribute
            best_h += diff / N
            best_h = np.clip(best_h, 0, 1)
    
    # Final verification
    if abs(np.sum(best_h) - N/2) > 1e-6:
        # Fallback to constant
        return np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END

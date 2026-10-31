import time
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    Uses a frequency-domain approach: the peak of the autoconvolution is
    related to the L-infinity norm of the Fourier transform. We use a
    projected subgradient method to minimize the peak sidelobe level
    subject to nonnegativity and normalization constraints.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Normalize: fix sum(h) = 1 so the problem is well-defined
    # Autoconvolution at lag k: sum_i h_i * h_{i+k}
    # Peak = max_k |autoconv(k)| for k != 0, normalized by autoconv(0)
    # Actually, let's think about what "normalized maximum of autoconvolution" means
    
    # The autoconvolution of a step function with heights h_i on N intervals:
    # (f*f)(t) where t is the shift. For discrete step function, 
    # the autoconvolution at shift k (in units of interval width) is:
    # R(k) = sum_i h_i * h_{i+k} for k >= 0 (with appropriate bounds)
    # and R(-k) = R(k)
    
    # The "normalized maximum" likely means: max_k R(k) / R(0)
    # where R(0) = sum h_i^2
    
    # For uniform h: R(0) = N * h^2, R(k) = (N-k)*h^2 for k=0,...,N-1
    # So max R(k)/R(0) = R(0)/R(0) = 1 (at k=0)
    # But if we exclude k=0, then max is R(1)/R(0) = (N-1)/N
    
    # Actually, the first autocorrelation inequality C1 typically bounds
    # the first nonzero autocorrelation coefficient. Let me reconsider.
    
    # C1 is likely: for a function with certain properties, 
    # |autocorr(1)| <= C1 * autocorr(0)
    # or similar. The "normalized maximum of autoconvolution" 
    # might be max_{k!=0} |R(k)| / R(0)
    
    # For uniform: max_{k!=0} R(k)/R(0) = (N-1)/N * N/N... 
    # R(0) = N*h^2, R(1) = (N-1)*h^2, so R(1)/R(0) = (N-1)/N
    
    # Can we do better than uniform? Let's try to minimize max_{k!=0} |R(k)| / R(0)
    
    # This is a non-convex problem in general, but we can try:
    # 1. Start with uniform
    # 2. Use subgradient descent on the peak sidelobe
    
    # Let's compute the autoconvolution efficiently using FFT
    # R(k) for k = -(N-1), ..., N-1
    # Use zero-padding: create array of length 2N, put h in first N, 
    # then FFT, square magnitude, IFFT gives autoconvolution
    
    def compute_autoconv(h):
        """Compute full autoconvolution using FFT."""
        # Pad to length at least 2N-1
        L = 2 * N
        h_pad = np.zeros(L)
        h_pad[:N] = h
        H = np.fft.rfft(h_pad)
        ac = np.fft.irfft(H * np.conj(H), n=L)
        # ac[k] for k=0,...,L-1 corresponds to shifts
        # For real input, ac is symmetric: ac[k] = ac[L-k]
        # The meaningful shifts are 0 to N-1
        return ac[:N]
    
    def compute_normalized_peak(h):
        """Compute max_{k!=0} |R(k)| / R(0)."""
        ac = compute_autoconv(h)
        r0 = ac[0]
        if r0 < 1e-15:
            return 0.0
        # Exclude k=0
        peak = np.max(np.abs(ac[1:]))
        return peak / r0
    
    def compute_subgradient(h):
        """
        Compute subgradient of the normalized peak w.r.t. h.
        The peak is at some k*, so gradient is related to 
        the k*-th autoconvolution pattern.
        """
        ac = compute_autoconv(h)
        r0 = ac[0]
        if r0 < 1e-15:
            return np.zeros(N)
        
        # Find the shift k* that achieves the peak (excluding k=0)
        # We want to minimize max_{k!=0} |ac[k]| / ac[0]
        # Let f = max_{k!=0} |ac[k]| / ac[0]
        # Subgradient w.r.t. h: if ac[k*] is the maximizing term,
        # then d(ac[k*])/dh_i involves h_{i-k*} and h_{i+k*}
        
        # For k >= 0: ac[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k]
        # d(ac[k])/dh[i] = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0)
        
        # Find all k that achieve (or nearly achieve) the peak
        peak_val = np.max(np.abs(ac[1:]))
        if peak_val < 1e-15:
            return np.zeros(N)
        
        # Threshold for "active" constraints
        threshold = peak_val * 0.99
        active_k = np.where(np.abs(ac[1:]) >= threshold)[0] + 1
        
        grad = np.zeros(N)
        for k in active_k:
            sign = np.sign(ac[k]) if ac[k] != 0 else 1.0
            # d(ac[k])/dh[i] for i = 0,...,N-1
            # ac[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
            # d/dh[i] = h[i+k] if i < N-k, else 0
            #           + h[i-k] if i >= k, else 0
            for i in range(N):
                g = 0.0
                if i + k < N:
                    g += h[i + k]
                if i >= k:
                    g += h[i - k]
                grad[i] += sign * g
        
        # Normalize by r0^2 (since f = peak/r0, and we're differentiating)
        # Actually, f = max_k |ac[k]| / ac[0]
        # df/dh = (1/ac[0]) * d(peak)/dh - (peak/ac[0]^2) * d(ac[0])/dh
        # d(ac[0])/dh[i] = 2*h[i]
        
        # For simplicity, just use the subgradient of the peak term
        # and project onto the constraint
        grad /= (r0 * r0)  # rough normalization
        
        return grad
    
    # Start with uniform
    h = np.ones(N) / N  # normalize so sum = 1
    
    best_h = h.copy()
    best_score = compute_normalized_peak(best_h)
    
    # Subgradient descent with adaptive step size
    lr = 0.01
    n_iter = 0
    max_iter = 5000
    
    while n_iter < max_iter:
        if time.time() - start_time > deadline:
            break
        
        grad = compute_subgradient(h)
        
        # Projected subgradient step: stay in nonnegative orthant
        h_new = h - lr * grad
        h_new = np.maximum(h_new, 0)
        
        # Renormalize to keep sum(h) = 1
        s = np.sum(h_new)
        if s < 1e-15:
            h_new = np.ones(N) / N
        else:
            h_new /= s
        
        # Check if improved
        score_new = compute_normalized_peak(h_new)
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            # Reduce learning rate slightly on improvement
            lr *= 0.95
        else:
            # Backtrack
            lr *= 0.8
            h_new = h
        
        h = h_new
        n_iter += 1
        
        # Early stopping if lr too small
        if lr < 1e-10:
            break
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    n_restarts = 3
    
    for _ in range(n_restarts):
        if time.time() - start_time > deadline:
            break
        
        # Random initialization with some structure
        h_init = np.abs(rng.standard_normal(N))
        h_init /= np.sum(h_init)
        
        lr_r = 0.005
        for _ in range(1000):
            if time.time() - start_time > deadline:
                break
            
            grad_r = compute_subgradient(h_init)
            h_r = h_init - lr_r * grad_r
            h_r = np.maximum(h_r, 0)
            s_r = np.sum(h_r)
            if s_r < 1e-15:
                h_r = np.ones(N) / N
            else:
                h_r /= s_r
            
            score_r = compute_normalized_peak(h_r)
            if score_r < best_score:
                best_score = score_r
                best_h = h_r.copy()
                lr_r *= 0.95
            else:
                lr_r *= 0.8
                h_r = h_init
            
            h_init = h_r
    
    # Final check: compare with uniform
    uniform_score = compute_normalized_peak(np.ones(N) / N)
    if uniform_score < best_score:
        best_h = np.ones(N) / N
        best_score = uniform_score
    
    # Scale back to original convention (heights, not normalized)
    # The score depends on the ratio, so scaling doesn't matter
    # Return the heights
    return best_h.tolist()
# EVOLVE-BLOCK-END

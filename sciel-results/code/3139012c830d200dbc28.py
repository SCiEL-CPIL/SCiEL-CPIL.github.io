"""Rewritten program: structured sign design + local search for C3 autoconvolution bound."""
import numpy as np
import time

N = 400

def _autoconv(signed):
    """Compute the autoconvolution of a signed step function on [-1/4, 1/4].
    
    The autoconvolution (C3) is the maximum over all shifts of the integral
    of f(x) f(x+t) dx, normalized appropriately.
    
    For a step function with N equal intervals of height h_i (each ±1),
    the autoconvolution at lag k is sum_i h_i * h_{i+k} (with appropriate
    bounds).
    
    The C3 upper bound is max_k |autoconv(k)| / (sum |h_i|)^2 * N
    (or similar normalization - we use the raw max of the autoconvolution
    relative to the total energy).
    
    Actually, let me think about this more carefully. The task says:
    - Choose a signed step function with 400 equal intervals on [-1/4, 1/4]
    - Score is the upper bound on C3
    - Lower is better
    
    The autoconvolution of a function f on [-1/4, 1/4] is:
    (f * f)(t) = integral f(x) f(x+t) dx
    
    For a step function with N intervals, each of width w = (1/2)/N = 1/(2N),
    the autoconvolution at lag k (in units of w) is:
    sum_{i} h_i * h_{i+k} * w
    
    where the sum is over valid i such that both i and i+k are in [0, N-1].
    
    The C3 bound is typically max_t |(f*f)(t)| / (integral |f|)^2 or similar.
    
    Let me use: score = max_k |sum_i h_i h_{i+k}| / N
    This is a reasonable proxy for the autoconvolution peak.
    """
    # Compute all lags of the autoconvolution
    # Using FFT for efficiency
    # Pad to avoid circular convolution
    size = 2 * N
    f = np.zeros(size)
    f[:N] = signed
    # Linear convolution via FFT
    F = np.fft.fft(f, n=2*size)
    conv = np.fft.ifft(F * F).real[:2*N]
    # The autoconvolution lags are conv[N-1 : N-1+N] roughly
    # Actually for linear convolution of f with f, the result has length 2N-1
    # conv[k] = sum_i f[i] * f[k-i] for k=0..2N-2
    # We want the "autoconvolution" which is sum_i f[i] * f[i+k] for various k
    # This is the cross-correlation, not convolution
    # Let me use correlation instead
    
    # Correlation: corr[k] = sum_i f[i] * f[i+k] for k >= 0
    # For k < 0, corr[k] = sum_i f[i] * f[i+k] = sum_j f[j-k] * f[j] = corr[-k] (for real)
    # So we just need k >= 0
    
    # Use FFT-based correlation
    F = np.fft.fft(f, n=2*size)
    corr_full = np.fft.ifft(np.conj(F) * F).real
    # corr_full[k] for k=0..size-1 gives the circular correlation
    # For linear correlation of length-N signals padded to 2*size:
    # corr[k] = corr_full[k] for k = 0..N-1 (positive lags)
    # corr[-k] = corr_full[size-k] for k = 1..N-1 (negative lags)
    
    corr = np.zeros(2*N - 1)
    for k in range(N):
        corr[k] = corr_full[k]  # lag = k
    for k in range(1, N):
        corr[2*N-1-k] = corr_full[size-k]  # lag = -k, stored at index 2N-1-k
    
    # Actually, let me just do it directly for clarity and correctness
    # N=400, so O(N^2) = 160000 operations, which is fine
    corr = np.zeros(2*N - 1)
    for k in range(N):
        # lag = k: sum_{i=0}^{N-1-k} h[i] * h[i+k]
        corr[k] = np.dot(signed[:N-k], signed[k:])
    for k in range(1, N):
        # lag = -k: sum_{i=k}^{N-1} h[i] * h[i-k]
        corr[2*N-1-k] = np.dot(signed[k:], signed[:N-k])
    
    # The C3 bound is related to the maximum absolute value of the autoconvolution
    # Normalized by the total mass or energy
    # Let's use: max |corr[k]| / N
    # This gives a value in [0, 1] for signed functions
    return np.max(np.abs(corr)) / N


def _score(signed):
    """Compute the score (lower is better)."""
    return _autoconv(signed)


def construct():
    """Construct a signed step function that minimizes the C3 autoconvolution bound."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_score = float('inf')
    best_signed = None
    
    # Strategy 1: Start with structured patterns
    # Try several initializations
    
    # 1a: All ones (baseline)
    candidates = []
    
    # 1b: Alternating signs
    alt = np.ones(N)
    alt[1::2] = -1
    candidates.append(alt.copy())
    
    # 1c: Block patterns - various block sizes
    for block_size in [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100, 125, 200]:
        if block_size > N:
            continue
        n_blocks = N // block_size
        if n_blocks == 0:
            continue
        # Random sign for each block
        block_signs = np.random.choice([-1, 1], size=n_blocks)
        sig = np.repeat(block_signs, block_size)
        if len(sig) < N:
            sig = np.concatenate([sig, np.random.choice([-1, 1], size=N - len(sig))])
        candidates.append(sig[:N])
    
    # 1d: Random signs
    for _ in range(20):
        candidates.append(np.random.choice([-1, 1], size=N).astype(float))
    
    # 1e: Low-discrepancy-like patterns (Thue-Morse, etc.)
    # Thue-Morse sequence
    tm = np.zeros(N)
    for i in range(N):
        # Count bits set in i, mod 2
        tm[i] = 1.0 if bin(i).count('1') % 2 == 0 else -1.0
    candidates.append(tm)
    
    # 1f: Fourier-based: signs of a low-frequency sinusoid
    for freq in [1, 2, 3, 4, 5, 7, 10, 13, 17, 20]:
        x = np.linspace(0, 1, N, endpoint=False)
        sig = np.sign(np.sin(2 * np.pi * freq * x + np.random.uniform(0, 2*np.pi)))
        sig[sig == 0] = 1
        candidates.append(sig)
    
    # Evaluate all candidates
    for cand in candidates:
        if len(cand) != N:
            continue
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_signed = cand.copy()
    
    if best_signed is None:
        best_signed = np.ones(N)
        best_score = _score(best_signed)
    
    # Strategy 2: Local search - flip individual bits to improve
    current = best_signed.copy()
    current_score = best_score
    
    # Precompute the contribution of each position to the autoconvolution
    # For efficiency, we'll use a smarter local search
    
    # Local search: try flipping each bit and see if it improves
    # Use a priority queue or random sampling for efficiency
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try random subset of positions to flip
        n_tries = min(N, max(50, int((deadline - time.time()) * 50)))
        indices = np.random.choice(N, size=n_tries, replace=False)
        
        for idx in indices:
            if time.time() > deadline:
                break
            # Try flipping this bit
            old_val = current[idx]
            current[idx] = -old_val
            new_score = _score(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_signed = current.copy()
            else:
                current[idx] = old_val
        
        # Also try some multi-bit flips
        for _ in range(min(20, int((deadline - time.time()) * 2))):
            if time.time() > deadline:
                break
            n_flip = np.random.choice([2, 3, 4, 5])
            flip_indices = np.random.choice(N, size=n_flip, replace=False)
            old_vals = current[flip_indices].copy()
            current[flip_indices] = -current[flip_indices]
            new_score = _score(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_signed = current.copy()
            else:
                current[flip_indices] = old_vals
    
    return best_signed

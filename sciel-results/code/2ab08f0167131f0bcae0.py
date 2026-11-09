import time
import numpy as np
from scipy import signal

N = 400
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # safety margin before 150s


def _c3_prime_val(h):
    """
    Compute max_k |autocorr_k| / (sum h)^2 for signed step function h of length N.
    autocorr_k = sum_i h[i] * h[i+k] for k in [-(N-1), N-1], with zero padding.
    """
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if abs(total) < 1e-12:
        return np.inf
    denom = total * total
    ac = np.correlate(h, h, mode='full')
    return np.max(np.abs(ac)) / denom


def _c3_prime_fft(h):
    """
    Fast computation of max_k |autocorr_k| / (sum h)^2 using FFT.
    """
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if abs(total) < 1e-12:
        return np.inf
    denom = total * total
    
    # Use FFT for fast autocorrelation
    n = len(h)
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    
    H = np.fft.rfft(h, n=fft_size)
    ac = np.fft.irfft(H * np.conj(H), n=fft_size)[:2*n-1]
    
    return np.max(np.abs(ac)) / denom


def _compute_autocorr(h):
    """Compute full autocorrelation using FFT."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    ac = np.fft.irfft(H * np.conj(H), n=fft_size)[:2*n-1]
    return ac


def _walsh_hadamard_sequence(n, index):
    """Generate the n-th Walsh-Hadamard sequence of length 2^k >= n."""
    # Find smallest power of 2 >= n
    k = 1
    while (1 << k) < n:
        k += 1
    size = 1 << k
    
    # Compute the n-th Walsh function
    # Hadamard matrix construction
    def hadamard(m):
        if m == 1:
            return np.array([[1.0]])
        H = hadamard(m // 2)
        return np.block([[H, H], [H, -H]])
    
    H = hadamard(size)
    row = H[index % size]
    return row[:n]


def _fourier_basis_signal(n, freq_index, phase=0.0):
    """Generate a sign(sin) signal at a specific Fourier frequency."""
    t = np.arange(n)
    return np.sign(np.sin(2 * np.pi * freq_index * t / n + phase))


def _random_zero_mean(n, seed):
    """Generate a random ±1 sequence with exactly zero mean (or near-zero)."""
    rng = np.random.RandomState(seed)
    h = rng.choice([-1.0, 1.0], size=n)
    # Adjust to make sum closer to zero
    total = h.sum()
    if total > 0:
        # Flip some +1 to -1
        n_flip = int(total / 2)
        if n_flip > 0:
            plus_indices = np.where(h == 1.0)[0]
            flip_idx = rng.choice(plus_indices, size=min(n_flip, len(plus_indices)), replace=False)
            h[flip_idx] = -1.0
    elif total < 0:
        n_flip = int(-total / 2)
        if n_flip > 0:
            minus_indices = np.where(h == -1.0)[0]
            flip_idx = rng.choice(minus_indices, size=min(n_flip, len(minus_indices)), replace=False)
            h[flip_idx] = 1.0
    return h


def _m_sequence_like(n, seed):
    """Generate a sequence with good autocorrelation properties (m-sequence-like)."""
    rng = np.random.RandomState(seed)
    # Start with random, then apply local optimization using FFT
    h = rng.choice([-1.0, 1.0], size=n)
    current_val = _c3_prime_fft(h)
    # Quick local refinement
    for _ in range(100):
        i = rng.randint(0, n)
        trial = h.copy()
        trial[i] *= -1
        trial_val = _c3_prime_fft(trial)
        if trial_val < current_val - 1e-12:
            h = trial
            current_val = trial_val
    return h


def _local_search_fast(h, max_iters=500, time_budget=None):
    """
    Fast local search using analytical flip deltas.
    Returns (best_h, best_val).
    """
    h = h.copy()
    n = len(h)
    total = h.sum()
    ac_base = _compute_autocorr(h)
    
    if abs(total) < 1e-12:
        return h, np.inf
    
    current_val = np.max(np.abs(ac_base)) / (total * total)
    
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if time_budget is not None and time.time() > time_budget:
            break
        improved = False
        iters += 1
        
        # Find the best single flip
        best_improvement = current_val
        best_idx = -1
        best_new_total = total
        best_new_ac = None
        
        for i in range(n):
            s = h[i]
            new_total = total - 2 * s
            if abs(new_total) < 1e-12:
                continue
            new_denom = new_total * new_total
            
            # Compute delta to autocorrelation
            lags = np.arange(-(n-1), n)
            idx1 = i + lags
            idx2 = i - lags
            val1 = np.zeros(n * 2 - 1, dtype=float)
            val2 = np.zeros(n * 2 - 1, dtype=float)
            mask1 = (idx1 >= 0) & (idx1 < n)
            mask2 = (idx2 >= 0) & (idx2 < n)
            val1[mask1] = h[idx1[mask1]]
            val2[mask2] = h[idx2[mask2]]
            
            delta = -2 * s * (val1 + val2)
            delta[n-1] = 0  # lag 0 doesn't change
            
            new_ac = ac_base + delta
            new_max_abs = np.max(np.abs(new_ac))
            new_val = new_max_abs / new_denom
            
            if new_val < best_improvement - 1e-12:
                best_improvement = new_val
                best_idx = i
                best_new_total = new_total
                best_new_ac = new_ac
        
        if best_idx >= 0:
            s = h[best_idx]
            h[best_idx] = -s
            total = best_new_total
            ac_base = best_new_ac
            current_val = best_improvement
            improved = True
    
    return h, current_val


def construct():
    global START_TIME, DEADLINE
    
    # Start with the seed (all ones)
    best_h = np.ones(N)
    best_val = _c3_prime_val(best_h)
    
    # Generate diverse candidate patterns
    candidates = []
    
    # 1. Walsh-Hadamard sequences (orthogonal, good autocorrelation properties)
    for idx in range(0, 64):
        candidates.append(_walsh_hadamard_sequence(N, idx))
    
    # 2. Fourier basis signals at various frequencies
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150]:
        for phase in [0, np.pi/4, np.pi/2, np.pi/3, np.pi/6, 2*np.pi/3]:
            candidates.append(_fourier_basis_signal(N, freq, phase))
    
    # 3. Random zero-mean sequences (many different seeds)
    for seed in range(200):
        candidates.append(_random_zero_mean(N, seed))
    
    # 4. Alternating patterns with various block sizes
    for block_size in [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 50, 64, 80, 100, 128]:
        h = np.zeros(N)
        sign = 1.0
        for i in range(N):
            h[i] = sign
            if (i + 1) % block_size == 0:
                sign = -sign
        candidates.append(h)
    
    # 5. Chirp patterns
    t = np.arange(N)
    for f0 in [1, 2, 3, 5]:
        for f1 in [3, 5, 10, 20, 30, 50, 100]:
            if f1 != f0:
                freqs = f0 + (f1 - f0) * t / N
                phase = 2 * np.pi * np.cumsum(freqs) / N
                candidates.append(np.sign(np.sin(phase)))
    
    # 6. Random sequences (not zero-mean adjusted)
    for seed in range(100):
        rng = np.random.RandomState(seed)
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates using FFT for speed
    for h in candidates:
        if time.time() > DEADLINE - 30:
            break
        try:
            val = _c3_prime_fft(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
        except Exception:
            continue
    
    # Local refinement using fast analytical flip evaluation
    if time.time() < DEADLINE - 15:
        current_h, current_val = _local_search_fast(best_h, max_iters=2000, time_budget=DEADLINE - 10)
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
    
    # Try multiple restarts from different starting points if time permits
    if time.time() < DEADLINE - 20:
        for restart in range(10):
            if time.time() > DEADLINE - 10:
                break
            start_h = _random_zero_mean(N, 1000 + restart)
            h_candidate, val_candidate = _local_search_fast(start_h, max_iters=500, time_budget=DEADLINE - 10)
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate
    
    return best_h

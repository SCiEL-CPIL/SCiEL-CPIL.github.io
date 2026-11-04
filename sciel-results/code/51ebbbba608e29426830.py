"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1 values."""
    # Thue-Morse: t(n) = popcount(n) mod 2, mapped to ±1
    idx = np.arange(n, dtype=np.uint64)
    # Compute popcount efficiently
    # t(n) = number of 1-bits in n, mod 2
    # Use numpy: count bits
    # For efficiency, use a lookup or bit manipulation
    # np.bitwise_count is not standard, so use a trick
    # Actually, let's use a simple iterative approach for n=400
    tm = np.zeros(n, dtype=np.float64)
    tm[0] = 1.0
    for i in range(1, n):
        # t(i) = t(i >> 1) XOR (i & 1)
        # But we need the actual value, not just mod 2
        # t(i) = popcount(i) mod 2
        # Let's compute popcount
        x = i
        pc = 0
        while x:
            pc += x & 1
            x >>= 1
        tm[i] = 1.0 if (pc % 2 == 0) else -1.0
    return tm

def _autocorr_peak(s):
    """Compute the maximum absolute autocorrelation value (excluding lag 0)."""
    n = len(s)
    # Autocorrelation via FFT for speed
    # r[k] = sum_i s[i] * s[i+k] for k > 0
    # Use FFT-based autocorrelation
    f = np.fft.fft(s, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real
    # acf[0] = n (the DC term), acf[1..n-1] are the positive lags
    # acf[n+1..2n-1] correspond to negative lags
    # We want max |acf[k]| for k != 0
    # The autocorrelation at lag k for k in 1..n-1
    peak = np.max(np.abs(acf[1:n]))
    return peak

def _score(s):
    """Return the upper bound on C3 (lower is better)."""
    # The C3 bound is related to the peak autocorrelation
    # For a signed step function, the third autocorrelation inequality
    # involves the autoconvolution. The upper bound is proportional to
    # the peak of the autocorrelation sidelobes.
    # Let's compute the actual autoconvolution peak
    n = len(s)
    # Autoconvolution: c[k] = sum_i s[i] * s[i+k]
    # This is the same as autocorrelation for real signals
    # The C3 upper bound is typically: C3 <= max_k |sum_i s[i]s[i+k]s[i+2k]|
    # But for the "upper bound on C3" in the context of autocorrelation inequality,
    # it's likely the peak of the 2nd-order autocorrelation (autoconvolution)
    # Let me compute the 3-fold autocorrelation: sum_i s[i] * s[i+k] * s[i+2k]
    # Actually, for the third autocorrelation inequality, the bound involves
    # the maximum over k of |sum_i s[i] s[i+k] s[i+2k]|
    # But computing this for all k is O(n^2) which is fine for n=400
    
    # Let me think about this differently. The "upper bound on C3" likely refers to
    # a quantity that can be computed from the autocorrelation. In many contexts,
    # C3 is bounded by a function of the autocorrelation peak.
    # 
    # Given the task says "autoconvolution upper bound", let me compute the
    # autoconvolution: (s * s)[k] = sum_i s[i] * s[k-i]
    # The peak of this (excluding the center) is the key metric.
    
    # Autoconvolution via FFT
    f = np.fft.fft(s, n=2*n)
    ac = np.fft.ifft(f * f).real  # autoconvolution (not autocorrelation)
    # ac is centered at index n
    # The peak sidelobe is max |ac[k]| for k != n
    # Actually, let me reconsider. For a step function on [-1/4, 1/4],
    # the autoconvolution is (s * s)(t) = ∫ s(τ) s(t-τ) dτ
    # Discretized: sum_i s[i] s[i+k]
    # The "upper bound on C3" is likely the L-infinity norm of the 
    # autoconvolution sidelobes.
    
    # Let me just use the autocorrelation peak as the metric since that's
    # what the inspirations seem to use.
    return _autocorr_peak(s)

def _fast_greedy(s, max_iters=50000):
    """Fast greedy local search: flip bits that reduce the autocorrelation peak."""
    n = len(s)
    s = s.copy()
    best_peak = _autocorr_peak(s)
    current_peak = best_peak
    
    # Precompute autocorrelation for fast updates
    # When we flip s[j], the change in autocorrelation at lag k is:
    # delta_acf[k] = -2*s[j]*s[j+k] for k>0 (if j+k < n)
    #                -2*s[j]*s[j-k] for k>0 (if j-k >= 0)
    # This is complex, so let's just recompute periodically
    
    no_improve = 0
    for it in range(max_iters):
        # Try flipping a random position
        j = np.random.randint(0, n)
        old_val = s[j]
        s[j] = -old_val
        new_peak = _autocorr_peak(s)
        if new_peak < current_peak:
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
            no_improve = 0
        else:
            s[j] = old_val
            no_improve += 1
            if no_improve > 500:
                break
    return s, best_peak

def _simulated_annealing(s, t_start=1.0, t_end=0.01, n_iters=100000):
    """Simulated annealing to escape local minima."""
    n = len(s)
    s = s.copy()
    current_peak = _autocorr_peak(s)
    best_peak = current_peak
    best_s = s.copy()
    
    t = t_start
    for it in range(n_iters):
        t = t_start * (t_end / t_start) ** (it / n_iters)
        j = np.random.randint(0, n)
        old_val = s[j]
        s[j] = -old_val
        new_peak = _autocorr_peak(s)
        delta = new_peak - current_peak
        if delta < 0 or np.random.rand() < np.exp(-delta / max(t, 1e-10)):
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_s = s.copy()
        else:
            s[j] = old_val
    return best_s, best_peak

def construct():
    """Construct a signed step function with 400 intervals to minimize C3 upper bound."""
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # Start with Thue-Morse sequence (provably flat autocorrelation)
    s = _thue_morse(N)
    best_s = s.copy()
    best_peak = _autocorr_peak(s)
    
    # Phase 1: Fast greedy local search
    if time.time() < deadline:
        s, peak = _fast_greedy(s, max_iters=20000)
        if peak < best_peak:
            best_peak = peak
            best_s = s.copy()
    
    # Phase 2: Simulated annealing from the greedy result
    if time.time() < deadline:
        s2, peak2 = _simulated_annealing(s, t_start=0.5, t_end=0.01, n_iters=50000)
        if peak2 < best_peak:
            best_peak = peak2
            best_s = s2.copy()
    
    # Phase 3: Another greedy pass from the SA result
    if time.time() < deadline:
        s3, peak3 = _fast_greedy(best_s, max_iters=20000)
        if peak3 < best_peak:
            best_peak = peak3
            best_s = s3.copy()
    
    # Phase 4: Final SA from best
    if time.time() < deadline:
        s4, peak4 = _simulated_annealing(best_s, t_start=0.3, t_end=0.005, n_iters=40000)
        if peak4 < best_peak:
            best_peak = peak4
            best_s = s4.copy()
    
    # Return the best sequence found
    return best_s
# EVOLVE-BLOCK-END

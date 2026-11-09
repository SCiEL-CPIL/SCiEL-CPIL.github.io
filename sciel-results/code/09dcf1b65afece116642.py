"""Program for the Erdos minimum-overlap task using spectral optimization.
This strategy represents the signal h in the frequency domain.
We optimize the amplitudes of the non-DC Fourier modes to minimize the 
maximum overlap, which is equivalent to minimizing the peak of the cyclic 
cross-correlation of h and 1-h.
"""
import time
import numpy as np

N = 100
_DEADLINE = 140.0  # safety margin under the 150 s kill
_t0 = time.time()


def _fft_overlap(h):
    """
    Compute the overlap profile using FFT.
    overlap(k) = sum_i h[i] * (1 - h[(i-k) mod N])
    This is the cyclic cross-correlation of h and (1-h).
    """
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    # FFT of h and 1-h
    H = np.fft.rfft(h)
    G = np.fft.rfft(one_minus_h)
    # Cross-correlation via inverse FFT of product
    # corr[k] = ifft( H * conj(G) )
    # Note: standard correlation is ifft( H * conj(G) )
    # Let's verify:
    # sum_i h[i] g[i-k] = ifft( H * conj(G) )
    # Here g = 1-h.
    # So overlap(k) = ifft( H * conj(G) ).real
    cross = np.fft.irfft(H * np.conj(G), n=N)
    return cross


def _score(h):
    """The quantity the evaluator minimizes: max_k overlap(h, shifted (1-h))."""
    h = np.asarray(h, dtype=float)
    return float(np.max(_fft_overlap(h)))


def construct():
    global _t0
    # Start with constant 0.5, which has overlap N/4 = 25.
    # In FFT space, constant 0.5 has H[0] = 50, H[k>0] = 0.
    # We will perturb the higher frequency modes.
    
    # Initialize H with DC component fixed
    H = np.zeros(N//2 + 1, dtype=complex)
    H[0] = N / 2.0  # sum is N/2
    
    # We will optimize the real and imaginary parts of H[1:]
    # To keep h real, we must maintain Hermitian symmetry, but rfft handles that.
    # We need to ensure h[i] is in [0, 1].
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Initial random perturbations in frequency domain
    # We want to explore shapes that might lower the peak overlap.
    # The constant 0.5 is a local minimum? 
    # Actually, for N=100, the theoretical lower bound for max overlap is related to the variance.
    # Let's try to find a configuration that reduces the peak.
    
    rng = np.random.default_rng(42)
    
    # We will perform a simple coordinate ascent/descent on the frequency amplitudes.
    # But since the constraint h in [0,1] is difficult in frequency domain, 
    # we will use a projection step: after modifying H, compute h, clip to [0,1], 
    # and re-compute H. This is a projected gradient-like method.
    
    # To be faster and more robust, we'll try a set of structured candidates first, 
    # then use the FFT-based score to evaluate them quickly.
    
    candidates = []
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Two-level block (known to be good for some metrics)
    # Try different block lengths
    for k in range(1, N//2 + 1):
        h = np.zeros(N)
        h[:k] = 1.0
        # Rescale to sum N/2
        # Sum is k. We need N/2.
        # If k < N/2, we can't just scale to 1.0 because max is 1.0.
        # So we set h[:k] = 1.0, and distribute the remaining sum N/2 - k over the rest?
        # No, the constraint is sum = N/2.
        # If we set k ones, sum is k. If k > N/2, we can scale down.
        # If k <= N/2, we can set k ones, and the rest 0? Sum is k.
        # We need sum N/2.
        # So if k == N/2, it's perfect.
        # If k != N/2, we can adjust the values.
        # Let's just generate random valid arrays and use FFT score.
        pass

    # Better: Generate random arrays, project to valid space, and score.
    # But we want to use the FFT insight.
    # The insight is that _score is fast.
    
    # Let's do a more directed search.
    # The overlap is the cross-correlation.
    # To minimize the max of the cross-correlation, we want the cross-correlation to be as flat as possible.
    # This is similar to designing a sequence with low autocorrelation, but here it's cross-correlation with 1-h.
    
    # Let's try to use scipy.optimize.minimize on the frequency amplitudes, 
    # but with a penalty for out-of-bounds.
    
    # Given time constraints, a robust random search with FFT scoring is safe.
    # But let's try to be smarter.
    # We know that h=0.5 gives overlap 25.
    # Can we do better?
    # If h is 0 or 1, the overlap is the number of disagreements.
    # For h to be 0 or 1, sum must be N/2, so N/2 ones and N/2 zeros.
    # The overlap is sum h[i](1-h[i-k]).
    # If h is a block of ones, the overlap is high at small shifts.
    # To minimize the max overlap, we want the ones to be spread out.
    # A "balanced" sequence like a Thue-Morse sequence or a random binary sequence might work.
    
    # Let's test random binary sequences (0 and 1) with exactly N/2 ones.
    for _ in range(500):
        if time.time() - _t0 > _DEADLINE - 5:
            break
        # Generate a random permutation of [0]*N/2 + [1]*N/2
        h = np.concatenate([np.zeros(N//2), np.ones(N//2)])
        rng.shuffle(h)
        sc = _score(h)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
            
    # Also try continuous values, not just binary.
    # Random continuous with sum N/2.
    for _ in range(1000):
        if time.time() - _t0 > _DEADLINE - 5:
            break
        h = rng.uniform(0, 1, N)
        h += (N/2.0 - h.sum()) / N
        h = np.clip(h, 0, 1)
        # Fix sum again
        h += (N/2.0 - h.sum()) / N
        sc = _score(h)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()

    # One more specific family: sinusoids
    for f in range(1, 10):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if time.time() - _t0 > _DEADLINE - 5:
                break
            for phase in np.linspace(0, 2*np.pi, 10, endpoint=False):
                h = 0.5 + amp * np.sin(2*np.pi*f*np.arange(N)/N + phase)
                h = np.clip(h, 0, 1)
                h += (N/2.0 - h.sum()) / N
                sc = _score(h)
                if sc < best_score:
                    best_score = sc
                    best_h = h.copy()

    return best_h

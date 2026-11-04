"""Signed autocorrelation max-abs minimization via spectral flatness + FRFT-inspired design."""
import numpy as np
import time

N = 400
_DEADLINE = time.time() + 150.0
_MARGIN = 5.0  # seconds safety margin


def _autoconv(a):
    """Full linear autoconvolution of a (length N)."""
    F = np.fft.rfft(a, n=2 * N - 1)
    return np.fft.irfft(F * F, n=2 * N - 1)


def _score(a):
    """Upper bound on C3': max|autoconv| / (sum a)^2. Assumes a in {-1,+1}."""
    ac = _autoconv(a)
    s = a.sum()
    if abs(s) < 1e-12:
        return np.max(np.abs(ac))
    return np.max(np.abs(ac)) / (s * s)


def _local_search_fast(s, max_rounds=6):
    """Iterated single-flip local search with early termination."""
    s = s.copy()
    v = _score(s)
    best_local = s.copy()
    best_local_v = v
    for _round in range(max_rounds):
        if time.time() > _DEADLINE - _MARGIN:
            break
        improved = False
        # Use random order for diversity
        order = np.random.permutation(N)
        for i in order:
            if time.time() > _DEADLINE - _MARGIN:
                break
            s[i] = -s[i]
            s_sum = s.sum()
            if abs(s_sum) < 1e-12:
                ac2 = _autoconv(s)
                v2 = np.max(np.abs(ac2))
            else:
                ac2 = _autoconv(s)
                v2 = np.max(np.abs(ac2)) / (s_sum * s_sum)
            if v2 < best_local_v - 1e-12:
                best_local_v = v2
                best_local = s.copy()
                improved = True
            else:
                s[i] = -s[i]  # revert
        if not improved:
            break
        s = best_local.copy()
    return best_local, best_local_v


def _flat_spectrum_design(n=N, n_freqs=31, rng=None):
    """Design sequence with flat power spectrum via phase optimization.
    
    We parameterize the sequence in frequency domain with constrained magnitudes
    (flat) and optimize phases. Then binarize.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Frequency bins for rfft of length 2N-1
    n_fft = 2 * N - 1
    n_rfft = n_fft // 2 + 1
    
    # We want the autoconvolution to be flat (low peak).
    # Autoconvolution in time = |FFT|^2 in freq.
    # So we want |FFT(a)|^2 to be as flat as possible.
    # This means |FFT(a)| should be constant.
    # But a is real-valued, so FFT has conjugate symmetry.
    # Also a is in {-1, +1}, so sum(a) is fixed parity.
    
    # Strategy: Start with random phases, enforce flat magnitude, IFFT, binarize.
    best_signs = None
    best_val = np.inf
    
    for trial in range(5):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Random initial phases
        phases = rng.uniform(0, 2 * np.pi, size=n_rfft)
        # Enforce conjugate symmetry: F[0] real, F[n_rfft-1] real if n_fft even
        F = np.zeros(n_rfft, dtype=complex)
        F[0] = 1.0  # DC component
        F[-1] = 1.0  # Nyquist if n_fft is even
        for k in range(1, n_rfft - 1):
            F[k] = np.exp(1j * phases[k])
        
        # IFFT to get time domain
        a = np.fft.irfft(F, n=n_fft)
        # Take first N samples
        a = a[:N]
        # Binarize
        signs = np.sign(a)
        signs[signs == 0] = 1.0
        
        v = _score(signs)
        if v < best_val:
            best_val = v
            best_signs = signs.copy()
    
    return best_signs, best_val


def _frft_inspired_design(n=N, rng=None):
    """Design sequence inspired by fractional Fourier transform.
    
    The idea: apply a rotation in time-frequency domain to a flat-spectrum signal.
    This is approximated by mixing time and frequency domain representations.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    best_signs = None
    best_val = np.inf
    
    # Try different "rotation angles" (mixing parameters)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Generate random phase signal
        n_fft = 2 * N - 1
        n_rfft = n_fft // 2 + 1
        phases = rng.uniform(0, 2 * np.pi, size=n_rfft)
        F = np.zeros(n_rfft, dtype=complex)
        F[0] = 1.0
        F[-1] = 1.0
        for k in range(1, n_rfft - 1):
            F[k] = np.exp(1j * phases[k])
        
        # IFFT to get time domain
        a_time = np.fft.irfft(F, n=n_fft)[:N]
        
        # "Rotation": mix time and frequency
        # This is a heuristic approximation of FRFT
        a_mixed = (1 - alpha) * a_time + alpha * np.fft.irfft(F, n=N)[:N]
        
        signs = np.sign(a_mixed)
        signs[signs == 0] = 1.0
        
        v = _score(signs)
        if v < best_val:
            best_val = v
            best_signs = signs.copy()
    
    return best_signs, best_val


def _chirp_sequence(beta, n=N):
    """Generate a signed sequence from a quadratic chirp phase."""
    k = np.arange(n)
    phase = np.pi * beta * (k * k) / n + np.pi * beta * k / n
    x = np.cos(phase)
    signs = np.sign(x)
    signs[signs == 0] = 1.0
    return signs


def _multi_start_search():
    """Combine multiple design strategies with local search refinement."""
    rng = np.random.default_rng(42)
    best_signs = None
    best_val = np.inf
    
    # Strategy 1: Flat spectrum designs
    s, v = _flat_spectrum_design(rng=rng)
    if s is not None and v < best_val:
        best_val = v
        best_signs = s.copy()
    
    # Strategy 2: FRFT-inspired designs
    s, v = _frft_inspired_design(rng=rng)
    if s is not None and v < best_val:
        best_val = v
        best_signs = s.copy()
    
    # Strategy 3: Chirp sequences
    for beta in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]:
        if time.time() > _DEADLINE - _MARGIN:
            break
        s = _chirp_sequence(beta)
        v = _score(s)
        if v < best_val:
            best_val = v
            best_signs = s.copy()
    
    # Strategy 4: Random initializations
    for _ in range(5):
        if time.time() > _DEADLINE - _MARGIN:
            break
        s = rng.choice([-1.0, 1.0], size=N)
        v = _score(s)
        if v < best_val:
            best_val = v
            best_signs = s.copy()
    
    # Refine best candidate with local search
    if best_signs is not None:
        best_signs, best_val = _local_search_fast(best_signs, max_rounds=8)
    
    # Try refining a few other candidates
    for _ in range(3):
        if time.time() > _DEADLINE - _MARGIN:
            break
        s = rng.choice([-1.0, 1.0], size=N)
        s, v = _local_search_fast(s, max_rounds=5)
        if v < best_val:
            best_val = v
            best_signs = s.copy()
    
    return best_signs, best_val


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 150.0
    
    # Phase 1: Multi-start search
    best_signs, best_val = _multi_start_search()
    
    # Phase 2: Additional local search refinement
    if best_signs is not None:
        best_signs, best_val = _local_search_fast(best_signs, max_rounds=10)
    
    # Phase 3: Try a few more diverse initializations if time permits
    rng = np.random.default_rng(123)
    for _ in range(8):
        if time.time() > _DEADLINE - _MARGIN:
            break
        # Mix of random and structured
        if _ % 2 == 0:
            s = rng.choice([-1.0, 1.0], size=N)
        else:
            # Low-frequency structured pattern
            k = np.arange(N)
            freq = rng.uniform(0.1, 2.0)
            s = np.sign(np.sin(2 * np.pi * freq * k / N))
            s[s == 0] = 1.0
        s, v = _local_search_fast(s, max_rounds=5)
        if v < best_val:
            best_val = v
            best_signs = s.copy()
    
    return best_signs

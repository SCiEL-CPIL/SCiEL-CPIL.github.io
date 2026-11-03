"""
Spectral-approach program for the signed autocorrelation C3' task.

We model the autoconvolution via the circulant convolution matrix C (a real,
symmetric, positive-semidefinite matrix when using the appropriate embedding).
For a vector h of length N, the linear autoconvolution ac = h * h (linear) can be
related to the quadratic form h^T C_circ h where C_circ is a circulant matrix
whose first row is h (zero-padded to length 2N-1). However, the max absolute
value of the autoconvolution is not simply the largest eigenvalue.

Instead, we use the following insight:
- The autoconvolution at lag k is sum_i h[i] h[i+k].
- The maximum absolute autoconvolution is max_k |sum_i h[i] h[i+k]|.
- We want to minimize max_k |ac_k| / (sum h)^2.

Spectral approach:
- Consider the circulant matrix P whose first row is [h_0, 0, ..., 0, h_{N-1}, h_{N-2}, ..., h_1]
  This is the cyclic convolution matrix for h embedded in a circle of size 2N-1.
- The eigenvalues of P are lambda_j = DFT(h_padded)[j].
- The quadratic form h^T P h gives the sum of squares of the cyclic autoconvolution, not the max.

A more direct spectral optimization:
- We want to minimize max_k |ac_k| subject to (sum h)^2 = 1 (by scaling invariance).
- This is equivalent to finding h such that the autoconvolution is as flat as possible.
- For a circulant system, the optimal h would have autoconvolution proportional to delta (Kronecker),
  which corresponds to h having flat magnitude spectrum (like a Barker or Zadoff-Chu sequence).

We use the following strategy:
1. Generate candidate sequences with good autocorrelation properties (Zadoff-Chu, Barker-like,
   random with spectral flattening).
2. Use the spectral decomposition to compute the autoconvolution efficiently.
3. Apply local search in the spectral domain: adjust the phases of the DFT coefficients
   to flatten the autoconvolution.

Key insight: The autoconvolution in the time domain is the IDFT of |H|^2 where H is the DFT.
So max|ac| = max |IDFT(|H|^2)|. We want to minimize this.
This is equivalent to making |H|^2 as close to constant as possible (since IDFT of constant is delta).

So we optimize the phases of H (with fixed magnitudes from a good initial sequence) to
flatten |H|^2, which flattens the autoconvolution.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _zadoff_chu(n, root):
    """Generate a Zadoff-Chu sequence of length n with given root."""
    t = np.arange(n)
    return np.exp(-1j * np.pi * root * t * (t + 1) / n)

def _barker_like(n):
    """
    Generate a Barker-like sequence using a simple construction.
    Barker sequences have ideal autocorrelation properties.
    """
    # Use the length-13 Barker sequence as a template and repeat/modify
    barker13 = np.array([1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1])
    h = np.zeros(n)
    for i in range(n):
        h[i] = barker13[i % 13]
    return h

def _random_phase_sequence(n, seed=42):
    """
    Generate a sequence with random phases but structured magnitudes.
    This tends to have flatter autoconvolution.
    """
    rng = np.random.default_rng(seed)
    # Start with a random complex sequence
    h_complex = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    # Take real part for signed step function
    return np.real(h_complex)

def _spectral_flatten(h, max_iters=50, deadline=None):
    """
    Flatten the autoconvolution by adjusting phases in the spectral domain.
    
    The autoconvolution ac = IDFT(|H|^2) where H = DFT(h).
    We want to minimize max|ac|.
    
    Strategy: 
    1. Compute H = DFT(h)
    2. For each iteration, compute ac = IDFT(|H|^2)
    3. Find the lag k where |ac[k]| is maximum
    4. Adjust the phases of H to reduce the contribution at that lag
    
    This is a heuristic but can be effective.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    best_val = _score(h)
    best_h = h.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution
        ac = _autoconv_fft(h)
        max_idx = np.argmax(np.abs(ac))
        max_val = np.abs(ac[max_idx])
        
        if max_val < 1e-10:
            break
        
        # Try to reduce the peak at max_idx
        # The contribution to ac[max_idx] comes from pairs (i, i+max_idx)
        # We can try flipping signs of elements that contribute most to this peak
        
        # Get the DFT
        H = np.fft.rfft(h, fsize)
        
        # Compute the gradient of |ac[max_idx]| with respect to h
        # ac[max_idx] = sum_i h[i] h[i+max_idx] (with wraparound for linear conv)
        # For linear autoconvolution, ac[k] = sum_{i=0}^{N-1-k} h[i] h[i+k] for k >= 0
        # and ac[-k] = ac[k] by symmetry
        
        # Compute which pairs contribute to the peak
        if max_idx < N:
            # ac[max_idx] = sum_{i=0}^{N-1-max_idx} h[i] h[i+max_idx]
            contrib = np.zeros(N)
            for i in range(N - max_idx):
                contrib[i] += h[i + max_idx]
                contrib[i + max_idx] += h[i]
        else:
            # ac[N - max_idx] for negative lags
            lag = N - max_idx
            contrib = np.zeros(N)
            for i in range(N - lag):
                contrib[i] += h[i + lag]
                contrib[i + lag] += h[i]
        
        # Flip the element with largest |contrib| if it reduces the peak
        flip_idx = np.argmax(np.abs(contrib))
        h_test = h.copy()
        h_test[flip_idx] = -h_test[flip_idx]
        
        val_test = _score(h_test)
        if val_test < best_val:
            best_val = val_test
            best_h = h_test.copy()
            h = h_test
        else:
            # Try a small perturbation instead
            break
    
    return best_h, best_val

def _local_search_flip(h, max_passes=3, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 120s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Zadoff-Chu sequences (real part)
    for root in [1, 2, 3, 5, 7, 11]:
        zc = _zadoff_chu(N, root)
        inits.append(np.real(zc))
        inits.append(np.imag(zc))
    
    # 3. Barker-like sequence
    inits.append(_barker_like(N))
    
    # 4. Random phase sequences
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        inits.append(_random_phase_sequence(N, seed=seed))
    
    # 5. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(2 * np.pi * x / N))
    
    # 6. Chirp sequence
    phase = np.pi * x**2 / N
    inits.append(np.cos(phase))
    
    # Optimization loop
    for h_init in inits:
        if time.time() > deadline - 15:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1
        h_norm = h_init / s0
        
        # Spectral flattening
        h_flattened, val_flattened = _spectral_flatten(h_norm, max_iters=20, deadline=deadline-5)
        if val_flattened < best_score:
            best_score = val_flattened
            best_h = h_flattened.copy()
        
        # Local search refinement
        if time.time() < deadline - 5:
            h_refined, val_refined = _local_search_flip(best_h, max_passes=2, deadline=deadline-2)
            if val_refined < best_score:
                best_score = val_refined
                best_h = h_refined.copy()
    
    # Final local search refinement
    if time.time() < deadline - 3:
        h_refined, val_refined = _local_search_flip(best_h, max_passes=2, deadline=deadline-1)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END

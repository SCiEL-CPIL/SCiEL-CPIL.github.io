"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def _autoconv_max_abs(x):
    """Compute maximum absolute value of linear autoconvolution."""
    n = len(x)
    conv = np.convolve(x, x, mode='full')
    return np.max(np.abs(conv))


def _squared_integral(x):
    """Compute squared integral (sum of x squared, or (sum x)^2?)."""
    # The task says "divided by squared integral" - likely (sum x)^2
    return (np.sum(x)) ** 2


def _score(x):
    """Compute the score: max|autoconv| / (sum x)^2"""
    si = _squared_integral(x)
    if abs(si) < 1e-15:
        return np.inf
    return _autoconv_max_abs(x) / si


def construct():
    """
    Construct a signed step function by spectral shaping with random band-limited
    phase masks to minimize max|autoconvolution| / (sum x)^2.
    """
    best_x = None
    best_score = np.inf
    
    n_trials = 200
    
    for trial in range(n_trials):
        # Choose a random band width (number of frequency bins to activate)
        # Use a moderate band to balance peak reduction with DC content
        M = np.random.randint(10, N // 4 + 1)
        
        # Create frequency mask: activate M consecutive bins centered at DC
        # For real-valued output, we need Hermitian symmetry
        X = np.zeros(N, dtype=complex)
        
        # Determine which bins to activate
        # For real signal, X[k] = conj(X[N-k])
        # Activate bins 0, 1, ..., M//2 and their conjugates
        half_M = M // 2
        
        # Generate random phases for the positive frequency bins
        phases = np.random.uniform(0, 2 * np.pi, size=half_M + 1)
        
        # Set the activated bins
        for k in range(half_M + 1):
            if k == 0:
                # DC bin: real phase (0 or pi) to keep it real
                X[k] = np.exp(1j * phases[k])
                # For real output, X[0] should be real
                X[0] = np.sign(np.real(X[0])) if np.real(X[0]) != 0 else 1.0
            elif k < N - k:
                # Positive frequency
                X[k] = np.exp(1j * phases[k])
                # Enforce Hermitian symmetry
                X[N - k] = np.conj(X[k])
            else:
                # Nyquist bin (if N is even)
                X[k] = np.exp(1j * phases[k])
                # For real output, Nyquist should be real
                X[k] = np.sign(np.real(X[k])) if np.real(X[k]) != 0 else 1.0
        
        # Inverse DFT
        x = np.fft.ifft(X).real
        
        # Ensure non-zero sum
        s = np.sum(x)
        if abs(s) < 1e-10:
            continue
        
        # Compute score
        sc = _score(x)
        
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    # If no valid trial found, fall back to constant
    if best_x is None:
        best_x = np.ones(N) / N
    
    return best_x
# EVOLVE-BLOCK-END

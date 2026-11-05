"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 400

def _score_from_h(h):
    """Compute the task value from heights h."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))

def _params_to_h(params):
    """Convert frequency-domain parameters to real step heights.
    
    params layout:
      [0]: total amplitude scale (real, positive)
      [1:1+K]: magnitudes of K frequency bins (real, >= 0)
      [1+K:1+2K]: phases of K frequency bins (real, in [-pi, pi])
    
    We use a sparse DFT representation: h = IDFT of a spectrum with K non-zero bins.
    The DC bin is fixed to ensure sum(h) is controlled.
    """
    K = 10  # number of non-DC frequency bins
    amp = params[0]
    mags = params[1:1+K]
    phases = params[1+K:1+2*K]
    
    # Build spectrum: DC bin + K complex bins
    # For real h, spectrum must be Hermitian: H[-k] = conj(H[k])
    # We'll place bins at specific frequency indices.
    
    # Frequency indices to use (avoid DC=0 and Nyquist=N/2 for simplicity)
    freq_indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
    
    H = np.zeros(N, dtype=np.complex128)
    
    # Set DC to ensure reasonable sum
    H[0] = amp  # DC component
    
    for i, k in enumerate(freq_indices):
        H[k] = mags[i] * np.exp(1j * phases[i])
        H[N - k] = np.conj(H[k])  # Hermitian symmetry
    
    h = np.real(np.fft.ifft(H) * N)  # ifft includes 1/N, so multiply by N
    
    return h

def _objective(params):
    """Objective to minimize: the task value."""
    h = _params_to_h(params)
    return _score_from_h(h)

# EVOLVE-BLOCK-START

def construct():
    """Construct step heights using sparse frequency-domain optimization."""
    deadline = time.time() + 110  # safety margin before 120s limit
    
    K = 10  # number of non-DC frequency bins
    n_params = 1 + 2 * K  # 1 amplitude + K magnitudes + K phases
    
    # Bounds for each parameter
    bounds = [(0.1, 10.0)]  # amplitude
    for i in range(K):
        bounds.append((0.0, 5.0))   # magnitude
        bounds.append((-np.pi, np.pi))  # phase
    
    # Initial best: seed value
    best_h = np.ones(N)
    best_score = 2.0  # seed gives 2.0
    
    try:
        # Run differential evolution
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol',
            updating='immediate',
            callback=None,
            workers=-1
        )
        
        if result.success or result.fun < best_score:
            best_h = _params_to_h(result.x)
            best_score = _score_from_h(best_h)
        
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_h
    
    # Try a few more random restarts if time permits
    rng = np.random.RandomState(123)
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Random initial point
        x0 = np.array([
            rng.uniform(0.1, 10.0),
            *rng.uniform(0.0, 5.0, K),
            *rng.uniform(-np.pi, np.pi, K)
        ])
        
        try:
            result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=50,
                popsize=10,
                tol=1e-5,
                seed=restart + 100,
                polish=False,
                init=x0,
                updating='immediate',
                workers=-1
            )
            
            if result.fun < best_score:
                best_h = _params_to_h(result.x)
                best_score = _score_from_h(best_h)
                
        except Exception:
            continue
    
    return best_h

# EVOLVE-BLOCK-END

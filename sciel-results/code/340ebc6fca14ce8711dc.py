"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _autoconv_max_ratio(x):
    """Compute max|autoconvolution| / (sum x)^2 for a real signal x."""
    x = np.real(x)
    I = np.sum(x)
    if abs(I) < 1e-12:
        return 1e10
    n = len(x)
    size = 2 * n
    f = np.fft.fft(x, size)
    ac_full = np.fft.ifft(f * f, size)
    ac_full = np.real(ac_full)
    max_abs = np.max(np.abs(ac_full))
    return max_abs / (I * I)

def _zadoff_chu_sequence(N, root):
    """Generate a Zadoff-Chu sequence of length N with given root."""
    # ZC sequences are defined for prime lengths or lengths of form 4k+1, etc.
    # For general N, we can use a chirp-like sequence or pad.
    # Here we use a simple chirp: exp(j * pi * m * t^2 / N)
    # But ZC is better. For N=400, not prime.
    # We'll use a generic low-autocorrelation chirp.
    t = np.arange(N)
    phase = 2 * np.pi * root * t**2 / N
    return np.exp(1j * phase)

def _construct_signal(params):
    """
    params: [DC_offset, root]
    Returns real signal.
    """
    DC = params[0]
    root = params[1]
    # Generate complex chirp
    zc = _zadoff_chu_sequence(N, root)
    # Take real part
    ac_component = np.real(zc)
    # Normalize AC component to have unit variance or similar
    # We want to control the amplitude of the AC component
    # Let's scale it so that std is 1
    std = np.std(ac_component)
    if std > 1e-9:
        ac_component = ac_component / std
    # Signal = DC + AC
    x = DC + ac_component
    return x

def _objective(params):
    x = _construct_signal(params)
    return _autoconv_max_ratio(x)

def construct():
    global _START_TIME
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    
    # Strategy:
    # 1. The theoretical minimum for max|AC|/I^2 is 1/N, achieved by constant signals.
    # 2. We will search for signals close to constant but with optimized AC components
    #    to potentially reduce the max if the metric is sensitive to off-diagonals
    #    or to ensure we are at the global minimum.
    
    # Initialize with constant signal (known global min for full max metric)
    x_const = np.ones(N)
    best_score = _autoconv_max_ratio(x_const)
    best_x = x_const.copy()
    
    # Try to optimize DC and Chirp Root
    # DC should be large to minimize the ratio (if off-diagonals are non-zero)
    # If off-diagonals are zero, ratio is 1/N regardless of DC (as long as DC != 0).
    
    if remaining > 5.0:
        try:
            # Search over DC offset and Chirp Root
            # DC in [0.1, 10.0], Root in [0.1, 10.0]
            bounds = [(0.1, 10.0), (0.1, 10.0)]
            
            # Use a few random starts
            rng = np.random.default_rng(42)
            for i in range(10):
                if time.time() - _START_TIME > _DEADLINE - 2.0:
                    break
                x0 = rng.uniform(bounds[0][0], bounds[0][1], 1)
                x0 = np.append(x0, rng.uniform(bounds[1][0], bounds[1][1], 1))
                
                res = minimize(
                    _objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = _construct_signal(res.x)
        except Exception:
            pass
            
    return np.real(best_x)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _autoconv_bound(f):
    """Compute max_t |f*f(t)| for a step function f on [-1/4,1/4] with N intervals."""
    h = 1.0 / 200.0
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h
    return float(np.max(np.abs(conv)))

def _spectral_objective(params):
    """
    Objective function for spectral optimization.
    params: [amp_1, phase_1, amp_2, phase_2, ..., amp_k, phase_k]
    Returns the L-infinity norm of the autoconvolution of the discretized signal.
    """
    if time.time() > _DEADLINE:
        return 1e9
        
    k = len(params) // 2
    amps = params[:k]
    phases = params[k:]
    
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Construct continuous waveform
    waveform = np.zeros(N)
    for i in range(k):
        freq = i + 1
        waveform += amps[i] * np.sin(2 * np.pi * freq * t + phases[i])
        
    # Discretize to signed step function
    # We use a threshold at 0. If all same sign, adjust threshold slightly or return high penalty
    f = np.sign(waveform)
    
    # Handle edge case where all signs are the same (trivial solution with high autoconv)
    if np.all(f == 1) or np.all(f == -1):
        return 1e9
        
    return _autoconv_bound(f)

def construct():
    # We optimize a small number of harmonics.
    # More harmonics allow more complex shapes but increase optimization difficulty.
    # Let's try up to 5 harmonics.
    
    best_val = float('inf')
    best_params = None
    
    # Try different numbers of harmonics
    for k in [1, 2, 3, 4, 5]:
        if time.time() > _DEADLINE - 5:
            break
            
        # Initial guesses: random amplitudes and phases
        # We try a few random starts for each k
        for start_idx in range(5):
            if time.time() > _DEADLINE - 2:
                break
                
            rng = np.random.RandomState(start_idx * 100 + k)
            amps = rng.uniform(0.5, 1.5, size=k)
            phases = rng.uniform(0, 2 * np.pi, size=k)
            x0 = np.concatenate([amps, phases])
            
            try:
                # Use L-BFGS-B or Nelder-Mead? 
                # L-BFGS-B is generally faster for smooth-ish landscapes, but our objective 
                # is non-smooth due to the sign() and max() operations.
                # Nelder-Mead is derivative-free and robust to non-smoothness.
                # However, Nelder-Mead can be slow. Let's try a hybrid or just Nelder-Mead with tight bounds.
                
                # Actually, since the objective is piecewise constant (step function), 
                # gradient methods struggle. But the "underlying" continuous function is smooth.
                # The sign() makes it non-differentiable. 
                # A derivative-free method like Nelder-Mead is safer.
                
                res = minimize(
                    _spectral_objective, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5}
                )
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
                    
            except Exception:
                pass
                
    if best_params is None:
        # Fallback to alternating sequence
        return np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
        
    # Reconstruct the best candidate from the optimized parameters
    k = len(best_params) // 2
    amps = best_params[:k]
    phases = best_params[k:]
    
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    waveform = np.zeros(N)
    for i in range(k):
        freq = i + 1
        waveform += amps[i] * np.sin(2 * np.pi * freq * t + phases[i])
        
    f = np.sign(waveform)
    
    # Final check and local refinement if time permits
    if time.time() < _DEADLINE - 10:
        # Perform a quick local bit-flip search to clean up any discretization artifacts
        best_val_local = _autoconv_bound(f)
        improved = True
        while improved and time.time() < _DEADLINE - 2:
            improved = False
            for i in range(N):
                if time.time() > _DEADLINE - 2:
                    break
                f[i] = -f[i]
                val = _autoconv_bound(f)
                if val < best_val_local:
                    best_val_local = val
                    improved = True
                else:
                    f[i] = -f[i]
                    
    return f
# EVOLVE-BLOCK-END

"""
Strategy: Golay Complementary Pair Amplitude/Phase Optimization.

This approach differs structurally from the current program's non-harmonic sinusoidal
basis in two key ways:
1. **Representation**: Instead of arbitrary non-harmonic sinusoids, we use a sum of
   two Golay complementary sequences (e.g., length 256). Golay pairs are known to have
   perfect complementary autocorrelation (flat off-peak), which provides a very strong
   structural prior for low peak autocorrelation. We optimize the amplitudes and phases
   of these two sequences, as well as a small number of "correction" sinusoids to handle
   the remaining imperfections or to tune the DC component.
2. **Initialization**: We start from the exact Golay sequences scaled appropriately,
   rather than random sinusoids. This places the optimizer in a basin of attraction
   where the autocorrelation peak is already significantly lower than random or
   generic sinusoidal starts.

The objective is to minimize the peak of the autoconvolution of h, subject to
h having a non-zero sum (which we normalize to 1 for stability).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit

def _generate_golay_pair(length):
    """
    Generate a Golay complementary pair of length 'length' (power of 2).
    Returns two arrays of 1s and -1s.
    """
    if length < 2:
        raise ValueError("Length must be at least 2")
    if int(length) != length or (length & (length - 1)) != 0:
        raise ValueError("Length must be a power of 2")
    
    a = np.array([1])
    b = np.array([1])
    
    while len(a) < length:
        # Golay construction rule: a' = a, -b and b' = a, b
        a = np.concatenate([a, -b])
        b = np.concatenate([a, b])
        
    # Normalize to 1s and -1s (they are already, but just in case of float accumulation)
    a = np.sign(a)
    b = np.sign(b)
    return a, b

def _pad_to_n(seq, n):
    """
    Pad a sequence to length n by repeating and truncating.
    This allows using Golay pairs of length 256 for N=400.
    """
    if len(seq) >= n:
        return seq[:n]
    else:
        repeats = int(np.ceil(n / len(seq)))
        padded = np.tile(seq, repeats)
        return padded[:n]

def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _h_from_params(params):
    """
    Construct h from parameters.
    params = [A1, phi1, A2, phi2, A3, phi3, A4, phi4]
    A1, phi1: Amplitude and phase for Golay sequence 1
    A2, phi2: Amplitude and phase for Golay sequence 2
    A3, phi3: Amplitude and phase for correction sinusoid 1 (freq 1)
    A4, phi4: Amplitude and phase for correction sinusoid 2 (freq 2)
    """
    # Pre-computed Golay sequences of length 256, padded to 400
    # We use a fixed pair for speed, optimized via A and phi
    global _GOLAY1, _GOLAY2
    if not hasattr(_h_from_params, '_GOLAY1'):
        g1, g2 = _generate_golay_pair(256)
        _h_from_params._GOLAY1 = _pad_to_n(g1, N)
        _h_from_params._GOLAY2 = _pad_to_n(g2, N)
    
    A1 = params[0]
    phi1 = params[1]
    A2 = params[2]
    phi2 = params[3]
    A3 = params[4]
    phi3 = params[5]
    A4 = params[6]
    phi4 = params[7]
    
    # h = A1 * exp(i*phi1) * G1 + A2 * exp(i*phi2) * G2 + ...
    # But h must be real. So we use cos/sin or real/imag parts.
    # Let's use:
    # h = A1 * (cos(phi1) * G1 + sin(phi1) * G1_shifted?) 
    # Actually, simpler: h = A1 * G1 * cos(phi1) + A2 * G2 * cos(phi2) + ...
    # This restricts us to scaling and sign-flipping effectively if phi is 0 or pi.
    # To allow more flexibility, let's add the sinusoidal corrections properly.
    
    h = A1 * np.cos(phi1) * _h_from_params._GOLAY1 + A1 * np.sin(phi1) * np.roll(_h_from_params._GOLAY1, 1)
    h += A2 * np.cos(phi2) * _h_from_params._GOLAY2 + A2 * np.sin(phi2) * np.roll(_h_from_params._GOLAY2, 1)
    
    # Add low-frequency sinusoids to adjust DC and shape
    n = np.arange(N)
    h += A3 * np.cos(2 * np.pi * 1 * n / N + phi3)
    h += A4 * np.cos(2 * np.pi * 2 * n / N + phi4)
    
    return h

def _objective(params):
    h = _h_from_params(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    # Normalize h to have sum 1 to remove scale invariance and focus on shape
    h_norm = h / s
    conv = np.convolve(h_norm, h_norm)
    # We want to minimize max(conv). 
    # The task value is 2*N*max(conv)/s^2. If we normalize s=1, value is 2*N*max(conv).
    # So minimizing max(conv) is equivalent.
    return np.max(conv)

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initial best is the seed
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Generate Golay sequences
    g1, g2 = _generate_golay_pair(256)
    g1_padded = _pad_to_n(g1, N)
    g2_padded = _pad_to_n(g2, N)
    
    # Try different initializations for the optimization
    # 1. Pure Golay sum: h = g1 + g2
    # 2. Golay with random phases
    # 3. Golay with random amplitudes
    
    rng = np.random.default_rng(42)
    
    # Parameter structure: [A1, phi1, A2, phi2, A3, phi3, A4, phi4]
    # Bounds:
    # A: [0.1, 5.0]
    # phi: [0, 2*pi]
    bounds = [
        (0.1, 5.0), (0, 2*np.pi),
        (0.1, 5.0), (0, 2*np.pi),
        (0.0, 1.0), (0, 2*np.pi),
        (0.0, 1.0), (0, 2*np.pi)
    ]
    
    initializations = []
    
    # Init 1: Standard Golay sum
    initializations.append(np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    
    # Init 2: Random phases
    for _ in range(5):
        init = np.array([
            rng.uniform(0.5, 2.0), rng.uniform(0, 2*np.pi),
            rng.uniform(0.5, 2.0), rng.uniform(0, 2*np.pi),
            rng.uniform(0, 0.5), rng.uniform(0, 2*np.pi),
            rng.uniform(0, 0.5), rng.uniform(0, 2*np.pi)
        ])
        initializations.append(init)
        
    # Init 3: Antiphase Golay
    initializations.append(np.array([1.0, 0.0, 1.0, np.pi, 0.0, 0.0, 0.0, 0.0]))
    
    for init_params in initializations:
        if time.time() > deadline - 30:
            break
            
        try:
            result = minimize(
                _objective,
                init_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            
            if result.success:
                h_candidate = _h_from_params(result.x)
                v_candidate = _evaluate(h_candidate)
                
                if v_candidate < best_val:
                    best_val = v_candidate
                    best_h = h_candidate.copy()
                    
        except Exception:
            pass
            
    # Local refinement on the best solution found
    if time.time() < deadline - 10:
        h = best_h.copy()
        current_val = best_val
        step_size = 0.1
        rng_refine = np.random.default_rng(123)
        
        while time.time() < deadline - 5:
            improved = False
            for _ in range(100):
                idx = rng_refine.integers(0, N)
                h_new = h.copy()
                h_new[idx] += rng_refine.normal(0, step_size)
                
                v_new = _evaluate(h_new)
                if v_new < current_val:
                    current_val = v_new
                    h = h_new
                    improved = True
                    step_size *= 0.95
                else:
                    step_size = min(step_size * 1.05, 0.5)
                    
            if not improved:
                break
                
        if current_val < best_val:
            best_val = current_val
            best_h = h
            
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h


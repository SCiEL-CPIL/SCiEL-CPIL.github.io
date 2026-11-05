"""Zadoff-Chu inspired CAZAC approach with smooth-max optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0  # safety margin before 150s kill


def _value(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _smooth_max(x, tau=0.05):
    """Smooth approximation of max(x) using log-sum-exp."""
    m = np.max(x)
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))


def _smooth_value(h, tau=0.05):
    """Smooth version of the value for gradient-based optimization."""
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e6
    conv = np.convolve(h, h)
    sm = _smooth_max(conv, tau)
    return abs(2.0 * N * sm / (s * s))


def _zadoff_chu(N, root, shift=0):
    """Generate a Zadoff-Chu sequence of length N with given root and cyclic shift."""
    n = np.arange(N)
    zc = np.exp(1j * np.pi * root * (n * (2 * n + 1)) / N)
    # Cyclic shift
    zc = np.roll(zc, shift)
    return zc


def _build_from_zadoff(params):
    """Build h from Zadoff-Chu parameters: root, shift, amplitude scaling, sign flips."""
    root = int(params[0])
    shift = int(params[1]) % N
    amp = params[2]
    
    zc = _zadoff_chu(N, root, shift)
    h = amp * np.real(zc)
    return h


def _zadoff_objective(params):
    """Objective for optimizing Zadoff-Chu parameters."""
    h = _build_from_zadoff(params)
    if not np.all(np.isfinite(h)):
        return 1e6
    v = _smooth_value(h, tau=0.02)
    return v


def _optimize_zadoff(root, shift, deadline):
    """Optimize amplitude for a given Zadoff-Chu root and shift."""
    if time.time() > deadline:
        return None, np.inf
    
    # Start with amplitude = 1
    x0 = np.array([float(root), float(shift), 1.0])
    
    res = minimize(_zadoff_objective, x0, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
    
    if time.time() > deadline:
        return None, np.inf
    
    best_h = _build_from_zadoff(res.x)
    best_v = _value(best_h)
    return best_h, best_v


def _local_refine(h, deadline, max_iters=500):
    """Local refinement using projected gradient on the smooth objective."""
    if time.time() > deadline:
        return h, _value(h)
    
    best_h = h.copy()
    best_v = _value(h)
    
    # Use subgradient-style updates with smoothing
    tau = 0.01
    lr = 0.01
    s = np.sum(h)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        h_new = best_h.copy()
        # Compute gradient-like direction
        conv = np.convolve(h_new, h_new)
        sm = _smooth_max(conv, tau)
        val = abs(2.0 * N * sm / (s * s))
        
        # Perturb randomly
        h_new += np.random.normal(0, lr, N)
        
        v_new = _value(h_new)
        if v_new < best_v:
            best_h = h_new
            best_v = v_new
    
    return best_h, best_v


def _try_chirp_structure(deadline):
    """Try chirp-like structures with optimized parameters."""
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    # Try various chirp rates
    for rate in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        if time.time() > deadline:
            break
        
        n = np.arange(N)
        t = n / N
        # Real part of complex chirp
        h = np.cos(2 * np.pi * rate * t**2 + np.pi * 0.1)
        h = h - np.mean(h)  # Center it
        h = h / np.std(h) * 0.5  # Scale
        
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
    
    return best_h, best_v


def _try_dft_basis(deadline):
    """Try optimizing in DFT domain."""
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random frequency components
        k = rng.choice(N, size=10, replace=False)
        phases = rng.uniform(0, 2 * np.pi, 10)
        amps = rng.uniform(0.5, 1.5, 10)
        
        h = np.zeros(N)
        for ki, ph, am in zip(k, phases, amps):
            h += am * np.cos(2 * np.pi * ki * np.arange(N) / N + ph)
        
        h = h - np.mean(h)
        if np.std(h) < 1e-6:
            continue
        h = h / np.std(h) * 0.3
        
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
    
    return best_h, best_v


def construct():
    """Construct step heights using Zadoff-Chu inspired approach with optimization."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Start with seed
    best_h = np.ones(N)
    best_v = _value(best_h)  # Should be 2.0
    
    # Try Zadoff-Chu with various roots
    for root in [1, 3, 5, 7, 9, 11]:
        if time.time() > deadline:
            break
        
        for shift in range(0, N, 20):  # Sample shifts
            if time.time() > deadline:
                break
            
            h, v = _optimize_zadoff(root, shift, deadline)
            if h is not None and v < best_v:
                best_v = v
                best_h = h
    
    # Try chirp structures
    h_chirp, v_chirp = _try_chirp_structure(deadline)
    if v_chirp < best_v:
        best_v = v_chirp
        best_h = h_chirp
    
    # Try DFT basis
    h_dft, v_dft = _try_dft_basis(deadline)
    if v_dft < best_v:
        best_v = v_dft
        best_h = h_dft
    
    # Local refinement on the best found
    if time.time() < deadline:
        best_h, best_v = _local_refine(best_h, deadline, max_iters=1000)
    
    # Final check: if we didn't beat seed, return seed
    if best_v >= 2.0:
        return np.ones(N)
    
    return best_h

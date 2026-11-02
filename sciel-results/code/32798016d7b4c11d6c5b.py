import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    # Autoconvolution (linear, not circular)
    autoconv = np.convolve(h, h, mode='full')
    sup_norm = np.max(np.abs(autoconv))
    if sup_norm <= 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _construct_from_atoms(centers, widths, amplitudes):
    """
    Construct a height vector from a set of atoms.
    Each atom is a triangular pulse: max(0, 1 - |x - center| / width) * amplitude
    The grid is on [-1/4, 1/4] with N=50 points.
    """
    x = np.linspace(-0.25, 0.25, N)
    h = np.zeros(N, dtype=np.float64)
    for c, w, a in zip(centers, widths, amplitudes):
        if w <= 0:
            continue
        h += a * np.maximum(0, 1 - np.abs(x - c) / w)
    return h

def _encode_params(params, num_atoms):
    """Decode parameter vector into centers, widths, amplitudes."""
    # Each atom has 3 params: center, width, amplitude
    centers = []
    widths = []
    amplitudes = []
    for i in range(num_atoms):
        c = params[3*i]
        w = params[3*i+1]
        a = params[3*i+2]
        centers.append(c)
        widths.append(abs(w) + 1e-6)  # ensure positive width
        amplitudes.append(abs(a))  # ensure non-negative
    return centers, widths, amplitudes

def _negative_ratio(params, num_atoms):
    """Objective function for minimization (negative of ratio)."""
    centers, widths, amplitudes = _encode_params(params, num_atoms)
    h = _construct_from_atoms(centers, widths, amplitudes)
    ratio = _compute_ratio(h)
    return -ratio

def _try_num_atoms(num_atoms, deadline):
    """Try to optimize with a given number of atoms."""
    if time.time() > deadline:
        return None, -np.inf
    
    # Initial guess: spread atoms across the domain
    best_ratio = -np.inf
    best_params = None
    
    # Try a few random initializations
    n_starts = min(5, max(1, int((deadline - time.time()) / 2)))
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Random initialization
        np.random.seed(start_idx * 1000 + num_atoms)
        centers_init = np.random.uniform(-0.25, 0.25, num_atoms)
        widths_init = np.random.uniform(0.01, 0.1, num_atoms)
        amplitudes_init = np.random.uniform(0.1, 1.0, num_atoms)
        
        params_init = np.concatenate([centers_init, widths_init, amplitudes_init])
        
        # Bounds for differential evolution
        bounds = []
        for i in range(num_atoms):
            bounds.append((-0.25, 0.25))  # center
            bounds.append((0.005, 0.2))   # width
            bounds.append((0.01, 5.0))    # amplitude
        
        # Use differential evolution for global search
        try:
            result = differential_evolution(
                _negative_ratio,
                bounds,
                args=(num_atoms,),
                maxiter=50,
                popsize=10,
                tol=1e-4,
                seed=start_idx,
                polish=False,
                init='latinhypercube'
            )
            if result.fun < -best_ratio:
                best_ratio = -result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        return None, -np.inf
    
    centers, widths, amplitudes = _encode_params(best_params, num_atoms)
    h = _construct_from_atoms(centers, widths, amplitudes)
    ratio = _compute_ratio(h)
    
    return h, ratio

def construct():
    """
    Construct 50 nonnegative heights to maximize the ratio.
    Uses an atom-based representation with optimized positions, widths, and amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    best_h = None
    best_ratio = -np.inf
    
    # Try different numbers of atoms
    # Start with fewer atoms and increase
    for num_atoms in [1, 2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline:
            break
        
        h, ratio = _try_num_atoms(num_atoms, deadline)
        if h is not None and ratio > best_ratio:
            best_ratio = ratio
            best_h = h
    
    # If we didn't find anything good, fall back to a simple approach
    if best_h is None:
        # Single spike at center
        x = np.linspace(-0.25, 0.25, N)
        h_shape = np.maximum(0, 1 - np.abs(x) / 0.05)
        best_h = h_shape
        best_ratio = _compute_ratio(h_shape)
    
    # Normalize to have L1 norm = 1 (the ratio is scale-invariant, but let's be consistent)
    l1 = np.sum(best_h)
    if l1 > 0:
        best_h = best_h / l1
    
    # Ensure non-negative
    best_h = np.maximum(0, best_h)
    
    return best_h.tolist()

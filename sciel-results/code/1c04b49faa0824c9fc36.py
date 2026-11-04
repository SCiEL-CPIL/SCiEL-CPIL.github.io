"""Second autocorrelation inequality: maximize squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def _autoconvolution_peak(h):
    """Compute the supremum norm of the discrete autoconvolution of h."""
    size = 2 * N - 1
    h_pad = np.zeros(2 * size)
    h_pad[:N] = h
    H = np.fft.fft(h_pad, n=2 * size)
    auto = np.fft.ifft(H * H).real[:size]
    return np.max(np.abs(auto))

def _compute_ratio(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    peak = _autoconvolution_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _h_from_spectrum(P):
    """
    Given a complex array P of length N (the DFT coefficients),
    compute h = IFFT(|P|^2) normalized to have sum 1.
    This ensures h >= 0 since |P|^2 is non-negative and IFFT of 
    a real non-negative sequence... actually IFFT(|P|^2) is real 
    only if |P|^2 is conjugate-symmetric. We handle this by 
    constructing h directly as a non-negative sequence.
    
    Better approach: parameterize h directly via log-amplitudes 
    and phases, but ensure non-negativity.
    
    Actually, the cleanest DFT parameterization: 
    h_k = |P_k|^2 / sum(|P|^2) where P_k are complex DFT bins.
    But IFFT(|P|^2) gives h in time domain which may not be non-negative.
    
    Let's use a different structural approach: optimize the 
    amplitude and phase of DFT coefficients such that the 
    inverse transform is non-negative. We'll use softplus 
    on the time-domain values after IFFT.
    """
    # P is complex array of length N
    power = np.abs(P) ** 2
    # IFFT of power spectrum gives time-domain signal
    h_raw = np.fft.ifft(power).real
    # Softplus to ensure non-negativity
    h = np.log1p(np.exp(h_raw))
    h = np.maximum(h, 1e-15)
    total = np.sum(h)
    if total < 1e-15:
        return np.ones(N) / N
    return h / total

def _ratio_from_P(P):
    """Compute ratio from DFT parameterization."""
    h = _h_from_spectrum(P)
    return _compute_ratio(h)

def _neg_ratio_from_P(P):
    """Negative ratio for minimization."""
    return -_ratio_from_P(P)

def _initialize_P(h_initial):
    """Convert a time-domain signal to DFT parameterization."""
    h = np.asarray(h_initial, dtype=float)
    P = np.fft.fft(h)
    return P

def _optimize_dft(P0, deadline):
    """
    Optimize in DFT space. P is complex, so we flatten to real vector 
    of length 2N (real and imaginary parts).
    """
    P0 = np.asarray(P0, dtype=complex)
    x0 = np.concatenate([P0.real, P0.imag])
    
    def obj(x):
        P = np.empty(N, dtype=complex)
        P.real = x[:N]
        P.imag = x[N:]
        return -_ratio_from_P(P)
    
    best_x = x0.copy()
    best_val = -obj(x0)
    
    try:
        result = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        val = -result.fun
        if val > best_val:
            best_val = val
            best_x = result.x.copy()
    except Exception:
        pass
    
    P_best = np.empty(N, dtype=complex)
    P_best.real = best_x[:N]
    P_best.imag = best_x[N:]
    h_best = _h_from_spectrum(P_best)
    return h_best, best_val

def _optimize_dft_polish(P0, deadline):
    """Higher precision polish in DFT space."""
    P0 = np.asarray(P0, dtype=complex)
    x0 = np.concatenate([P0.real, P0.imag])
    
    def obj(x):
        P = np.empty(N, dtype=complex)
        P.real = x[:N]
        P.imag = x[N:]
        return -_ratio_from_P(P)
    
    best_x = x0.copy()
    best_val = -obj(x0)
    
    try:
        result = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
        )
        val = -result.fun
        if val > best_val:
            best_val = val
            best_x = result.x.copy()
    except Exception:
        pass
    
    P_best = np.empty(N, dtype=complex)
    P_best.real = best_x[:N]
    P_best.imag = best_x[N:]
    h_best = _h_from_spectrum(P_best)
    return h_best, best_val

def _direct_optimize(h0, deadline):
    """Direct optimization in time domain as fallback."""
    h0 = np.asarray(h0, dtype=float).copy()
    bounds = [(0, None)] * N
    
    def obj(h):
        return -_compute_ratio(h)
    
    best_h = h0.copy()
    best_val = _compute_ratio(h0)
    
    try:
        result = minimize(
            obj,
            h0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        val = _compute_ratio(result.x)
        if val > best_val:
            best_val = val
            best_h = result.x.copy()
    except Exception:
        pass
    
    return best_h, best_val

def _make_gaussian(center, sigma):
    x = np.arange(N, dtype=float)
    h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return h / np.sum(h)

def _make_sidon_spikes(k, seed=42):
    """Create sparse Sidon-like spikes."""
    rng = np.random.RandomState(seed)
    best_positions = None
    best_score = -1
    
    for attempt in range(100):
        positions = sorted(rng.choice(N, size=k, replace=False))
        sums = set()
        valid = True
        for i in range(len(positions)):
            for j in range(i, len(positions)):
                s = positions[i] + positions[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        if valid:
            min_dist = min(abs(positions[i] - positions[j]) 
                         for i in range(len(positions)) 
                         for j in range(i+1, len(positions)))
            if min_dist > best_score:
                best_score = min_dist
                best_positions = positions
    
    if best_positions is None:
        best_positions = list(np.linspace(2, N-3, k).astype(int))
    
    h = np.zeros(N)
    for p in best_positions:
        h[p] = 1.0
    return h / np.sum(h)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = -1
    
    # Generate diverse initializations
    inits = []
    
    # Single Gaussian at center
    inits.append(_make_gaussian(24.5, 3.0))
    
    # Two separated Gaussians
    x = np.arange(N, dtype=float)
    h_bi = np.exp(-0.5 * ((x - 10) / 2.5) ** 2) + np.exp(-0.5 * ((x - 40) / 2.5) ** 2)
    inits.append(h_bi / np.sum(h_bi))
    
    # Three peaks
    h_tri = (np.exp(-0.5 * ((x - 6) / 2.0) ** 2) + 
             np.exp(-0.5 * ((x - 25) / 2.0) ** 2) + 
             np.exp(-0.5 * ((x - 44) / 2.0) ** 2))
    inits.append(h_tri / np.sum(h_tri))
    
    # Sidon sparse spikes
    for k in [4, 5, 6]:
        for seed in [42, 123, 999]:
            inits.append(_make_sidon_spikes(k, seed))
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    # For each initialization, try DFT-space optimization
    for h0 in inits:
        if time.time() > deadline - 10:
            break
        
        # Try DFT parameterization
        P0 = _initialize_P(h0)
        h_dft, val_dft = _optimize_dft(P0, deadline - 5)
        if val_dft > best_val:
            best_val = val_dft
            best_h = h_dft.copy()
        
        # Also try direct optimization as cross-check
        if time.time() < deadline - 8:
            h_direct, val_direct = _direct_optimize(h0, deadline - 5)
            if val_direct > best_val:
                best_val = val_direct
                best_h = h_direct.copy()
    
    # Polish the best result in both spaces
    if best_h is not None and time.time() < deadline - 5:
        # DFT-space polish
        P_best = _initialize_P(best_h)
        h_polished, val_polished = _optimize_dft_polish(P_best, deadline - 2)
        if val_polished > best_val:
            best_val = val_polished
            best_h = h_polished.copy()
        
        # Direct-space polish
        if time.time() < deadline - 3:
            h_direct_polish, val_dp = _direct_optimize(best_h, deadline - 2)
            if val_dp > best_val:
                best_val = val_dp
                best_h = h_direct_polish.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = _compute_ratio(best_h)
    
    # Ensure nonnegativity and return as list
    best_h = np.maximum(best_h, 0)
    return list(best_h)

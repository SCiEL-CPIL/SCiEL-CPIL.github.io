"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_objective_and_grad(theta):
    """
    Direct spectral optimization: minimize max_k |H[k]|^2 / |H[0]|^2
    where H = DFT(h) and theta parameterizes h via h = IDFT(exp(1j*theta) * |H_init|)
    
    We work in the frequency domain. Let H[k] be the DFT of h.
    The circular autoconvolution is IDFT(|H|^2).
    We want to minimize max_k |IDFT(|H|^2)[k]| / |H[0]|^2.
    
    Since H[0] = sum(h) is fixed by normalization, we minimize max of the
    circular autoconvolution spectrum.
    
    Parameterize H[k] = A[k] * exp(1j * phi[k]) where A[k] >= 0.
    For real h, H[N-k] = conj(H[k]).
    
    We optimize over the magnitudes A[k] (with symmetry) to equalize the
    time-domain peaks.
    """
    n = N
    # theta contains the log-magnitudes of the positive-frequency DFT bins
    # theta[0] corresponds to DC, theta[1..n//2] to positive frequencies
    # We enforce H[0] = 1 (normalize later)
    
    num_bins = n // 2 + 1
    if len(theta) < num_bins:
        theta = np.pad(theta, (0, num_bins - len(theta)))
    theta = theta[:num_bins]
    
    # Reconstruct H with unit DC
    A = np.exp(theta - theta[0])  # Normalize so A[0] = 1
    H = np.zeros(n, dtype=complex)
    H[:num_bins] = A
    for k in range(1, n - num_bins + 1):
        H[n - k] = A[k]
    H[0] = 1.0  # DC component fixed
    
    # Compute circular autoconvolution: IDFT(|H|^2)
    power = np.abs(H)**2
    ac = np.fft.ifft(power).real  # Should be real
    
    # Max absolute value
    max_ac = np.max(np.abs(ac))
    
    # The objective is max_ac / |H[0]|^2 = max_ac / 1 = max_ac
    # But we need to convert back to time-domain h and compute the actual linear score
    h = np.fft.ifft(H).real
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(theta)
    
    # For gradient, use automatic differentiation or numerical
    # Here we use the spectral insight: the objective in frequency domain
    # is max_k |IDFT(|H|^2)[k]|
    
    return max_ac, np.zeros_like(theta)

def _optimize_spectral():
    """
    Optimize the spectral distribution to minimize the maximum autoconvolution peak.
    Uses the fact that circular autoconvolution = IDFT(|DFT(h)|^2).
    """
    t_start = time.time()
    deadline = t_start + 90.0
    n = N
    num_bins = n // 2 + 1
    
    # Initial guess: flat spectrum (random phase, equal magnitude)
    # This gives a delta-like autoconvolution
    best_score = np.inf
    best_h = np.ones(n) / n
    best_theta = np.zeros(num_bins)
    
    # Strategy: Start with various spectral initializations
    initializations = []
    
    # 1. Flat magnitude spectrum
    initializations.append(np.zeros(num_bins))
    
    # 2. Chirp-like in time -> linear phase in frequency
    for alpha in [0.5, 1.0, 2.0]:
        i = np.arange(n)
        h_init = np.cos(2 * np.pi * alpha * i**2 / n)
        h_init = h_init / np.sum(h_init)
        H_init = np.fft.fft(h_init)
        # Extract magnitudes
        mags = np.abs(H_init[:num_bins])
        mags = mags / mags[0] if mags[0] > 1e-9 else mags
        theta_init = np.log(np.maximum(mags, 1e-10))
        initializations.append(theta_init)
    
    # 3. Random phases with flat magnitude
    np.random.seed(42)
    for trial in range(3):
        theta_init = np.zeros(num_bins)
        initializations.append(theta_init)
    
    for theta_init in initializations:
        if time.time() > deadline - 10:
            break
        
        theta_init = theta_init.copy()
        
        def obj_and_grad(x):
            # x is the log-magnitudes
            A = np.exp(x - x[0])  # Normalize DC to 1
            H = np.zeros(n, dtype=complex)
            H[:num_bins] = A
            for k in range(1, n - num_bins + 1):
                H[n - k] = A[k]
            H[0] = 1.0
            
            power = np.abs(H)**2
            ac = np.fft.ifft(power).real
            max_ac = np.max(np.abs(ac))
            
            # Convert to h and compute actual score
            h = np.fft.ifft(H).real
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10, np.zeros_like(x)
            
            # For the optimization, we use the circular autoconv max as proxy
            # The actual linear score is what matters, but circular is smoother
            return max_ac, np.zeros_like(x)
        
        try:
            result = minimize(
                obj_and_grad,
                theta_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result.fun < best_score:
                # Convert back to h
                A = np.exp(result.x - result.x[0])
                H = np.zeros(n, dtype=complex)
                H[:num_bins] = A
                for k in range(1, n - num_bins + 1):
                    H[n - k] = A[k]
                H[0] = 1.0
                
                h = np.fft.ifft(H).real
                s = np.sum(h)
                if abs(s) > 1e-9:
                    h = h / s
                    score = _score(h)
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
                        best_theta = result.x.copy()
        except Exception:
            continue
    
    return best_h, best_score

def _local_search(h, max_iters=3):
    """Local search via sign flips."""
    t_start = time.time()
    deadline = t_start + 20.0
    h = h.copy()
    best_score = _score(h)
    
    for _ in range(max_iters):
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            h[idx] = -h[idx]
            new_val = _score(h)
            if new_val < best_score:
                best_score = new_val
                improved = True
            else:
                h[idx] = -h[idx]
        if not improved:
            break
    
    return h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Strategy 1: Spectral optimization
    if time.time() < deadline - 30:
        h_spec, score_spec = _optimize_spectral()
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Local search refinement
    if time.time() < deadline - 5:
        h_refined, score_refined = _local_search(best_h, max_iters=3)
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined.copy()
    
    return best_h

# EVOLVE-BLOCK-END

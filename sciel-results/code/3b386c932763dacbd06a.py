"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score_from_spectrum(H):
    """
    Given the full DFT H (length N), compute max|autoconv| / (sum h)^2.
    autoconv at lag k = h * shift_k(h) = inverse FFT of |H|^2 at position k.
    Actually, linear autoconv uses zero-padding. But for the ratio, we can work
    with circular autoconv if we're careful. Let's use the direct approach:
    max|autoconv_linear| / (sum h)^2
    
    But in frequency domain: linear autoconv spectrum is |H_pad|^2 where H_pad is
    FFT of zero-padded h. This is complex. Let's just compute in time domain for scoring,
    but optimize in frequency domain.
    """
    h = np.fft.irfft(H, N) if H.shape[0] == N//2+1 else np.fft.ifft(H).real
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return np.max(np.abs(ac)) / (s * s)

def _freq_objective(x):
    """
    Objective in frequency domain. x contains:
    - x[0] = H[0].real (DC, real for real h)
    - x[1] = H[1].real, x[2] = H[1].imag
    - ... for k=1 to N//2
    For real-valued h, H[k] = conj(H[N-k]), so we only need k=0..N//2.
    
    We minimize max_k |H[k]|^2 / H[0]^2 by normalizing H[0]=1 and minimizing max_{k>0} |H[k]|^2.
    """
    # Reconstruct H from x
    n_half = N // 2
    H = np.zeros(N, dtype=complex)
    H[0] = x[0]  # DC component, real
    for k in range(1, n_half + 1):
        H[k] = x[2*k - 1] + 1j * x[2*k]
        H[N - k] = np.conj(H[k])
    
    # Normalize so that sum(h) = H[0] = 1 (we'll enforce this via the optimization)
    # The objective is max_k |H[k]|^2 / H[0]^2
    H0 = H[0]
    if abs(H0) < 1e-12:
        return 1e10
    
    # Compute max |H[k]|^2 for k != 0
    max_abs_sq = np.max(np.abs(H[1:])**2)
    
    # The ratio is max|autoconv|/(sum h)^2. 
    # For circular convolution, this would be max|H|^2 / H[0]^2.
    # For linear convolution with zero padding, it's different.
    # Let's use the actual linear autoconv computation for accuracy.
    h = np.fft.ifft(H).real
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return np.max(np.abs(ac)) / (s * s)

def _freq_optimize():
    """
    Optimize in frequency domain. We want to minimize max|autoconv|/(sum h)^2.
    
    Key insight: if we can make H[0] large and all other H[k] small, the ratio is small.
    The constraint is that h must be real, which means H[k] = conj(H[N-k]).
    
    Strategy: parameterize the non-DC components and minimize their maximum magnitude,
    subject to H[0] being fixed (or normalized).
    """
    t_start = time.time()
    deadline = t_start + 90.0
    
    n_half = N // 2
    # Variables: H[0].real, then for k=1..n_half: H[k].real, H[k].imag
    # Total variables: 1 + 2*n_half
    n_vars = 1 + 2 * n_half
    
    # Initial guess: flat spectrum H[k] = 1/N for all k
    x0 = np.zeros(n_vars)
    x0[0] = 1.0  # H[0] = 1
    for k in range(1, n_half + 1):
        x0[2*k - 1] = 1.0 / N  # H[k].real
        x0[2*k] = 0.0           # H[k].imag
    
    best_x = x0.copy()
    best_score = _freq_objective(x0)
    
    # Multiple initializations
    initializations = []
    
    # 1. Flat spectrum
    initializations.append(x0.copy())
    
    # 2. Decaying spectrum: H[k] = c / (1 + k^alpha)
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        for k in range(1, n_half + 1):
            x_init[2*k - 1] = 1.0 / (1.0 + k**alpha)
            x_init[2*k] = 0.0
        initializations.append(x_init)
    
    # 3. Cosine decay: H[k] = cos(pi*k/(2*n_half))
    x_init = np.zeros(n_vars)
    x_init[0] = 1.0
    for k in range(1, n_half + 1):
        x_init[2*k - 1] = np.cos(np.pi * k / (2 * n_half))
        x_init[2*k] = 0.0
    initializations.append(x_init)
    
    # 4. Random with small non-DC
    rng = np.random.default_rng(42)
    for _ in range(3):
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        x_init[1:] = rng.normal(0, 0.01, size=2*n_half)
        initializations.append(x_init)
    
    for x_init in initializations:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _freq_objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Convert best frequency-domain solution to time domain
    H = np.zeros(N, dtype=complex)
    H[0] = best_x[0]
    for k in range(1, n_half + 1):
        H[k] = best_x[2*k - 1] + 1j * best_x[2*k]
        H[N - k] = np.conj(H[k])
    
    h = np.fft.ifft(H).real
    
    # Local search: perturb individual frequency components
    if time.time() < deadline - 15:
        x_local = best_x.copy()
        best_local = best_score
        
        for _ in range(10):
            improved = False
            # Try small perturbations on each variable
            for idx in range(1, n_vars):  # Skip DC
                if time.time() > deadline - 5:
                    break
                for delta in [0.001, -0.001, 0.01, -0.01]:
                    x_local[idx] += delta
                    new_val = _freq_objective(x_local)
                    if new_val < best_local:
                        best_local = new_val
                        improved = True
                    else:
                        x_local[idx] -= delta
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_x = x_local.copy()
            # Reconstruct h
            H = np.zeros(N, dtype=complex)
            H[0] = best_x[0]
            for k in range(1, n_half + 1):
                H[k] = best_x[2*k - 1] + 1j * best_x[2*k]
                H[N - k] = np.conj(H[k])
            h = np.fft.ifft(H).real
    
    return h, best_score

def _time_domain_refine(h):
    """Local search in time domain with sign flips and small perturbations."""
    t_start = time.time()
    deadline = t_start + 25.0
    
    h_refined = h.copy()
    best_local = _score(h_refined)
    
    # Sign flips
    for _ in range(5):
        improved = False
        for idx in range(N):
            if time.time() > deadline - 5:
                break
            h_refined[idx] = -h_refined[idx]
            new_val = _score(h_refined)
            if new_val < best_local:
                best_local = new_val
                improved = True
            else:
                h_refined[idx] = -h_refined[idx]
        if not improved:
            break
    
    # Small additive perturbations
    for _ in range(3):
        improved = False
        rng = np.random.default_rng(int(time.time() * 1000) % 10000)
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            idx = rng.integers(0, N)
            delta = rng.normal(0, 0.01)
            h_refined[idx] += delta
            new_val = _score(h_refined)
            if new_val < best_local:
                best_local = new_val
                improved = True
            else:
                h_refined[idx] -= delta
        if not improved:
            break
    
    return h_refined, best_local

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 120s limit
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Strategy 1: Frequency-domain optimization
    if time.time() < deadline - 30:
        h_freq, score_freq = _freq_optimize()
        if score_freq < best_score:
            best_score = score_freq
            best_h = h_freq.copy()
    
    # Strategy 2: Time-domain local refinement
    if time.time() < deadline - 5:
        h_ref, score_ref = _time_domain_refine(best_h)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
    
    return best_h

# EVOLVE-BLOCK-END

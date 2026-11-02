import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    # Normalization: score = max_conv * n / (total^2 * w)
    score = max_conv * n / (total * total * w)
    return score

def _spectral_to_heights(params):
    """Convert spectral parameters to step heights.
    
    params: [sigma, bump_center, bump_width, bump_amp, offset]
    - sigma: width of main Gaussian in frequency domain
    - bump_center: center of secondary bump in frequency domain  
    - bump_width: width of secondary bump
    - bump_amp: amplitude of secondary bump
    - offset: DC offset adjustment
    
    Returns nonnegative heights normalized to sum=1.
    """
    sigma = params[0]
    bump_center = params[1]
    bump_width = params[2]
    bump_amp = params[3]
    offset = params[4]
    
    # Frequency indices for rfft of length N
    freqs = np.fft.rfftfreq(N, d=1.0/N)  # 0 to 0.5
    # Scale to [0, 1] for easier parameterization
    freq_norm = freqs / 0.5  # now in [0, 1]
    
    # Main Gaussian spectrum
    spec = np.exp(-0.5 * (freq_norm / sigma) ** 2)
    
    # Add secondary bump
    spec += bump_amp * np.exp(-0.5 * ((freq_norm - bump_center) / bump_width) ** 2)
    
    # Apply offset (shifts the spectrum)
    spec = np.maximum(spec + offset, 0.0)
    
    # Convert spectrum to time-domain heights via inverse FFT
    # spec is |F|^2, so F = sqrt(spec) * exp(i*phase)
    # For real signals, use Hermitian symmetry
    F_mag = np.sqrt(spec)
    # Create complex F with zero phase (symmetric signal)
    F_complex = np.zeros(N, dtype=np.complex128)
    F_complex[:len(F_mag)] = F_mag
    # Enforce Hermitian symmetry for real signal
    if N > 1:
        F_complex[1:] = np.conj(F_complex[N-1:0:-1])
    
    h = np.fft.ifft(F_complex).real
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-10:
        h = np.ones(N) / N
    else:
        h /= np.sum(h)
    
    return h

def _spectral_objective(params, alpha=100.0):
    """Compute smooth max autoconvolution from spectral parameters."""
    h = _spectral_to_heights(params)
    
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    # Smooth max via log-sum-exp
    m = np.max(conv)
    if m < 0:
        m = 0
    exp_conv = np.exp(alpha * (conv - m))
    sm = m + np.log(np.sum(exp_conv)) / alpha
    
    return sm

def _spectral_grad(params, alpha=100.0):
    """Compute gradient of smooth max autoconvolution w.r.t. spectral params."""
    h = _spectral_to_heights(params)
    
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    m = np.max(conv)
    if m < 0:
        m = 0
    exp_conv = np.exp(alpha * (conv - m))
    sm = m + np.log(np.sum(exp_conv)) / alpha
    weights = exp_conv / np.sum(exp_conv)
    
    # Gradient w.r.t. h
    W = np.fft.rfft(weights)
    H = np.fft.rfft(h)
    corr = np.fft.irfft(W * np.conj(H), n=M)
    grad_h = 2.0 * corr[:N]
    
    # Now chain rule: d(sm)/d(params) = sum_i grad_h[i] * d(h[i])/d(params)
    # Use finite differences for the spectral-to-heights mapping
    grad_params = np.zeros(len(params))
    eps = 1e-6
    for j in range(len(params)):
        params_plus = params.copy()
        params_plus[j] += eps
        h_plus = _spectral_to_heights(params_plus)
        dh_dparam = (h_plus - h) / eps
        grad_params[j] = np.dot(grad_h, dh_dparam)
    
    return sm, grad_params

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    rng = np.random.default_rng(42)
    
    # Start with uniform
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Spectral parameter bounds
    # sigma: [0.01, 0.5] - width of main Gaussian
    # bump_center: [0.0, 1.0] - center of bump in normalized freq
    # bump_width: [0.01, 0.3] - width of bump
    # bump_amp: [-0.5, 2.0] - amplitude of bump (can be negative to create dip)
    # offset: [-0.5, 0.5] - DC offset
    bounds = [
        (0.01, 0.5),    # sigma
        (0.0, 1.0),     # bump_center
        (0.01, 0.3),    # bump_width
        (-0.5, 2.0),    # bump_amp
        (-0.5, 0.5),    # offset
    ]
    
    # Generate diverse initial spectral parameters
    init_params_list = []
    
    # Baseline: broad Gaussian
    init_params_list.append([0.3, 0.5, 0.1, 0.0, 0.0])
    # Narrower Gaussian
    init_params_list.append([0.15, 0.5, 0.1, 0.0, 0.0])
    # Very broad
    init_params_list.append([0.45, 0.5, 0.1, 0.0, 0.0])
    # With bump at edge
    init_params_list.append([0.25, 0.1, 0.15, 0.5, 0.0])
    # With bump at middle
    init_params_list.append([0.25, 0.5, 0.15, 0.5, 0.0])
    # With negative bump (dip)
    init_params_list.append([0.25, 0.5, 0.15, -0.3, 0.0])
    # With offset
    init_params_list.append([0.25, 0.5, 0.15, 0.0, 0.2])
    init_params_list.append([0.25, 0.5, 0.15, 0.0, -0.2])
    
    # Random initializations
    for _ in range(20):
        params = [
            rng.uniform(0.05, 0.45),   # sigma
            rng.uniform(0.0, 1.0),     # bump_center
            rng.uniform(0.02, 0.25),   # bump_width
            rng.uniform(-0.3, 1.5),    # bump_amp
            rng.uniform(-0.3, 0.3),    # offset
        ]
        init_params_list.append(params)
    
    # Evaluate all initial candidates
    for params in init_params_list:
        if time_left() < 10:
            break
        h = _spectral_to_heights(params)
        score = compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Now optimize using spectral parameters
    # Try multiple alpha values and starting points
    for alpha in [50.0, 100.0, 200.0, 400.0]:
        if time_left() < 15:
            break
        
        # Try from best current solution
        # Find spectral params that give best_h (approximately)
        # For now, just try all initial params
        for params_init in init_params_list[:10]:
            if time_left() < 10:
                break
            
            # Add small perturbation
            params_start = np.array(params_init) + rng.normal(0, 0.01, len(params_init))
            # Clip to bounds
            for j, (lo, hi) in enumerate(bounds):
                params_start[j] = np.clip(params_start[j], lo, hi)
            
            try:
                res = minimize(
                    lambda p: _spectral_objective(p, alpha),
                    params_start,
                    jac=lambda p: _spectral_grad(p, alpha)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if res.success or res.fun < best_score:
                    h_res = _spectral_to_heights(res.x)
                    score = compute_score(h_res)
                    if score < best_score:
                        best_score = score
                        best_h = h_res.copy()
            except Exception:
                pass
    
    # Final refinement with highest alpha
    if time_left() > 15:
        alpha_final = 800.0
        # Try from a few good starting points
        for params_init in init_params_list[:5]:
            if time_left() < 10:
                break
            params_start = np.array(params_init)
            for j, (lo, hi) in enumerate(bounds):
                params_start[j] = np.clip(params_start[j], lo, hi)
            
            try:
                res = minimize(
                    lambda p: _spectral_objective(p, alpha_final),
                    params_start,
                    jac=lambda p: _spectral_grad(p, alpha_final)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                
                if res.success or res.fun < best_score:
                    h_res = _spectral_to_heights(res.x)
                    score = compute_score(h_res)
                    if score < best_score:
                        best_score = score
                        best_h = h_res.copy()
            except Exception:
                pass
    
    # Ensure non-negativity and normalize
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) > 0:
        best_h /= np.sum(best_h)
    
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2 using linear autoconv."""
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

def _circular_autoconv_from_H(H):
    """Compute circular autoconvolution from full DFT H."""
    P = np.abs(H) ** 2
    c = np.fft.ifft(P).real
    return c

def _spectral_objective(x, beta=100.0):
    """
    Smooth spectral objective. x parameterizes H[0].real and H[k].real, H[k].imag for k=1..N/2.
    Uses soft-max approximation of max|c[k]| / H[0]^2 where c = ifft(|H|^2).
    """
    n_half = N // 2
    H = np.zeros(N, dtype=complex)
    H[0] = x[0]
    for k in range(1, n_half + 1):
        H[k] = x[2*k - 1] + 1j * x[2*k]
        H[N - k] = np.conj(H[k])
    
    H0 = H[0]
    if abs(H0) < 1e-12:
        return 1e10
    
    P = np.abs(H) ** 2
    c = np.fft.ifft(P).real
    abs_c = np.abs(c)
    
    # Soft max: log-sum-exp approximation
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    log_sum_exp = m + np.log(np.sum(np.exp((abs_c - m) * beta)) / N)
    return log_sum_exp / (H0 * H0)

def _spectral_objective_exact(x):
    """Exact max-based objective for final evaluation."""
    n_half = N // 2
    H = np.zeros(N, dtype=complex)
    H[0] = x[0]
    for k in range(1, n_half + 1):
        H[k] = x[2*k - 1] + 1j * x[2*k]
        H[N - k] = np.conj(H[k])
    
    H0 = H[0]
    if abs(H0) < 1e-12:
        return 1e10
    
    P = np.abs(H) ** 2
    c = np.fft.ifft(P).real
    return np.max(np.abs(c)) / (H0 * H0)

def _spectral_optimize():
    """
    Optimize in frequency domain using spectral decomposition approach.
    The circular autoconv is c = ifft(|H|^2). We minimize max|c[k]|/H[0]^2.
    
    Key: parameterize H directly and use smooth objective for optimization,
    then refine with exact objective.
    """
    t_start = time.time()
    deadline = t_start + 85.0
    
    n_half = N // 2
    n_vars = 1 + 2 * n_half
    
    best_x = None
    best_score = np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Constant signal: H[0]=1, H[k]=0 for k>0
    x0 = np.zeros(n_vars)
    x0[0] = 1.0
    initializations.append(x0.copy())
    
    # 2. Low-pass: H[k] = exp(-k^2 / (2*sigma^2))
    for sigma in [2.0, 5.0, 10.0, 20.0]:
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        for k in range(1, n_half + 1):
            x_init[2*k - 1] = np.exp(-k**2 / (2.0 * sigma**2))
            x_init[2*k] = 0.0
        initializations.append(x_init)
    
    # 3. Cosine window spectrum
    for width in [0.3, 0.5, 0.7, 1.0]:
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        for k in range(1, n_half + 1):
            if k / n_half < width:
                x_init[2*k - 1] = np.cos(np.pi * k / (2.0 * n_half * width))
            else:
                x_init[2*k - 1] = 0.0
            x_init[2*k] = 0.0
        initializations.append(x_init)
    
    # 4. Alternating signs in spectrum to create cancellations in autoconv
    rng = np.random.default_rng(123)
    for _ in range(5):
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        for k in range(1, n_half + 1):
            amp = rng.uniform(0.01, 0.1)
            phase = rng.uniform(0, 2*np.pi)
            x_init[2*k - 1] = amp * np.cos(phase)
            x_init[2*k] = amp * np.sin(phase)
        initializations.append(x_init)
    
    # 5. Designed to minimize circular autoconv: try to make c[k] ~ 0 for k != 0
    # This requires |H|^2 to be concentrated at k=0, but with phase control
    for amp_scale in [0.05, 0.1, 0.2]:
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        for k in range(1, n_half + 1):
            # Alternate real/imag to spread energy in c
            if k % 2 == 0:
                x_init[2*k - 1] = amp_scale / k
                x_init[2*k] = 0.0
            else:
                x_init[2*k - 1] = 0.0
                x_init[2*k] = amp_scale / k
        initializations.append(x_init)
    
    # Optimize each initialization
    for init_idx, x_init in enumerate(initializations):
        if time.time() > deadline - 8:
            break
        
        # Use smooth objective for optimization
        try:
            result = minimize(
                _spectral_objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            score_exact = _spectral_objective_exact(result.x)
            if score_exact < best_score:
                best_score = score_exact
                best_x = result.x.copy()
        except Exception:
            continue
    
    if best_x is None:
        return None, np.inf
    
    # Refine with exact objective using small steps
    if time.time() < deadline - 5:
        x_ref = best_x.copy()
        best_ref = best_score
        
        for iteration in range(20):
            if time.time() > deadline - 3:
                break
            improved = False
            # Try coordinate descent on a subset of variables
            step_size = 0.01 / (1 + iteration * 0.5)
            for idx in range(1, n_vars, max(1, n_vars // 100)):
                if time.time() > deadline - 3:
                    break
                for delta in [step_size, -step_size]:
                    x_ref[idx] += delta
                    new_val = _spectral_objective_exact(x_ref)
                    if new_val < best_ref:
                        best_ref = new_val
                        improved = True
                    else:
                        x_ref[idx] -= delta
            if not improved:
                break
        
        if best_ref < best_score:
            best_score = best_ref
            best_x = x_ref.copy()
    
    # Convert to time domain
    n_half = N // 2
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
    deadline = t_start + 20.0
    
    h_refined = h.copy()
    best_local = _score(h_refined)
    
    # Sign flips
    for _ in range(4):
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
        for _ in range(80):
            if time.time() > deadline - 3:
                break
            idx = rng.integers(0, N)
            delta = rng.normal(0, 0.005)
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

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 120s limit
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline: constant signal
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Strategy 1: Spectral-domain optimization
    if time.time() < deadline - 30:
        h_spec, score_spec = _spectral_optimize()
        if h_spec is not None and score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Strategy 2: Time-domain local refinement
    if time.time() < deadline - 5:
        h_ref, score_ref = _time_domain_refine(best_h)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
    
    return best_h

# EVOLVE-BLOCK-END

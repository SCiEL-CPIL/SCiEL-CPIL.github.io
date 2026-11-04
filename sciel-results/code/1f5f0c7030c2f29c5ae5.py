import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _smooth_spectral_objective(x, alpha=8.0):
    """
    Smooth surrogate: penalize off-diagonal autoconvolution peaks via soft-max,
    normalized by S^2. Encourages flat spectrum away from DC.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_lin(x)
    off = np.abs(c[1:]) if len(c) > 1 else np.array([0.0])
    m = np.max(off)
    if m < 1e-14:
        return 0.0
    soft = m + np.log(np.sum(np.exp(alpha * (off - m)))) / alpha
    return soft / (s * s)

def _spectral_flatness_objective(x, alpha=8.0):
    """
    Direct spectral objective: minimize the sum of |F(omega)|^2 for omega != 0,
    normalized by S^2. This is exactly the sum of squared off-diagonal c_k / S^2,
    which upper-bounds max|c_k|/S^2.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    # |F(0)|^2 = S^2 (DC). We want to minimize the rest.
    off_energy = np.sum(np.abs(fx[1:]) ** 2)
    return off_energy / (s * s)

def _optimize_continuous(x0, deadline, use_spectral=True):
    """
    Optimize the continuous relaxation using L-BFGS-B with a smooth objective.
    """
    obj = _spectral_flatness_objective if use_spectral else _smooth_spectral_objective
    try:
        res = scipy.optimize.minimize(
            obj, x0, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        return res.x
    except Exception:
        return x0

def _spectral_projection(x, n_iter=20):
    """
    Iteratively project toward flat off-DC spectrum:
    1. Compute FFT
    2. Set magnitude to constant for non-DC bins (keeping phase)
    3. IFFT, take real part
    This is like iterative spectral flattening.
    """
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    for _ in range(n_iter):
        fx = np.fft.rfft(x, size)
        # Keep DC, flatten others to a common magnitude
        dc = fx[0]
        if abs(dc) < 1e-10:
            break
        # Target magnitude for non-DC: keep total energy reasonable
        # Use the average magnitude of non-DC bins as target
        non_dc_mag = np.abs(fx[1:])
        if len(non_dc_mag) > 0:
            target_mag = np.mean(non_dc_mag)
            fx_new = fx.copy()
            fx_new[0] = dc
            for i in range(1, len(fx)):
                phase = np.angle(fx[i])
                fx_new[i] = target_mag * np.exp(1j * phase)
            x = np.real(np.fft.irfft(fx_new, size))[:n]
    return x

def _refine_with_local_search(x, deadline):
    """
    Local refinement: try small perturbations to reduce the actual max|c_k|/S^2.
    """
    best_x = x.copy()
    best_val = _metric_std(best_x)
    n = len(x)
    s = np.sum(best_x)
    
    # Try perturbing individual elements
    for trial in range(200):
        if time.time() > deadline:
            break
        # Random perturbation
        idx = np.random.randint(0, n)
        delta = np.random.uniform(-0.5, 0.5)
        x_trial = best_x.copy()
        x_trial[idx] += delta
        # Preserve sum
        s_trial = np.sum(x_trial)
        if abs(s_trial) < 1e-10:
            continue
        x_trial = x_trial * (s / s_trial)
        val = _metric_std(x_trial)
        if val < best_val - 1e-12:
            best_val = val
            best_x = x_trial
    
    return best_x

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Strategy 1: Continuous optimization from multiple starting points
    starts = [
        np.ones(N),
        rng.standard_normal(N),
        np.sin(np.linspace(0, 2 * np.pi, N)),
        rng.choice([-1.0, 1.0], size=N).astype(float),
    ]
    
    for start in starts:
        if time.time() > deadline - 30:
            break
        s = np.sum(start)
        if abs(s) < 1e-10:
            start = start + 1.0
            s = np.sum(start)
        start = start / s  # normalize sum to 1
        
        # Continuous optimization
        x_opt = _optimize_continuous(start, deadline, use_spectral=True)
        # Preserve sum
        s_opt = np.sum(x_opt)
        if abs(s_opt) < 1e-10:
            continue
        x_opt = x_opt / s_opt
        val = _metric_std(x_opt)
        if val < best_val:
            best_val = val
            best_x = x_opt
    
    # Strategy 2: Spectral projection from best
    if time.time() < deadline - 15:
        x_spec = _spectral_projection(best_x, n_iter=30)
        s_spec = np.sum(x_spec)
        if abs(s_spec) > 1e-10:
            x_spec = x_spec / s_spec
            val_spec = _metric_std(x_spec)
            if val_spec < best_val:
                best_val = val_spec
                best_x = x_spec
    
    # Strategy 3: Local refinement
    if time.time() < deadline - 5:
        x_ref = _refine_with_local_search(best_x, deadline)
        val_ref = _metric_std(x_ref)
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref
    
    return best_x

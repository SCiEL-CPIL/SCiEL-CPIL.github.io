"""Structural change: direct optimization of h with smooth max surrogate for the autoconvolution peak.
Uses a log-sum-exp smoothed version of the supremum norm to make the objective differentiable,
then applies L-BFGS-B on the log-heights. Also tries a DFT-based parameterization as a second family.
The key insight is that the ratio (L2^2)/(L1 * sup|autoconv|) is scale-invariant, so we can
fix one normalization and optimize the shape."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = time.time() + 140.0


def _time_left():
    return DEADLINE - time.time()


def _autoconv_peak_smooth(h, beta=50.0):
    """Smooth approximation of sup|autoconv| using log-sum-exp."""
    ac = np.convolve(h, h, mode='full')
    ac_abs = np.abs(ac)
    # Log-sum-exp trick for numerical stability
    m = np.max(ac_abs)
    if m <= 0:
        return 0.0
    # (1/beta) * log(sum(exp(beta * x))) approximates max(x) as beta -> inf
    beta_ac = beta * ac_abs / m
    max_beta = np.max(beta_ac)
    lse = m + (1.0 / beta) * (max_beta + np.log(np.sum(np.exp(beta_ac - max_beta))))
    return lse


def _ratio_from_h(h):
    """Compute raw objective: (L2^2) / (L1 * sup(|autoconv|))."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 0.0
    ac = np.convolve(h, h, mode='full')
    s = float(np.max(np.abs(ac)))
    if s <= 0.0:
        return 0.0
    return l2sq / (l1 * s)


def _smooth_ratio_from_h(h, beta=50.0):
    """Smoothed version of the ratio for gradient-based optimization."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 0.0
    ac = np.convolve(h, h, mode='full')
    ac_abs = np.abs(ac)
    m = np.max(ac_abs)
    if m <= 0:
        return 0.0
    beta_ac = beta * ac_abs / m
    max_beta = np.max(beta_ac)
    lse = m + (1.0 / beta) * (max_beta + np.log(np.sum(np.exp(beta_ac - max_beta))))
    return l2sq / (l1 * lse)


def _objective_log_heights(x, beta=50.0):
    """
    Objective for optimization over log-heights.
    x is a vector of length N representing log(h_k).
    h_k = exp(x_k) is automatically nonnegative.
    We normalize by fixing sum(h) = 1 (or equivalently, the ratio is scale-invariant).
    """
    h = np.exp(x)
    # Normalize to avoid overflow issues; ratio is scale-invariant
    h = h / np.sum(h)
    return -_smooth_ratio_from_h(h, beta)


def _gradient_log_heights(x, beta=50.0):
    """Finite-difference gradient for the log-heights objective."""
    h0 = np.exp(x)
    h0 = h0 / np.sum(h0)
    f0 = _smooth_ratio_from_h(h0, beta)
    
    grad = np.zeros(N)
    eps = 1e-7
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        h_plus = np.exp(x_plus)
        h_plus = h_plus / np.sum(h_plus)
        f_plus = _smooth_ratio_from_h(h_plus, beta)
        grad[i] = -(f_plus - f0) / eps
    return grad


def _objective_log_heights_and_grad(x, beta=50.0):
    """Returns (-smooth_ratio, gradient) for L-BFGS-B."""
    h = np.exp(x)
    h = h / np.sum(h)
    f = _smooth_ratio_from_h(h, beta)
    grad = _gradient_log_heights(x, beta)
    return -f, grad


def _dft_parameterization(coeffs, max_freq=5):
    """
    Parameterize h_k = |sum_j c_j * omega^(j*k)|^2 where omega = exp(2*pi*i/N).
    coeffs is a real vector of length 2*(max_freq+1).
    This ensures h_k >= 0 and has spectral structure.
    """
    n_coeffs = max_freq + 1
    c = np.zeros(n_coeffs, dtype=complex)
    for j in range(n_coeffs):
        c[j] = coeffs[2*j] + 1j * coeffs[2*j+1]
    
    omega = np.exp(2j * np.pi * np.arange(N) / N)
    # Evaluate polynomial at all N roots of unity
    vals = np.zeros(N, dtype=complex)
    for k in range(N):
        w = omega[k]
        val = 0.0 + 0.0j
        for j in range(n_coeffs - 1, -1, -1):
            val = val * w + c[j]
        vals[k] = val
    h = np.abs(vals) ** 2
    return h


def _ratio_from_dft(coeffs, max_freq=5):
    """Compute ratio from DFT parameterization."""
    h = _dft_parameterization(coeffs, max_freq)
    if np.sum(h) <= 0:
        return 0.0
    return _ratio_from_h(h)


def _smooth_ratio_from_dft(coeffs, max_freq=5, beta=50.0):
    """Smoothed ratio from DFT parameterization."""
    h = _dft_parameterization(coeffs, max_freq)
    if np.sum(h) <= 0:
        return 0.0
    return _smooth_ratio_from_h(h, beta)


def _objective_dft_and_grad(coeffs, max_freq=5, beta=50.0):
    """Returns (-smooth_ratio, gradient) for DFT parameterization."""
    f0 = _smooth_ratio_from_dft(coeffs, max_freq, beta)
    dim = 2 * (max_freq + 1)
    grad = np.zeros(dim)
    eps = 1e-6
    for i in range(dim):
        c_plus = coeffs.copy()
        c_plus[i] += eps
        f_plus = _smooth_ratio_from_dft(c_plus, max_freq, beta)
        grad[i] = -(f_plus - f0) / eps
    return -f0, grad


def _make_initial_log_heights(rng, style='gaussian'):
    """Create initial log-height vectors for multi-start."""
    if style == 'gaussian':
        center = rng.uniform(0, N)
        width = rng.uniform(3, 15)
        x = np.arange(N, dtype=float)
        h = np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
        h = h / np.sum(h)
        return np.log(h + 1e-15)
    elif style == 'bump':
        # A bump function
        center = rng.uniform(5, N - 5)
        width = rng.uniform(4, 12)
        x = np.arange(N, dtype=float)
        t = (x - center) / width
        h = np.where(np.abs(t) < 1.0, (1 - t**2)**2, 0.0)
        h = np.maximum(h, 1e-10)
        h = h / np.sum(h)
        return np.log(h)
    elif style == 'random_smooth':
        # Random smooth function via low-frequency Fourier
        n_freq = rng.randint(2, 6)
        freqs = rng.choice(np.arange(1, N//2 + 1), size=n_freq, replace=False)
        amps = rng.randn(n_freq) * 0.5
        phases = rng.uniform(0, 2*np.pi, size=n_freq)
        x = np.arange(N, dtype=float)
        h = np.ones(N)
        for f, a, p in zip(freqs, amps, phases):
            h += a * np.cos(2*np.pi*f*x/N + p)
        h = np.maximum(h, 1e-10)
        h = h / np.sum(h)
        return np.log(h)
    else:
        # Uniform
        return np.zeros(N)


def _local_optimize_log_heights(x0, max_iter=1000, beta=50.0):
    """Run L-BFGS-B on log-heights."""
    try:
        res = minimize(
            _objective_log_heights_and_grad,
            x0,
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        h = np.exp(res.x)
        h = h / np.sum(h)
        val = _ratio_from_h(h)
        return res.x, val, h
    except Exception:
        h = np.exp(x0)
        h = h / np.sum(h)
        val = _ratio_from_h(h)
        return x0, val, h


def _local_optimize_dft(coeffs0, max_freq=5, max_iter=800, beta=50.0):
    """Run L-BFGS-B on DFT coefficients."""
    try:
        res = minimize(
            _objective_dft_and_grad,
            coeffs0,
            args=(max_freq, beta),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        h = _dft_parameterization(res.x, max_freq)
        h = h / np.sum(h)
        val = _ratio_from_h(h)
        return res.x, val, h
    except Exception:
        h = _dft_parameterization(coeffs0, max_freq)
        h = h / np.sum(h)
        val = _ratio_from_h(h)
        return coeffs0, val, h


def construct():
    best_val = 0.0
    best_h = None
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Multi-start with log-heights parameterization
    styles = ['gaussian', 'bump', 'random_smooth']
    n_starts = 12
    
    for i in range(n_starts):
        if _time_left() < 15.0:
            break
        
        style = styles[i % len(styles)]
        x0 = _make_initial_log_heights(rng, style)
        
        # Evaluate initial
        h0 = np.exp(x0)
        h0 = h0 / np.sum(h0)
        val0 = _ratio_from_h(h0)
        if val0 > best_val:
            best_val = val0
            best_h = h0.copy()
        
        # Local optimization
        if _time_left() < 20.0:
            break
        
        x_opt, val_opt, h_opt = _local_optimize_log_heights(x0, max_iter=800, beta=50.0)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 2: DFT parameterization
    max_freq = 5
    dim_dft = 2 * (max_freq + 1)
    
    for trial in range(6):
        if _time_left() < 15.0:
            break
        
        # Initialize DFT coefficients
        c0 = np.zeros(dim_dft)
        c0[0] = 1.0  # DC term
        # Add small random perturbations
        c0 += rng.randn(dim_dft) * 0.1
        c0[0] = 1.0  # Keep DC dominant
        
        val0 = _ratio_from_dft(c0, max_freq)
        if val0 > best_val:
            best_val = val0
            h0 = _dft_parameterization(c0, max_freq)
            h0 = h0 / np.sum(h0)
            best_h = h0.copy()
        
        if _time_left() < 20.0:
            break
        
        c_opt, val_opt, h_opt = _local_optimize_dft(c0, max_freq=max_freq, max_iter=600, beta=50.0)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 3: Polish the best solution with higher beta (closer to true max)
    if best_h is not None and _time_left() > 25.0:
        # Re-optimize from best_h with higher beta for sharper peak approximation
        x_best = np.log(np.maximum(best_h, 1e-15))
        # Try with increasing beta
        for beta in [100.0, 200.0]:
            if _time_left() < 15.0:
                break
            x_polish, val_polish, h_polish = _local_optimize_log_heights(
                x_best, max_iter=1500, beta=beta
            )
            if val_polish > best_val:
                best_val = val_polish
                best_h = h_polish.copy()
                x_best = x_polish
    
    # Phase 4: Try a few more diverse starts if time permits
    if _time_left() > 30.0:
        for i in range(8):
            if _time_left() < 10.0:
                break
            x0 = _make_initial_log_heights(rng, 'random_smooth')
            x_opt, val_opt, h_opt = _local_optimize_log_heights(x0, max_iter=500, beta=50.0)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    
    # Final: ensure nonnegative and return
    if best_h is None:
        best_h = np.ones(N) / N
    
    best_h = np.maximum(best_h, 0.0)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    
    return [float(v) for v in best_h]


if __name__ == '__main__':
    r = construct()
    print(len(r), _ratio_from_h(r))

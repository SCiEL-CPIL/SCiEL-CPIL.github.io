import time
import numpy as np
from scipy.optimize import minimize

N = 50
w = 0.5 / N  # step width = 0.01
_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _time_left():
    return DEADLINE - _elapsed()


def _autoconv_max(h):
    """Compute max of autoconvolution of h (with w factor)."""
    h = np.maximum(h, 0.0)
    c0 = np.dot(h, h)
    cmax = c0
    for m in range(1, N):
        cm = np.dot(h[:N - m], h[m:])
        if cm > cmax:
            cmax = cm
    return w * cmax


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    acmax = _autoconv_max(h)
    if acmax <= 0:
        return 0.0
    return s2 / (s1 * acmax)


def _ratio_neg(h):
    """Negative of ratio for minimization."""
    return -_ratio(h)


def _dft_parameterize(z):
    """Convert DFT coefficients z (complex, length N) to time-domain h (nonnegative)."""
    h = np.fft.ifft(z).real
    h = np.maximum(h, 0.0)
    return h


def _dft_inverse(h):
    """Convert time-domain h to DFT coefficients."""
    return np.fft.fft(h)


def _optimize_from_z(z0, maxiter=200, tol=1e-10):
    """Optimize from initial DFT coefficients using gradient-based method."""
    def _ratio_and_grad(z_complex):
        # Convert to real/imag for scipy
        z_real = np.concatenate([z_complex.real, z_complex.imag])
        
        def _obj(x):
            z_c = np.array(x[:N] + 1j * x[N:])
            h = _dft_parameterize(z_c)
            return -_ratio(h)
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(2 * N)
        f0 = _obj(z_real)
        for i in range(2 * N):
            z_real_p = z_real.copy()
            z_real_p[i] += eps
            f_p = _obj(z_real_p)
            grad[i] = (f_p - f0) / eps
        
        return f0, grad
    
    # Convert z0 to real/imag representation
    z0_real = np.concatenate([z0.real, z0.imag])
    
    try:
        result = minimize(_ratio_and_grad, z0_real, method='BFGS',
                         options={'maxiter': maxiter, 'gtol': tol, 'disp': False})
        z_opt = np.array(result.x[:N] + 1j * result.x[N:])
        h_opt = _dft_parameterize(z_opt)
        return h_opt, -result.fun
    except Exception:
        return _dft_parameterize(z0), _ratio(_dft_parameterize(z0))


def _local_optimize(h0, maxiter=100):
    """Local optimization using coordinate descent on heights."""
    h = np.array(h0, dtype=float)
    best_f = _ratio(h)
    
    for rnd in range(maxiter):
        if _time_left() < 10:
            break
        improved = False
        for i in range(N):
            if _time_left() < 10:
                break
            orig = h[i]
            # Try several perturbations
            for factor in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 0.0]:
                h[i] = orig * factor if orig > 0 else factor
                f = _ratio(h)
                if f > best_f:
                    best_f = f
                    improved = True
                else:
                    h[i] = orig
        if not improved:
            break
    
    return h, best_f


def _gaussian_profile(sigma, center=25.0):
    """Create a Gaussian profile."""
    x = np.arange(N)
    h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return h


def _lorentzian_profile(gamma, center=25.0):
    """Create a Lorentzian profile."""
    x = np.arange(N)
    h = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return h


def _power_profile(power, center=25.0):
    """Create a power-law profile."""
    x = np.arange(N)
    max_d = max(center, N - 1 - center)
    h = np.maximum(1.0 - np.abs(x - center) / max_d, 0.0) ** power
    return h


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_f = _ratio(best_h)
    
    # Phase 1: Try various parametric profiles
    if _time_left() > 60:
        # Gaussian profiles
        for sigma in [3, 5, 7, 10, 15, 20, 25]:
            for center in [10, 15, 20, 25, 30, 35, 40]:
                if _time_left() < 50:
                    break
                h = _gaussian_profile(sigma, center)
                f = _ratio(h)
                if f > best_f:
                    best_f = f
                    best_h = h.copy()
        
        # Lorentzian profiles
        for gamma in [2, 3, 5, 7, 10, 15]:
            for center in [10, 15, 20, 25, 30, 35, 40]:
                if _time_left() < 40:
                    break
                h = _lorentzian_profile(gamma, center)
                f = _ratio(h)
                if f > best_f:
                    best_f = f
                    best_h = h.copy()
        
        # Power profiles
        for power in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for center in [10, 15, 20, 25, 30, 35, 40]:
                if _time_left() < 30:
                    break
                h = _power_profile(power, center)
                f = _ratio(h)
                if f > best_f:
                    best_f = f
                    best_h = h.copy()
    
    # Phase 2: DFT-based optimization from multiple initial points
    if _time_left() > 40:
        rng = np.random.default_rng(42)
        
        # Start from best_h
        z0 = _dft_inverse(best_h)
        h_opt, f_opt = _optimize_from_z(z0, maxiter=100)
        if f_opt > best_f:
            best_f = f_opt
            best_h = h_opt.copy()
        
        # Try random DFT initializations
        for _ in range(20):
            if _time_left() < 20:
                break
            z_rand = np.array(rng.standard_normal(N) + 1j * rng.standard_normal(N))
            # Ensure reasonable magnitude
            z_rand *= 0.1
            h_opt, f_opt = _optimize_from_z(z_rand, maxiter=50)
            if f_opt > best_f:
                best_f = f_opt
                best_h = h_opt.copy()
    
    # Phase 3: Local coordinate descent refinement
    if _time_left() > 20:
        best_h, best_f = _local_optimize(best_h, maxiter=50)
    
    # Phase 4: More DFT optimization from refined point
    if _time_left() > 15:
        z0 = _dft_inverse(best_h)
        h_opt, f_opt = _optimize_from_z(z0, maxiter=200)
        if f_opt > best_f:
            best_f = f_opt
            best_h = h_opt.copy()
    
    # Phase 5: Final local refinement
    if _time_left() > 10:
        best_h, best_f = _local_optimize(best_h, maxiter=100)
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    return [float(v) for v in best_h]

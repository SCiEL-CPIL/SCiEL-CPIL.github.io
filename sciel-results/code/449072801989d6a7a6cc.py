import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_ratio(f):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf)"""
    f = np.asarray(f, dtype=np.float64)
    l2_sq = np.sum(f**2)
    l1 = np.sum(f)
    if l1 <= 0:
        return 0.0
    # Linear autoconvolution
    ff = np.convolve(f, f, mode='full')
    sup = np.max(ff)
    if sup <= 0:
        return 0.0
    return l2_sq / (l1 * sup)

def _make_f_from_z(z):
    """Given complex z, return f = IDFT(|z|^2), ensuring non-negativity."""
    power = np.abs(z)**2
    f = np.fft.ifft(power).real
    # Clip to non-negative
    f = np.maximum(f, 0.0)
    return f

def _objective(params):
    """Negative ratio for minimization. params = [re(z), im(z)] flattened."""
    n = len(params) // 2
    z = params[:n] + 1j * params[n:]
    f = _make_f_from_z(z)
    return -_compute_ratio(f)

def _try_optimize(z0, max_iter=500, timeout_remaining=None):
    """Run optimization from initial z0."""
    n = len(z0)
    x0 = np.concatenate([z0.real, z0.imag])
    
    def obj(x):
        if timeout_remaining is not None and time.time() > timeout_remaining:
            return 1e10
        z = x[:n] + 1j * x[n:]
        f = _make_f_from_z(z)
        return -_compute_ratio(f)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.fun < 0:
            z_opt = res.x[:n] + 1j * res.x[n:]
            f_opt = _make_f_from_z(z_opt)
            return f_opt
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_f = [1.0] * N
    best_score = _compute_ratio(np.array(best_f))
    
    rng = np.random.default_rng(42)
    
    # Diverse initializations
    def init_gaussian():
        t = np.linspace(-0.25, 0.25, N)
        sigma = rng.uniform(0.05, 0.2)
        center = rng.uniform(-0.15, 0.15)
        f = np.exp(-0.5 * ((t - center)/sigma)**2)
        return f
    
    def init_lorentzian():
        t = np.linspace(-0.25, 0.25, N)
        gamma = rng.uniform(0.02, 0.15)
        center = rng.uniform(-0.15, 0.15)
        f = gamma / (gamma**2 + (t - center)**2)
        return f
    
    def init_power_law():
        t = np.linspace(-0.25, 0.25, N)
        alpha = rng.uniform(0.5, 3.0)
        f = np.abs(t + 0.01)**(-alpha)
        f /= np.max(f)
        return f
    
    def init_two_peak():
        t = np.linspace(-0.25, 0.25, N)
        c1 = rng.uniform(-0.2, -0.05)
        c2 = rng.uniform(0.05, 0.2)
        s1 = rng.uniform(0.03, 0.1)
        s2 = rng.uniform(0.03, 0.1)
        f = np.exp(-0.5*((t-c1)/s1)**2) + np.exp(-0.5*((t-c2)/s2)**2)
        return f
    
    def init_uniform():
        return np.ones(N)
    
    def init_random_smooth():
        # Random smooth function via low-freq Fourier
        k = np.arange(N)
        coeffs = rng.normal(0, 1, N) + 1j * rng.normal(0, 1, N)
        # Keep only low frequencies
        mask = np.zeros(N, dtype=bool)
        n_low = rng.integers(2, 8)
        mask[:n_low] = True
        mask[-n_low:] = True
        mask[n_low//2:n_low//2+2] = True  # DC and neighbors
        coeffs[~mask] = 0
        f = np.real(np.fft.ifft(coeffs))
        f = np.maximum(f, 0)
        if np.max(f) > 0:
            f /= np.max(f)
        return f
    
    initializers = [init_gaussian, init_lorentzian, init_power_law, init_two_peak, init_uniform, init_random_smooth]
    
    # For each initialization, convert to z-space and optimize
    n_restarts = 30
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        init_fn = initializers[i % len(initializers)]
        f_init = init_fn()
        
        # Convert f to z-space: we want f = IDFT(|z|^2)
        # So |z|^2 = DFT(f), thus z = sqrt(DFT(f)) * phase
        F = np.fft.fft(f_init)
        F = np.maximum(F.real, 1e-10)  # Ensure non-negative for sqrt
        # z such that |z|^2 = F => z = sqrt(F) * exp(i*theta)
        z0 = np.sqrt(F + 1e-10) * np.exp(1j * rng.uniform(0, 2*np.pi, N))
        
        # Try optimization
        timeout_rem = deadline - 5.0
        f_opt = _try_optimize(z0, max_iter=200, timeout_remaining=timeout_rem)
        
        if f_opt is not None:
            score = _compute_ratio(f_opt)
            if score > best_score:
                best_score = score
                best_f = list(f_opt)
        
        # Also try the initial f directly
        score_init = _compute_ratio(f_init)
        if score_init > best_score:
            best_score = score_init
            best_f = list(f_init)
    
    # Additional local refinement on best solution
    if time.time() < deadline - 10:
        f_best = np.array(best_f)
        F = np.fft.fft(f_best)
        F = np.maximum(F.real, 1e-10)
        z_best = np.sqrt(F + 1e-10) * np.exp(1j * rng.uniform(0, 2*np.pi, N))
        f_refined = _try_optimize(z_best, max_iter=500, timeout_remaining=deadline - 5)
        if f_refined is not None:
            score = _compute_ratio(f_refined)
            if score > best_score:
                best_score = score
                best_f = list(f_refined)
    
    return best_f

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.signal import fftconvolve

N = 400
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _autoconv(x):
    """Compute autoconvolution of x with itself."""
    return fftconvolve(x, x, mode='full')

def _metric(x):
    """Compute the metric: max(|autoconv|) / (sum(x)^2)."""
    ac = _autoconv(x)
    integral = np.sum(x)
    if integral == 0:
        return np.inf
    return np.max(np.abs(ac)) / (integral * integral)

def _random_sign_vector(n, seed=0):
    """Generate a random sign vector."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n)

def _local_search(x, max_iter=5000, time_budget=10.0):
    """Local search: flip one element at a time if it improves the metric."""
    start_time = time.time()
    best_metric = _metric(x)
    best_x = x.copy()
    n = len(x)
    
    # Precompute autoconvolution
    ac = _autoconv(best_x)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Try flipping a random element
        i = np.random.randint(n)
        old_val = best_x[i]
        best_x[i] = -old_val
        
        # Update autoconvolution efficiently? For now, recompute
        new_ac = _autoconv(best_x)
        new_metric = np.max(np.abs(new_ac)) / (np.sum(best_x) * np.sum(best_x))
        
        if new_metric < best_metric:
            best_metric = new_metric
            ac = new_ac
        else:
            best_x[i] = old_val
    
    return best_x, best_metric

def _spectral_phase_optimization(n, time_budget=30.0):
    """
    Represent x as impulse response of an all-pass filter with random phase.
    Optimize phases via gradient descent on the metric.
    
    The key insight: all-pass filters have flat magnitude spectrum |X(f)| = 1 for all f,
    which means the autoconvolution energy is spread uniformly. We then round to signs.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Number of all-pass stages
    n_stages = min(10, n // 4)
    
    # Random initial phases
    phases = rng.uniform(-np.pi, np.pi, size=n_stages)
    
    # Learning rate
    lr = 0.1
    best_metric = np.inf
    best_x = None
    
    for step in range(500):
        if time.time() - start_time > time_budget:
            break
        
        # Compute impulse response of all-pass filter
        # All-pass filter: H(z) = prod_k (z^{-1} + a_k) / (1 + a_k * z^{-1})
        # where a_k = e^{i*phase_k}
        # Impulse response can be computed via inverse FFT of H(f)
        
        # Frequency grid
        freqs = np.fft.fftfreq(n) * 2 * np.pi
        
        # Compute H(f) for each all-pass stage
        # H_k(f) = (e^{-if} + e^{i*phase_k}) / (1 + e^{i*phase_k} * e^{-if})
        # = (e^{-if} + e^{i*phase_k}) / (1 + e^{i*(phase_k - f)})
        
        H = np.ones(n, dtype=complex)
        for k in range(n_stages):
            phase_k = phases[k]
            # H_k(f) = (e^{-if} + e^{i*phase_k}) / (1 + e^{i*(phase_k - f)})
            num = np.exp(-1j * freqs) + np.exp(1j * phase_k)
            den = 1.0 + np.exp(1j * (phase_k - freqs))
            H *= num / den
        
        # Impulse response
        x = np.real(np.fft.ifft(H))
        
        # Compute metric
        ac = _autoconv(x)
        integral = np.sum(x)
        if abs(integral) < 1e-10:
            metric = np.inf
        else:
            metric = np.max(np.abs(ac)) / (integral * integral)
        
        if metric < best_metric:
            best_metric = metric
            best_x = x.copy()
        
        # Gradient descent on phases
        # We need d(metric)/d(phases)
        # This is complex, so use finite differences or automatic differentiation
        # For simplicity, use a heuristic: perturb phases and keep if improved
        
        improved = False
        for k in range(n_stages):
            old_phase = phases[k]
            delta = lr * (1.0 if rng.random() > 0.5 else -1.0)
            phases[k] = old_phase + delta
            
            # Recompute H and x
            H2 = np.ones(n, dtype=complex)
            for j in range(n_stages):
                phase_j = phases[j]
                num = np.exp(-1j * freqs) + np.exp(1j * phase_j)
                den = 1.0 + np.exp(1j * (phase_j - freqs))
                H2 *= num / den
            
            x2 = np.real(np.fft.ifft(H2))
            ac2 = _autoconv(x2)
            integral2 = np.sum(x2)
            if abs(integral2) < 1e-10:
                metric2 = np.inf
            else:
                metric2 = np.max(np.abs(ac2)) / (integral2 * integral2)
            
            if metric2 < metric:
                improved = True
                metric = metric2
                if metric < best_metric:
                    best_metric = metric
                    best_x = x2.copy()
            else:
                phases[k] = old_phase
        
        if not improved:
            lr *= 0.9
    
    if best_x is None:
        return _random_sign_vector(n, 42), np.inf
    
    # Round to signs
    x_sign = np.sign(best_x)
    x_sign[x_sign == 0] = 1.0
    metric_sign = _metric(x_sign)
    
    return x_sign, metric_sign

def _hill_climbing_from_random(n, num_starts=5, time_per_start=10.0):
    """Multiple random starts with local search."""
    start_time = time.time()
    best_x = None
    best_metric = np.inf
    
    for s in range(num_starts):
        if time.time() - start_time > time_per_start * (s + 1):
            break
        
        x = _random_sign_vector(n, seed=1000 + s)
        x, metric = _local_search(x, max_iter=2000, time_budget=time_per_start)
        
        if metric < best_metric:
            best_metric = metric
            best_x = x
    
    return best_x, best_metric

def construct():
    """Construct a sign vector minimizing the metric."""
    start_time = time.time()
    
    # Strategy 1: Spectral phase optimization
    x_spec, metric_spec = _spectral_phase_optimization(N, time_budget=30.0)
    
    # Strategy 2: Hill climbing from random starts
    x_hill, metric_hill = _hill_climbing_from_random(N, num_starts=3, time_per_start=10.0)
    
    # Pick the best
    if metric_spec < metric_hill:
        return x_spec
    else:
        return x_hill
# EVOLVE-BLOCK-END

"""Program that minimizes the autocorrelation peak via spectral construction and local optimization."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

# EVOLVE-BLOCK-START
def _value(h):
    """Compute the task's objective value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def _fft_convolve(h):
    """Fast full autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    # Next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    return conv


def _smooth_max(conv, beta=100.0):
    """Log-sum-exp approximation of max for smoothness."""
    m = np.max(conv)
    if m < -1e10:
        return m
    return m + np.log(np.sum(np.exp((conv - m) * beta)) / len(conv)) / beta


def _smooth_objective(h, beta=100.0):
    """Smooth surrogate objective using log-sum-exp for max."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_convolve(h)
    smax = _smooth_max(conv, beta)
    return abs(2.0 * N * smax / (s ** 2))


def _smooth_gradient(h, beta=100.0, eps=1e-7):
    """Numerical gradient of smooth objective."""
    grad = np.zeros(N)
    f0 = _smooth_objective(h, beta)
    for i in range(N):
        hp = h.copy()
        hp[i] += eps
        grad[i] = (_smooth_objective(hp, beta) - f0) / eps
    return grad


def _spectral_construct(frequencies, amplitudes, phases, zero_mean=True):
    """Construct h as sum of sines with given params, optionally zero-mean."""
    idx = np.arange(N)
    h = np.zeros(N)
    for f, a, p in zip(frequencies, amplitudes, phases):
        h += a * np.sin(2.0 * np.pi * f * idx / N + p)
    if zero_mean:
        h -= np.mean(h)
    return h


def _pulse_superposition(pulses, zero_mean=True):
    """Construct h as sum of rectangular pulses.
    pulses: list of (center, width, amplitude)
    """
    h = np.zeros(N)
    for center, width, amp in pulses:
        start = int(center - width // 2)
        end = int(center + width // 2 + 1)
        # Wrap around for periodic boundary
        if start < 0 or end > N:
            # Handle wrap-around
            for i in range(start, end):
                h[i % N] += amp
        else:
            h[start:end] += amp
    if zero_mean:
        h -= np.mean(h)
    return h


def _optimize_local(h0, max_iter=2000, tol=1e-10, beta=50.0):
    """Local optimization using L-BFGS-B on the smooth objective."""
    s = np.sum(h0)
    if abs(s) < 1e-15:
        h0 = h0 + 1.0  # shift to avoid zero sum
        s = np.sum(h0)
    
    def objective(x):
        return _smooth_objective(x, beta)
    
    def gradient(x):
        return _smooth_gradient(x, beta)
    
    result = minimize(objective, h0, method='L-BFGS-B', jac=gradient,
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
    return result.x


def _optimize_pulse_params(pulse_params, max_iter=1000, tol=1e-10):
    """Optimize pulse parameters (centers, widths, amplitudes) directly."""
    n_pulses = len(pulse_params)
    
    def pulse_obj(params):
        # params: [c1, w1, a1, c2, w2, a2, ...]
        pulses = []
        for i in range(n_pulses):
            c = params[3*i]
            w = max(1, int(abs(params[3*i+1])))
            a = params[3*i+2]
            pulses.append((c, w, a))
        h = _pulse_superposition(pulses, zero_mean=True)
        if abs(np.sum(h)) < 1e-10:
            return 1e10
        return _smooth_objective(h, beta=30.0)
    
    result = minimize(pulse_obj, pulse_params, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
    return result.x


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    np.random.seed(42)
    
    # Strategy 1: Spectral construction with random parameters
    for trial in range(80):
        if time.time() - start_time > deadline:
            break
        
        # Choose random frequencies (integers for periodicity)
        n_harmonics = np.random.randint(5, 30)
        freqs = np.random.choice(range(1, N // 2), size=n_harmonics, replace=False)
        amps = np.random.uniform(0.1, 2.0, size=n_harmonics)
        phases = np.random.uniform(0, 2 * np.pi, size=n_harmonics)
        
        h = _spectral_construct(freqs, amps, phases, zero_mean=True)
        if abs(np.sum(h)) < 1e-10:
            h += 0.01
        
        val = _value(h)
        if val < best_val:
            best_h = h.copy()
            best_val = val
        
        # Local optimization from this starting point
        if time.time() - start_time > deadline - 40:
            break
        try:
            h_opt = _optimize_local(h, max_iter=800, tol=1e-9, beta=50.0)
            val_opt = _value(h_opt)
            if val_opt < best_val:
                best_h = h_opt.copy()
                best_val = val_opt
        except Exception:
            pass
    
    # Strategy 2: Sparse pulse superposition
    n_pulse_trials = 30
    for trial in range(n_pulse_trials):
        if time.time() - start_time > deadline - 30:
            break
        
        # Random number of pulses: 3 to 8
        n_pulses = np.random.randint(3, 9)
        pulse_params = np.zeros(3 * n_pulses)
        for i in range(n_pulses):
            pulse_params[3*i] = np.random.uniform(0, N)  # center
            pulse_params[3*i+1] = np.random.uniform(5, 40)  # width
            pulse_params[3*i+2] = np.random.uniform(-1, 1)  # amplitude
        
        # First evaluate the initial construction
        pulses = []
        for i in range(n_pulses):
            c = pulse_params[3*i]
            w = max(1, int(abs(pulse_params[3*i+1])))
            a = pulse_params[3*i+2]
            pulses.append((c, w, a))
        h = _pulse_superposition(pulses, zero_mean=True)
        if abs(np.sum(h)) < 1e-10:
            h += 0.01
        val = _value(h)
        if val < best_val:
            best_h = h.copy()
            best_val = val
        
        # Optimize pulse parameters
        if time.time() - start_time > deadline - 25:
            break
        try:
            opt_params = _optimize_pulse_params(pulse_params, max_iter=500, tol=1e-8)
            pulses_opt = []
            for i in range(n_pulses):
                c = opt_params[3*i]
                w = max(1, int(abs(opt_params[3*i+1])))
                a = opt_params[3*i+2]
                pulses_opt.append((c, w, a))
            h_opt = _pulse_superposition(pulses_opt, zero_mean=True)
            if abs(np.sum(h_opt)) < 1e-10:
                h_opt += 0.01
            val_opt = _value(h_opt)
            if val_opt < best_val:
                best_h = h_opt.copy()
                best_val = val_opt
            
            # Further polish with direct h optimization
            if time.time() - start_time < deadline - 15:
                h_polish = _optimize_local(h_opt, max_iter=1000, tol=1e-10, beta=80.0)
                val_polish = _value(h_polish)
                if val_polish < best_val:
                    best_h = h_polish.copy()
                    best_val = val_polish
        except Exception:
            pass
    
    # Strategy 3: Try structured patterns
    for pattern_type in ['block', 'sine_mod', 'chebyshev']:
        if time.time() - start_time > deadline - 20:
            break
        
        if pattern_type == 'block':
            for block_size in [5, 10, 20, 25, 40, 50]:
                if time.time() - start_time > deadline - 15:
                    break
                h = np.zeros(N)
                n_blocks = N // block_size
                for b in range(n_blocks):
                    h[b*block_size:(b+1)*block_size] = 1.0 if b % 2 == 0 else -0.5
                h -= np.mean(h)
                if abs(np.sum(h)) < 1e-10:
                    h += 0.01
                val = _value(h)
                if val < best_val:
                    best_h = h.copy()
                    best_val = val
                if time.time() - start_time > deadline - 12:
                    break
                try:
                    h_opt = _optimize_local(h, max_iter=800, tol=1e-10, beta=60.0)
                    val_opt = _value(h_opt)
                    if val_opt < best_val:
                        best_h = h_opt.copy()
                        best_val = val_opt
                except Exception:
                    pass
        elif pattern_type == 'sine_mod':
            idx = np.arange(N)
            carrier = np.random.uniform(3, 10)
            mod_freq = np.random.uniform(0.5, 3)
            h = np.sin(2 * np.pi * carrier * idx / N) * np.sin(2 * np.pi * mod_freq * idx / N)
            h -= np.mean(h)
            if abs(np.sum(h)) < 1e-10:
                h += 0.01
            val = _value(h)
            if val < best_val:
                best_h = h.copy()
                best_val = val
            if time.time() - start_time > deadline - 12:
                break
            try:
                h_opt = _optimize_local(h, max_iter=1000, tol=1e-10, beta=60.0)
                val_opt = _value(h_opt)
                if val_opt < best_val:
                    best_h = h_opt.copy()
                    best_val = val_opt
            except Exception:
                pass
        else:  # chebyshev
            x = np.linspace(-1, 1, N)
            order = np.random.randint(10, 50)
            h = np.polynomial.chebyshev.chebval(x, np.random.normal(0, 1, order + 1))
            h -= np.mean(h)
            if abs(np.sum(h)) < 1e-10:
                h += 0.01
            val = _value(h)
            if val < best_val:
                best_h = h.copy()
                best_val = val
            if time.time() - start_time > deadline - 12:
                break
            try:
                h_opt = _optimize_local(h, max_iter=1000, tol=1e-10, beta=60.0)
                val_opt = _value(h_opt)
                if val_opt < best_val:
                    best_h = h_opt.copy()
                    best_val = val_opt
            except Exception:
                pass
    
    # Final: high-precision polish from best
    if time.time() - start_time < deadline - 5:
        try:
            h_final = _optimize_local(best_h, max_iter=5000, tol=1e-12, beta=100.0)
            val_final = _value(h_final)
            if val_final < best_val:
                best_h = h_final.copy()
                best_val = val_final
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END

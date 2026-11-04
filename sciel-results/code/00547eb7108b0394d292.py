"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _soft_max_abs(conv, beta=200.0):
    """Smooth approximation of max |conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    if max_val == 0:
        return 0.0
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _smooth_metric(x, beta=200.0):
    """Smooth surrogate: soft_max |autoconv| / (integral)^2."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)

def _spectral_phase_optimize(x_init, deadline, rng):
    """
    Optimize the phase of the FFT while keeping the magnitude fixed.
    This is a different strategy: we fix the power spectrum and vary the phase.
    """
    x = x_init.copy()
    best_x = x.copy()
    best_val = _metric(x)
    
    fx = np.fft.rfft(x, _FFT_SIZE)
    mag = np.abs(fx)
    phase = np.angle(fx)
    
    step_size = 0.1
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Reconstruct x from current phase
        fx_new = mag * np.exp(1j * phase)
        x_new = np.fft.irfft(fx_new, _FFT_SIZE)[:N]
        
        val_new = _metric(x_new)
        if val_new < best_val:
            best_val = val_new
            best_x = x_new.copy()
        
        # Perturb phase
        phase += step_size * rng.normal(0, 1, size=phase.shape)
        # Wrap phase to [-pi, pi]
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
        
        # Adaptive step size
        if iteration % 100 == 0 and step_size > 0.01:
            step_size *= 0.9
    
    return best_x, best_val

def _try_local_search_fast(x, deadline, rng, max_rounds=200000):
    """Fast local search with in-place modifications and early termination."""
    best_x = x.copy()
    best_val = _metric(x)
    no_improve = 0
    step_scale = 0.3
    total_steps = 0
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 200:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.0005:
                break
        
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        # Try multiple perturbations per round
        best_delta = 0
        best_idx = None
        for trial in range(3):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
            total_steps += 1
        else:
            no_improve += 1
    
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0

    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(2024)

    # Phase 1: Spectral phase optimization
    if time.time() < deadline - 50:
        x_spec, val_spec = _spectral_phase_optimize(best_x, deadline - 50, rng)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec

    # Phase 2: Fast local search
    if time.time() < deadline - 5:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 5, rng, max_rounds=100000)

    return best_x
# EVOLVE-BLOCK-END

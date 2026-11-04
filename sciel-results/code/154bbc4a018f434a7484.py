import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

# Number of frequency bins we can control (rfft gives _FFT_SIZE//2 + 1 bins)
N_FREQ = _FFT_SIZE // 2 + 1


def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    return np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]


def _metric(x):
    """max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)


def _metric_offdiag(x):
    """max |autoconv[k]| for k != 0, divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    # Exclude lag 0
    max_off = np.max(np.abs(c[1:])) if len(c) > 1 else 0.0
    return float(max_off) / (s * s)


def _soft_max_abs(c, beta=100.0):
    """Smooth approximation of max(|c|) using log-sum-exp."""
    abs_c = np.abs(c)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    shifted = abs_c - m
    return m + np.log(np.sum(np.exp(beta * shifted))) / beta


def _objective_phases(phases, beta=100.0, peak_weight=1.0):
    """
    Objective in frequency domain: given phases of rfft, reconstruct signal,
    compute autoconvolution, return smooth max |autoconv| / (sum)^2.
    We also add a penalty for the DC bin magnitude deviation to keep sum nonzero.
    """
    # Reconstruct complex spectrum with unit magnitude and given phases
    # The DC bin (index 0) is special: for real signal, X[0] is real.
    # We'll set X[0] to be real and positive (or controlled by first phase).
    mag = np.ones(N_FREQ)
    # For a real signal, the rfft output has the constraint that X[0] is real.
    # Let's handle this: set phase[0] to 0 (so X[0] = 1, real positive)
    # Actually, let's allow phase[0] to control the sign of the DC component.
    # X[0] = exp(i * phase[0]) but for real signal it must be real.
    # So we'll just use cos(phase[0]) for the DC magnitude sign, or fix it.
    # Simpler: fix X[0] = 1 (real), and optimize the rest.
    X = np.exp(1j * phases)
    X[0] = 1.0  # Fix DC to be real and positive
    
    # Inverse rfft to get time-domain signal
    x = np.fft.irfft(X, _FFT_SIZE)[:N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    
    # Compute autoconvolution
    fx = np.fft.rfft(x, _FFT_SIZE)
    c = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    
    # Soft max of |c| excluding lag 0
    if len(c) > 1:
        soft_max = _soft_max_abs(c[1:], beta)
    else:
        soft_max = 0.0
    
    return soft_max / (s * s)


def _objective_phases_with_peak_penalty(phases, beta=100.0, peak_weight=1.0):
    """
    Objective that explicitly penalizes peak-to-mean ratio of autocorrelation.
    This is equivalent to minimizing max|c_k|/mean|c_k| type quantity,
    but we use the normalized version.
    """
    mag = np.ones(N_FREQ)
    X = np.exp(1j * phases)
    X[0] = 1.0
    
    x = np.fft.irfft(X, _FFT_SIZE)[:N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    
    fx = np.fft.rfft(x, _FFT_SIZE)
    c = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    
    if len(c) > 1:
        offdiag = c[1:]
        abs_offdiag = np.abs(offdiag)
        soft_max = _soft_max_abs(offdiag, beta)
        mean_val = np.mean(abs_offdiag) if len(abs_offdiag) > 0 else 1e-10
        # Peak-to-mean ratio, normalized
        if mean_val > 1e-10:
            peak_to_mean = soft_max / mean_val
        else:
            peak_to_mean = soft_max
    else:
        peak_to_mean = 0.0
    
    base_metric = _soft_max_abs(c[1:], beta) / (s * s) if len(c) > 1 else 0.0
    return base_metric + peak_weight * peak_to_mean / (s * s) if s != 0 else 1e10


def _phases_to_signal(phases):
    """Convert phase vector to time-domain signal with unit sum."""
    X = np.exp(1j * phases)
    X[0] = 1.0
    x = np.fft.irfft(X, _FFT_SIZE)[:N]
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s
    return x


def _generate_random_phases(rng, n_phase):
    """Generate random phase vector."""
    phases = rng.uniform(-np.pi, np.pi, size=n_phase)
    phases[0] = 0.0  # Fix DC
    return phases


def _optimize_phases_lbfgs(phases_init, deadline, max_iters=500, beta=100.0):
    """Optimize phases using L-BFGS-B."""
    result = minimize(
        _objective_phases,
        phases_init,
        method='L-BFGS-B',
        options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8},
        args=(beta,)
    )
    return result.x, result.fun


def _optimize_phases_sine_sum(rng, deadline, n_harmonics=10, max_iters=200):
    """
    Generate signals as sums of sinusoids with optimized amplitudes and phases.
    x[n] = sum_k a_k * cos(2*pi*f_k*n + phi_k)
    This is a different structural family.
    """
    # Choose random frequencies (in cycles/sample)
    freqs = rng.uniform(0.01, 0.49, size=n_harmonics)
    # Initial amplitudes and phases
    amps = rng.uniform(0.1, 1.0, size=n_harmonics)
    phis = rng.uniform(-np.pi, np.pi, size=n_harmonics)
    
    def _sine_signal(params):
        a = params[:n_harmonics]
        p = params[n_harmonics:]
        x = np.zeros(N)
        n = np.arange(N)
        for k in range(n_harmonics):
            x += a[k] * np.cos(2 * np.pi * freqs[k] * n + p[k])
        s = np.sum(x)
        if abs(s) > 1e-12:
            x = x / s
        return x
    
    def _sine_objective(params):
        x = _sine_signal(params)
        c = _autoconv_lin(x)
        if len(c) > 1:
            return _soft_max_abs(c[1:], 100.0) / (np.sum(x) ** 2)
        return 1e10
    
    best_params = np.concatenate([amps, phis])
    best_val = _sine_objective(best_params)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        result = minimize(
            _sine_objective,
            best_params,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    
    x = _sine_signal(best_params)
    return x, _metric(x)


def _subgradient_polish(x, max_iters=50, tol=1e-8):
    """Subgradient refinement for the non-smooth max|c_k| objective."""
    x = x.copy()
    n = len(x)
    
    for _ in range(max_iters):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return x
        
        c = _autoconv_lin(x)
        abs_c = np.abs(c)
        m = np.max(abs_c)
        
        if m < 1e-15:
            return x
        
        # Find which lags are at the maximum
        threshold = m * 0.999
        active_mask = abs_c >= threshold
        
        # Compute subgradient w.r.t. x using convolution structure
        grad = np.zeros(n)
        
        for k in range(len(c)):
            if not active_mask[k]:
                continue
            sign_k = np.sign(c[k]) if c[k] != 0 else 1.0
            # c[k] = sum_i x[i] x[i+k] (with appropriate bounds)
            # d/dc_k / dx[i] = x[i+k] + x[i-k] (where defined)
            for i in range(n):
                contrib = 0.0
                if i + k < n:
                    contrib += x[i + k]
                if i - k >= 0:
                    contrib += x[i - k]
                grad[i] += sign_k * contrib
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            return x
        
        step_size = 0.05 / grad_norm
        x_new = x - step_size * grad
        
        # Project: preserve sum
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new * (s / s_new)
        
        c_new = _autoconv_lin(x_new)
        m_new = np.max(np.abs(c_new))
        
        if abs(m_new - m) < tol:
            return x_new
        
        x = x_new
    
    return x


def _differential_evolution_phases(rng, deadline, pop_size=20, max_gen=30, n_phase=31):
    """
    Differential evolution over the phase space.
    This is a global optimization method that can escape local minima.
    """
    # Initialize population
    population = [_generate_random_phases(rng, n_phase) for _ in range(pop_size)]
    # Add a few structured initializations
    population[0] = np.zeros(n_phase)  # Flat spectrum
    if n_phase > 2:
        population[1] = np.linspace(0, 2*np.pi, n_phase)
        population[1][0] = 0.0
    
    best_phases = population[0].copy()
    best_val = _objective_phases(best_phases, 100.0)
    
    # Evaluate initial population
    for i, ph in enumerate(population):
        v = _objective_phases(ph, 100.0)
        if v < best_val:
            best_val = v
            best_phases = ph.copy()
    
    for gen in range(max_gen):
        if time.time() > deadline:
            break
        
        new_population = [best_phases.copy()]  # Keep best
        
        for i in range(pop_size - 1):
            # Differential evolution mutation
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            F = 0.8  # Mutation factor
            mutant = population[a] + F * (population[b] - population[c])
            # Wrap phases to [-pi, pi]
            mutant = (mutant + np.pi) % (2 * np.pi) - np.pi
            mutant[0] = 0.0
            
            # Crossover
            CR = 0.9
            trial = mutant.copy()
            for j in range(n_phase):
                if rng.random() < CR or j == 0:
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i % pop_size][j]
            
            # Evaluate
            v_trial = _objective_phases(trial, 100.0)
            v_current = _objective_phases(population[i % pop_size], 100.0)
            
            if v_trial < v_current:
                new_population.append(trial)
                if v_trial < best_val:
                    best_val = v_trial
                    best_phases = trial.copy()
            else:
                new_population.append(population[i % pop_size].copy())
        
        population = new_population
    
    x = _phases_to_signal(best_phases)
    return x, _metric(x)


def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Strategy 1: Differential evolution over frequency phases (global search)
    if time.time() < deadline - 40:
        try:
            n_phase = min(N_FREQ, 31)  # Use up to 31 phase parameters
            x_de, v_de = _differential_evolution_phases(
                rng, deadline - 40, pop_size=15, max_gen=25, n_phase=n_phase
            )
            if v_de < best_val:
                best_val = v_de
                best_x = x_de
        except Exception:
            pass
    
    # Strategy 2: L-BFGS-B polish on the best phase solution
    if time.time() < deadline - 20:
        try:
            # Extract phases from best_x
            fx = np.fft.rfft(best_x, _FFT_SIZE)
            phases_init = np.angle(fx)[:min(N_FREQ, 31)]
            phases_init[0] = 0.0
            
            x_lbfgs, v_lbfgs = _optimize_phases_lbfgs(
                phases_init, deadline - 20, max_iters=200, beta=100.0
            )
            if v_lbfgs < best_val:
                best_val = v_lbfgs
                best_x = _phases_to_signal(x_lbfgs)
        except Exception:
            pass
    
    # Strategy 3: Sine-sum structural family
    if time.time() < deadline - 15:
        try:
            for n_harm in [5, 8, 12]:
                if time.time() > deadline - 12:
                    break
                x_sine, v_sine = _optimize_phases_sine_sum(
                    rng, deadline - 12, n_harmonics=n_harm, max_iters=30
                )
                if v_sine < best_val:
                    best_val = v_sine
                    best_x = x_sine
        except Exception:
            pass
    
    # Strategy 4: Subgradient polish on the best continuous solution
    if time.time() < deadline - 10:
        try:
            x_sub = _subgradient_polish(best_x, max_iters=40, tol=1e-8)
            v_sub = _metric(x_sub)
            if v_sub < best_val:
                best_val = v_sub
                best_x = x_sub
        except Exception:
            pass
    
    # Strategy 5: Another round of differential evolution with smaller population
    if time.time() < deadline - 5:
        try:
            n_phase = min(N_FREQ, 20)
            x_de2, v_de2 = _differential_evolution_phases(
                rng, deadline - 5, pop_size=10, max_gen=15, n_phase=n_phase
            )
            if v_de2 < best_val:
                best_val = v_de2
                best_x = x_de2
        except Exception:
            pass
    
    # Final subgradient polish
    if time.time() < deadline - 2:
        try:
            x_final = _subgradient_polish(best_x, max_iters=20, tol=1e-10)
            v_final = _metric(x_final)
            if v_final < best_val:
                best_val = v_final
                best_x = x_final
        except Exception:
            pass
    
    return best_x

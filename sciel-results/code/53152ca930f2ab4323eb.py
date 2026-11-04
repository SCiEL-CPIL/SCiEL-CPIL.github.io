"""Rewritten program for the third autocorrelation inequality task using model-and-optimize."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 145.0  # wall-clock limit with safety margin


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _build_h(params):
    """
    Build h from parameters: h[n] = offset + sum_k amp[k] * cos(2*pi*freq[k]*n/N + phase[k])
    params layout: [offset, freq1, phase1, amp1, freq2, phase2, amp2, ...]
    """
    offset = params[0]
    n = np.arange(N)
    h = np.full(N, offset)
    for k in range(1, len(params) - 2, 3):
        freq = params[k]
        phase = params[k + 1]
        amp = params[k + 2]
        h += amp * np.cos(2 * np.pi * freq * n / N + phase)
    return h


def _build_chirp(params):
    """
    Build h from chirp parameters: h[n] = offset + amp * cos(2*pi*(f0*n + chirp*n^2/N) + phase)
    params layout: [offset, amp, f0, chirp, phase]
    """
    offset, amp, f0, chirp, phase = params
    n = np.arange(N)
    phase_arg = 2 * np.pi * (f0 * n / N + chirp * n**2 / N**2) + phase
    h = offset + amp * np.cos(phase_arg)
    return h


def _chirp_objective(params):
    """Objective for chirp parameters."""
    h = _build_chirp(params)
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val


def _objective(params, n_modes):
    """
    Objective function: minimize the task value given sinusoid parameters.
    params: [offset, freq1, phase1, amp1, freq2, phase2, amp2, ...]
    """
    h = _build_h(params)
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val


def _optimize_one(n_modes, init_params, deadline):
    """
    Optimize sinusoid parameters for a given number of modes.
    Returns (best_params, best_value).
    """
    try:
        res = minimize(
            _objective,
            init_params,
            args=(n_modes,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        best_params = res.x
        best_val = _objective(best_params, n_modes)
        return best_params, best_val
    except Exception:
        return init_params, _objective(init_params, n_modes)


def _local_refine(h, deadline, max_iter=20000):
    """
    Local refinement: try small random perturbations and keep improvements.
    """
    best_val = _evaluate(h)
    best_h = h.copy()
    rng = np.random.default_rng(42)
    
    step_size = 0.1
    for step in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try perturbing 1-5 random indices
        n_perturb = rng.integers(1, 6)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _evaluate(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-10:
            best_val = v_new
            best_h = h_new
            # Reduce step size on success
            step_size *= 0.98
        elif step % 1000 == 0:
            # Gradually reduce step size if no recent improvement
            step_size *= 0.999
    
    return best_h, best_val


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    Uses a model-and-optimize approach: parameterize h as sinusoids and fit parameters.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    rng = np.random.default_rng(123)
    
    # Track best solution
    best_val = 2.0  # seed value
    best_h = np.ones(N)
    
    # Phase 1: Optimize sinusoid models with varying numbers of modes
    # Try different numbers of sinusoidal modes (1 to 8)
    for n_modes in range(1, 9):
        if time.time() > deadline - 15:
            break
        
        # Generate multiple initial parameter sets
        n_inits = max(3, 12 - n_modes)  # More inits for fewer modes
        for init_idx in range(n_inits):
            if time.time() > deadline - 10:
                break
            
            # Build initial parameters
            offset = rng.uniform(0.5, 2.0)
            params = [offset]
            for k in range(n_modes):
                freq = rng.uniform(1, 10)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.1, 1.5)
                params.extend([freq, phase, amp])
            
            # Optimize
            opt_params, opt_val = _optimize_one(n_modes, params, deadline - 5)
            
            if np.isfinite(opt_val) and opt_val < best_val:
                best_val = opt_val
                best_h = _build_h(opt_params)
    
    # Phase 2: Chirp optimization using differential evolution + Nelder-Mead
    if time.time() < deadline - 20:
        # Define bounds for chirp parameters: [offset, amp, f0, chirp, phase]
        bounds = [
            (0.1, 3.0),      # offset
            (0.1, 2.0),      # amp
            (0.5, 10.0),     # f0
            (0.01, 5.0),     # chirp
            (0.0, 2 * np.pi) # phase
        ]
        
        # Use differential evolution for global search
        try:
            de_result = differential_evolution(
                _chirp_objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                polish=True
            )
            if de_result.success and de_result.fun < best_val:
                best_val = de_result.fun
                best_h = _build_chirp(de_result.x)
        except Exception:
            pass
        
        # If DE didn't find improvement, try Nelder-Mead from a few random starts
        if time.time() < deadline - 15:
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                init_params = np.array([
                    rng.uniform(0.5, 2.0),
                    rng.uniform(0.1, 1.5),
                    rng.uniform(1.0, 5.0),
                    rng.uniform(0.1, 2.0),
                    rng.uniform(0, 2 * np.pi)
                ])
                try:
                    nm_result = minimize(
                        _chirp_objective,
                        init_params,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if nm_result.fun < best_val:
                        best_val = nm_result.fun
                        best_h = _build_chirp(nm_result.x)
                except Exception:
                    pass
    
    # Phase 3: Local refinement of the best candidate
    if time.time() < deadline - 10:
        best_h, best_val = _local_refine(best_h, deadline - 5)
    
    # Phase 4: Try structured patterns that might have low autocorrelation
    if time.time() < deadline - 15:
        # Pattern 1: Alternating signs with varying magnitudes
        # h[n] = (-1)^n * (a + b*sin(2*pi*c*n/N))
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            a = rng.uniform(0.5, 2.0)
            b = rng.uniform(0.1, 1.0)
            c = rng.uniform(1, 5)
            phase = rng.uniform(0, 2 * np.pi)
            n = np.arange(N)
            h_test = (-1)**n * (a + b * np.sin(2 * np.pi * c * n / N + phase))
            
            if np.sum(h_test) == 0:
                continue
            
            # Optimize this as a starting point
            h_opt, v_opt = _local_refine(h_test, deadline - 5, max_iter=5000)
            if np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
        
        # Pattern 2: Block patterns with alternating signs
        for trial in range(15):
            if time.time() > deadline - 10:
                break
            
            n_blocks = rng.integers(4, 20)
            block_size = N // n_blocks
            h_test = np.zeros(N)
            sign = 1
            for i in range(0, N, block_size):
                val = rng.uniform(0.5, 1.5)
                h_test[i:min(i + block_size, N)] = sign * val
                sign *= -1
            
            if np.sum(h_test) == 0:
                continue
            
            h_opt, v_opt = _local_refine(h_test, deadline - 5, max_iter=5000)
            if np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
        
        # Pattern 3: Low-frequency dominant patterns
        for trial in range(15):
            if time.time() > deadline - 10:
                break
            
            # Very low frequency to make autocorrelation flatter
            freq = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.5, 1.5)
            offset = rng.uniform(0.3, 1.0)
            n = np.arange(N)
            h_test = offset + amp * np.cos(2 * np.pi * freq * n / N + phase)
            
            if np.sum(h_test) == 0:
                continue
            
            h_opt, v_opt = _local_refine(h_test, deadline - 5, max_iter=5000)
            if np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
    
    # Phase 5: Final local refinement
    if time.time() < deadline - 5:
        best_h, best_val = _local_refine(best_h, deadline - 2, max_iter=10000)
    
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        # Fall back to seed if optimization failed
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END

"""Combined approach: M-sequence structured initialization with smooth log-sum-exp optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _smooth_objective(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)

def _smooth_grad(h, beta=50.0):
    """
    Gradient of the smooth objective f(h) = | 2N * softmax(conv) / s^2 |.
    """
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N)
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    # dM/dh_j = 2 * np.correlate(weights, h, mode='valid')
    d_soft_max = 2.0 * np.correlate(weights, h, mode='valid')
    
    # Objective: f = 2N * soft_max / s^2 (assuming positive)
    sign = 1.0
    if (2.0 * N * soft_max / (s * s)) < 0:
        sign = -1.0
        
    grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    
    return grad

def _generate_m_sequence(n, seed=1):
    """
    Generate an M-sequence (maximal length sequence) using a linear feedback shift register.
    For n bits, we need a primitive polynomial. Common ones:
    n=8: x^8 + x^4 + x^3 + x^2 + 1
    n=10: x^10 + x^3 + 1
    n=12: x^12 + x^6 + x^4 + x + 1
    n=15: x^15 + x + 1
    n=16: x^16 + x^5 + x^3 + x^2 + 1
    n=20: x^20 + x^3 + 1
    n=24: x^24 + x^5 + x^3 + x + 1
    n=30: x^30 + x^6 + x^4 + x + 1
    n=32: x^32 + x^7 + x^5 + x^3 + 1
    """
    # Choose n that gives us at least N bits
    # We'll use n=16 (65535 bits) which is more than enough for N=400
    # But let's pick a reasonable n. Let's use n=16 for good properties.
    
    # Primitive polynomials for various n (coefficients for x^0 to x^n-1, with x^n implicit)
    # Format: list of positions of 1s (excluding the leading x^n term)
    primitive_polys = {
        4: [0, 1],       # x^4 + x + 1
        5: [0, 2],       # x^5 + x^2 + 1
        6: [0, 1, 2],    # x^6 + x + 1 ... actually x^6 + x^5 + 1
        7: [0, 1, 2, 4], # x^7 + x + 1 ... let me use known ones
        8: [0, 1, 2, 3], # x^8 + x^4 + x^3 + x^2 + 1
        10: [0, 2],      # x^10 + x^3 + 1
        12: [0, 1, 3, 5],# x^12 + x^6 + x^4 + x + 1
        15: [0],         # x^15 + x + 1
        16: [0, 1, 2, 4],# x^16 + x^5 + x^3 + x^2 + 1
        20: [0, 2],      # x^20 + x^3 + 1
        24: [0, 1, 2, 4],# x^24 + x^5 + x^3 + x + 1
        30: [0, 1, 3, 5],# x^30 + x^6 + x^4 + x + 1
        32: [0, 2, 4, 6],# x^32 + x^7 + x^5 + x^3 + 1
    }
    
    # Use n=16
    n_lfsr = 16
    taps = primitive_polys[n_lfsr]
    
    # Initialize LFSR with seed
    state = seed
    if state == 0:
        state = 1
    
    sequence = []
    for _ in range(n):
        # Compute feedback bit
        feedback = 0
        for tap in taps:
            if (state >> tap) & 1:
                feedback ^= 1
        
        # Shift right and insert feedback at MSB
        state = (state >> 1) | (feedback << (n_lfsr - 1))
        
        # Output bit is LSB
        sequence.append(state & 1)
    
    return np.array(sequence, dtype=np.float64)

def _generate_structured_inits():
    """Generate diverse structured initial candidates."""
    candidates = []
    t = np.arange(N)
    
    # M-sequence based
    for seed in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
        seq = _generate_m_sequence(N, seed=seed)
        # Map to {-1, 1}
        h = 2.0 * seq - 1.0
        # Add small DC offset to ensure non-zero sum
        s = np.sum(h)
        if abs(s) < 1e-10:
            h[0] += 0.01
        candidates.append(h.copy())
        # Also try with different scaling
        candidates.append(h * 0.5 + 0.25)
    
    # Cosine modes
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * t / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Multi-frequency cosine
    rng = np.random.default_rng(42)
    for trial in range(20):
        n_freq = rng.integers(2, 6)
        freqs = sorted(rng.uniform(1, 40, n_freq))
        phases = rng.uniform(0, 2*np.pi, n_freq)
        amps = rng.uniform(0.2, 1.0, n_freq)
        dc = rng.uniform(-0.1, 0.1)
        h = np.ones(N) * dc
        for f, p, a in zip(freqs, phases, amps):
            h += a * np.cos(2*np.pi*f*t/N + p)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Block structure
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(3):
            rng_block = np.random.default_rng(seed + 1000)
            h = np.zeros(N)
            for b in range(nblocks):
                val = rng_block.normal(0, 1)
                h[b*block_size:(b+1)*block_size] = val
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Random Gaussian
    for trial in range(15):
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Evaluate all candidates and return top ones
    init_vals = []
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _eval_value(h)
        if np.isfinite(v):
            init_vals.append((v, h.copy()))
    
    init_vals.sort(key=lambda x: x[0])
    return init_vals

def _local_refine(h0, max_time=25.0):
    """Local refinement using smooth objective with multiple betas."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _eval_value(h0)
    
    for beta in [30.0, 60.0, 100.0, 150.0]:
        if time.time() - t0 > max_time * 0.8:
            break
        try:
            def obj_fn(h, b=beta):
                return _smooth_objective(h, beta=b)
            
            def grad_fn(h, b=beta):
                return _smooth_grad(h, beta=b)
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _eval_value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _multi_start(h_best, n_starts=5, max_time=20.0):
    """Multi-start local refinement."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _eval_value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        scale = 0.1 + 0.05 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=50.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=50.0),
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 300, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _eval_value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Start with seed
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    # Phase 1: Generate diverse structured initializations
    if time_left() > 60:
        init_vals = _generate_structured_inits()
        if init_vals:
            # Try the top few initializations
            top_k = min(5, len(init_vals))
            for i in range(top_k):
                if elapsed() > DEADLINE - 40:
                    break
                v_init, h_init = init_vals[i]
                if v_init < best_val:
                    best_val = v_init
                    best_h = h_init.copy()
    
    # Phase 2: Local refinement from best initialization
    if time_left() > 30:
        best_h, best_val = _local_refine(best_h, max_time=min(25.0, time_left() - 25.0))
    
    # Phase 3: Multi-start refinement
    if time_left() > 20:
        best_h, best_val = _multi_start(best_h, n_starts=5, max_time=min(20.0, time_left() - 15.0))
    
    # Phase 4: Final high-precision polish
    if time_left() > 10:
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=100.0),
                best_h,
                jac=lambda h: _smooth_grad(h, beta=100.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _eval_value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h

"""Program for the autocorrelation inequality task; optimized structural search."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 110.0  # Safety margin below 120s limit

def _compute_value(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)"""
    if np.sum(h) == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)  # signed max, no absolute value inside
    return abs(2 * N * max_conv / (np.sum(h) ** 2))

def _random_candidate(rng):
    """Generate a random candidate with some structure to explore the space."""
    # Mix of random patterns that might reduce peak autoconvolution
    strategy = rng.integers(0, 4)
    if strategy == 0:
        # Random signs
        h = rng.choice([-1.0, 1.0], size=N)
    elif strategy == 1:
        # Alternating blocks of random length
        h = np.zeros(N)
        i = 0
        while i < N:
            length = rng.integers(1, 10)
            sign = rng.choice([-1.0, 1.0])
            h[i:i+length] = sign
            i += length
    elif strategy == 2:
        # Smooth random walk
        h = np.cumsum(rng.standard_normal(N))
        h -= np.mean(h)  # Center it
        h = np.sign(h)  # Binarize
    else:
        # Sine wave with random frequency
        freq = rng.uniform(0.1, 5.0)
        phase = rng.uniform(0, 2 * np.pi)
        h = np.sign(np.sin(2 * np.pi * freq * np.arange(N) + phase))
    return h

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Start with the seed (constant ones) which gives value 2.0
    best_h = np.ones(N)
    best_val = 2.0
    
    # Try some structured candidates first
    candidates = []
    
    # 1. Alternating pattern
    alt = np.array([1.0, -1.0] * (N // 2))
    candidates.append(alt)
    
    # 2. Block pattern (4 blocks of 100)
    block = np.concatenate([
        np.ones(100),
        -np.ones(100),
        np.ones(100),
        -np.ones(100)
    ])
    candidates.append(block)
    
    # 3. Random walk signs
    rw = np.sign(np.cumsum(rng.standard_normal(N)))
    candidates.append(rw)
    
    # 4. Sine-based
    sine = np.sign(np.sin(2 * np.pi * 2.0 * np.arange(N) / N))
    candidates.append(sine)
    
    # 5. Pure random
    rand = rng.choice([-1.0, 1.0], size=N)
    candidates.append(rand)
    
    # Evaluate all candidates
    for h in candidates:
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT * 0.3:
        return best_h
    
    # Now do local optimization on the best candidate
    # Use gradient-free optimization (Nelder-Mead) on a subset of parameters
    # Represent h as a function of a few parameters for efficiency
    
    # Try optimizing a sine wave: h[i] = a * sin(b*i + c) + d
    def sine_params_to_h(params):
        a, b, c, d = params
        i = np.arange(N)
        h = a * np.sin(b * i + c) + d
        return h
    
    # Initial guess from best_h
    # Fit a rough sine to best_h
    i = np.arange(N)
    # Use FFT to find dominant frequency
    fft_vals = np.fft.fft(best_h)
    freqs = np.fft.fftfreq(N)
    max_freq_idx = np.argmax(np.abs(fft_vals))
    init_b = abs(freqs[max_freq_idx]) * 2 * np.pi
    init_a = np.max(best_h) - np.min(best_h)
    init_c = 0.0
    init_d = np.mean(best_h)
    
    def objective(params):
        h = sine_params_to_h(params)
        return _compute_value(h)
    
    try:
        result = minimize(
            objective,
            [init_a, init_b, init_c, init_d],
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
        )
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT * 0.6:
            best_h_opt = sine_params_to_h(result.x)
            val_opt = _compute_value(best_h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = best_h_opt.copy()
    except Exception:
        pass
    
    # Try block pattern optimization
    # h = [a]*L1 + [b]*L2 + [c]*L3 + [d]*L4 where L1+L2+L3+L4 = N
    def block_params_to_h(params):
        a, b, c, d, L1, L2, L3 = params
        L4 = N - L1 - L2 - L3
        if L4 < 1 or L1 < 1 or L2 < 1 or L3 < 1:
            return np.ones(N)  # Invalid, return seed
        h = np.concatenate([
            np.full(int(L1), a),
            np.full(int(L2), b),
            np.full(int(L3), c),
            np.full(int(L4), d)
        ])
        return h
    
    init_block = [1.0, -1.0, 1.0, -1.0, 100.0, 100.0, 100.0]
    
    def block_objective(params):
        h = block_params_to_h(params)
        return _compute_value(h)
    
    try:
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT * 0.5:
            result_block = minimize(
                block_objective,
                init_block,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-5}
            )
            elapsed = time.time() - start_time
            if elapsed < TIME_LIMIT * 0.7:
                best_h_block = block_params_to_h(result_block.x)
                val_block = _compute_value(best_h_block)
                if val_block < best_val:
                    best_val = val_block
                    best_h = best_h_block.copy()
    except Exception:
        pass
    
    # Final random restarts with time remaining
    elapsed = time.time() - start_time
    max_iter = 50
    for _ in range(max_iter):
        if time.time() - start_time > TIME_LIMIT * 0.9:
            break
        h = _random_candidate(rng)
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h

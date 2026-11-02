"""Erdos minimum overlap: Barker bang-bang + continuous bisection refinement."""
import numpy as np
import time

N = 100
DEADLINE = 145  # safety margin before 150s kill


def _time_left(t0):
    return max(0.0, DEADLINE - (time.time() - t0))


def _overlap_vector(h):
    """Compute the full overlap vector: O[k] = sum_i h[i]*(1-h[(i+k)%N])."""
    one_minus_h = 1.0 - h
    fft_h = np.fft.fft(h)
    fft_omh = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(fft_h * np.conj(fft_omh)).real
    return corr


def _max_overlap(h):
    """Maximum overlap of h with shifted (1-h)."""
    return np.max(_overlap_vector(h))


def _project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    diff = target - np.sum(h)
    for _ in range(100):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-12
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] = np.minimum(h[mask] + step, 1.0)
        else:
            mask = h > 1e-12
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] = np.maximum(h[mask] + step, 0.0)
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        interior = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(interior):
            h[interior] += diff / np.sum(interior)
            h = np.clip(h, 0.0, 1.0)
    return h


def _barker_init(seed=42):
    """Generate a Barker-like binary sequence mapped to [0,1] step heights."""
    rng = np.random.default_rng(seed)
    # Try to find a good binary pattern: minimize max autocorrelation of (2h-1)
    best_bits = None
    best_score = float('inf')
    for trial in range(200):
        bits = rng.integers(0, 2, N)
        x = 2.0 * bits - 1.0  # +/-1
        # Autocorrelation
        ac = np.fft.ifft(np.abs(np.fft.fft(x)) ** 2).real
        max_ac = np.max(ac) - ac[0]  # exclude DC
        if max_ac < best_score:
            best_score = max_ac
            best_bits = bits.copy()
    if best_bits is None:
        best_bits = rng.integers(0, 2, N)
    h = best_bits.astype(np.float64) * 0.5  # map 0/1 to 0/0.5
    # Scale to have sum = N/2: if sum is already N/2 we're done
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0, 1)
    return _project(h)


def _barker_search(t0, max_iters=5000):
    """Local search over binary patterns (flip bits) to minimize max overlap."""
    h = _barker_init()
    bits = (h > 0.25).astype(int)
    best_val = _max_overlap(h)
    best_bits = bits.copy()
    
    for it in range(max_iters):
        if _time_left(t0) < 10:
            break
        # Random flip
        idx = np.random.randint(0, N)
        bits[idx] = 1 - bits[idx]
        h_new = bits.astype(np.float64) * 0.5
        s = np.sum(h_new)
        if s > 0:
            h_new = h_new * (N / 2.0) / s
        h_new = np.clip(h_new, 0, 1)
        h_new = _project(h_new)
        val = _max_overlap(h_new)
        if val < best_val:
            best_val = val
            best_bits = bits.copy()
            h = h_new
        else:
            bits[idx] = 1 - bits[idx]  # revert
    return _project(best_bits.astype(np.float64) * 0.5 * (N / 2.0) / np.sum(best_bits.astype(np.float64) * 0.5)), best_val


def _continuous_refine(h0, t0, max_iters=3000):
    """Subgradient descent on the continuous problem with soft-min."""
    h = _project(h0.copy())
    best_h = h.copy()
    best_val = _max_overlap(h)
    alpha = 0.05
    soft_alpha = 80.0
    
    for it in range(max_iters):
        if _time_left(t0) < 8:
            break
        
        corr = _overlap_vector(h)
        m = np.max(corr)
        shifted = np.clip((corr - m) * soft_alpha, -50, 50)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        
        # Gradient of overlap w.r.t. h
        # O[k] = sum_i h[i] * (1 - h[(i+k)%N])
        # dO[k]/dh[j] = (1 - h[(j+k)%N]) - h[(j-k)%N]
        indices = np.arange(N)
        grad = np.zeros(N)
        for k in range(N):
            g_k = (1.0 - h[(indices + k) % N]) - h[(indices - k) % N]
            grad += weights[k] * g_k
        
        h_new = h - alpha * grad
        h_new = _project(h_new)
        val = _max_overlap(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
        h = h_new
        
        if it % 50 == 0 and it > 0:
            alpha *= 0.92
        soft_alpha = min(soft_alpha * 1.002, 400.0)
    
    return best_h, best_val


def _project_t(h, t):
    """Project h so that it takes values in {0, t, 1} approximately, with sum=N/2.
    Here we use a 3-level quantization: values near 0, t, 1."""
    h = np.asarray(h, dtype=np.float64).copy()
    # Quantize to nearest of {0, t, 1}
    candidates = np.array([0.0, t, 1.0])
    dists = np.abs(h[:, None] - candidates[None, :])
    h_q = candidates[np.argmin(dists, axis=1)]
    # Adjust to meet sum constraint
    h_q = _project(h_q)
    return h_q


def _construct():
    t0 = time.time()
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Barker-style binary search
    if _time_left(t0) > 30:
        h_b, val_b = _barker_search(t0, max_iters=3000)
        if val_b < best_val:
            best_val = val_b
            best_h = h_b.copy()
    
    # Phase 2: Multiple diverse initializations
    if _time_left(t0) > 20:
        inits = []
        # Uniform
        inits.append(np.full(N, 0.5))
        # Barker
        for seed in [42, 123, 999]:
            inits.append(_barker_init(seed))
        # Sinusoidal
        for freq in [1, 2, 3, 5, 7, 10]:
            for A in [0.3, 0.4, 0.5]:
                h = 0.5 + A * np.sin(2 * np.pi * freq * np.arange(N) / N)
                inits.append(_project(np.clip(h, 0, 1)))
        # Random
        rng = np.random.default_rng(777)
        for _ in range(8):
            h = rng.uniform(0, 1, N)
            inits.append(_project(h))
        
        for h0 in inits:
            val = _max_overlap(h0)
            if val < best_val:
                best_val = val
                best_h = h0.copy()
            if _time_left(t0) < 15:
                break
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_val = _max_overlap(best_h)
    
    # Phase 3: Continuous refinement from best
    if _time_left(t0) > 10:
        h_opt, val_opt = _continuous_refine(best_h, t0, max_iters=2000)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 4: Try 3-level quantization refinement
    if _time_left(t0) > 10:
        for t_val in [0.25, 0.33, 0.4, 0.5, 0.6, 0.75]:
            if _time_left(t0) < 8:
                break
            h_q = _project_t(best_h, t_val)
            val_q = _max_overlap(h_q)
            if val_q < best_val:
                best_val = val_q
                best_h = h_q.copy()
            # Refine from quantized
            if _time_left(t0) > 5:
                h_opt, val_opt = _continuous_refine(h_q, t0, max_iters=500)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
    
    # Phase 5: Final polish
    if _time_left(t0) > 3:
        h_opt, val_opt = _continuous_refine(best_h, t0, max_iters=500)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    final_h = _project(best_h)
    return final_h


def construct():
    return _construct()

"""Search for a low-autocorrelation signed step function (400 equal intervals).

Uses a continuous flat-spectrum initialization (gradient descent on L2 norm of
autocorrelation sidelobes) to produce a strong binary seed, then refines with
simulated annealing and greedy/local-search polish on the peak sidelobe metric.
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _compute_acf(h):
    """Compute aperiodic autocorrelation via FFT, return ac[0..N-1]."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1."""
    seq = np.zeros(n)
    seq[0] = 1.0
    for i in range(1, n):
        seq[i] = seq[i >> 1] * (1.0 if (i & 1) == 0 else -1.0)
    return seq


def _flip_single(h, ac, i):
    """Flip h[i] and update ac in place."""
    val = h[i]
    if i + 1 < N:
        ac[1 : N - i] -= 2.0 * val * h[i + 1 : N]
    if i > 0:
        ac[1 : i + 1] -= 2.0 * val * h[i - 1 : -1 : -1]
    h[i] = -val


def _flip_block(h, ac, i, L):
    """Flip h[i:i+L] in place and recompute ac via FFT."""
    h[i:i + L] *= -1.0
    ac[:] = _compute_acf(h)


def _continuous_flat_spectrum_init(seed, time_budget):
    """
    Continuous gradient descent on L2 norm of autocorrelation sidelobes.
    Returns a binary ±1 vector (sign of the continuous solution).
    Objective: sum_{k=1}^{N-1} ac[k]^2  (smooth surrogate for peak sidelobe).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=N)
    # Normalize to unit norm for stability
    norm = np.linalg.norm(x)
    if norm > 1e-12:
        x /= norm

    start = time.time()
    lr = 0.01
    momentum = 0.9
    v = np.zeros(N)

    while time.time() - start < time_budget:
        # Compute autocorrelation
        H = np.fft.fft(x, _FFT_SIZE)
        ac = np.fft.ifft(H * np.conj(H)).real[:N]
        
        # Gradient of sum_{k=1}^{N-1} ac[k]^2 w.r.t. x
        # ac[k] = sum_{i=0}^{N-1-k} x[i] x[i+k]
        # d/dx[j] ac[k] = sum_{i: i+k=j} x[i] + sum_{i: i=j+k} x[i+k]
        #               = x[j-k] (if j>=k) + x[j+k] (if j+k<N)
        # So grad[j] = 2 * sum_{k=1}^{N-1} ac[k] * (x[j-k] + x[j+k])  (with bounds)
        grad = np.zeros(N)
        for k in range(1, N):
            if time.time() - start > time_budget:
                break
            # For each j where j-k >= 0: contributes x[j-k]
            # For each j where j+k < N: contributes x[j+k]
            # ac[k] * d(ac[k])/dx[j]
            # d(ac[k])/dx[j] = x[j-k] (if j>=k) + x[j+k] (if j+k < N)
            # So grad[j] += 2 * ac[k] * (x[j-k] if j>=k else 0 + x[j+k] if j+k<N else 0)
            # We can compute this via shifts:
            # contribution from x[j-k]: for j in [k, N-1], add 2*ac[k]*x[j-k] to grad[j]
            # contribution from x[j+k]: for j in [0, N-1-k], add 2*ac[k]*x[j+k] to grad[j]
            if k < N:
                # x[j-k] term: grad[k:] += 2*ac[k]*x[:N-k]
                grad[k:] += 2.0 * ac[k] * x[:N - k]
                # x[j+k] term: grad[:N-k] += 2*ac[k]*x[k:]
                grad[:N - k] += 2.0 * ac[k] * x[k:]
        
        # Momentum update
        v = momentum * v - lr * grad
        x = x + v
        
        # Renormalize occasionally to prevent drift
        norm = np.linalg.norm(x)
        if norm > 1e-12:
            x /= norm
        
        # Decay learning rate
        lr *= 0.999

    # Project to binary
    return np.sign(x)


def _anneal(seed, time_budget, init_h=None):
    """
    Simulated annealing with single-flip and occasional block-flip moves.
    """
    rng = np.random.default_rng(seed)
    
    if init_h is not None:
        h = init_h.copy()
    else:
        h = rng.choice([-1.0, 1.0], size=N)
    
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = 2.0
    T_min = 1e-8
    start = time.time()
    
    while time.time() - start < time_budget:
        r = rng.random()
        if r < 0.85:
            i = int(rng.integers(0, N))
            _flip_single(h, ac, i)
            new_peak = float(np.max(np.abs(ac[1:])))
            d = new_peak - cur
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                cur = new_peak
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                _flip_single(h, ac, i)
        else:
            L = int(rng.integers(2, 9))
            i = int(rng.integers(0, N - L + 1))
            _flip_block(h, ac, i, L)
            new_peak = float(np.max(np.abs(ac[1:])))
            d = new_peak - cur
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                cur = new_peak
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                _flip_block(h, ac, i, L)
        
        T = max(T_min, T * 0.99995)
    
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """Greedy single-flip polish."""
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        
        for i in range(N):
            _flip_single(h, ac, i)
            c = float(np.max(np.abs(ac[1:])))
            if c < best_c:
                best_c = c
                best_i = i
            _flip_single(h, ac, i)
        
        if best_i < 0:
            break
        
        _flip_single(h, ac, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    
    return best_h, best


def _local_search(h, time_budget, start):
    """More aggressive local search."""
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        improved = False
        
        best_i = -1
        best_c = cur
        for i in range(N):
            _flip_single(h, ac, i)
            c = float(np.max(np.abs(ac[1:])))
            if c < best_c:
                best_c = c
                best_i = i
            _flip_single(h, ac, i)
        
        if best_i >= 0:
            _flip_single(h, ac, best_i)
            cur = best_c
            improved = True
            if cur < best:
                best = cur
                best_h = h.copy()
        
        for _ in range(20):
            if time.time() - start >= time_budget:
                break
            L = int(np.random.randint(2, 5))
            i = int(np.random.randint(0, N - L + 1))
            _flip_block(h, ac, i, L)
            c = float(np.max(np.abs(ac[1:])))
            if c < cur:
                cur = c
                if cur < best:
                    best = cur
                    best_h = h.copy()
                improved = True
            else:
                _flip_block(h, ac, i, L)
        
        if not improved:
            break
    
    return best_h, best


def construct():
    try:
        hard_deadline = 145.0
        start = time.time()
        
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)
        
        # Phase 0: Continuous flat-spectrum initialization (strong seed)
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed - 20.0
        if remaining > 5.0:
            cont_seed = _continuous_flat_spectrum_init(42, min(10.0, remaining - 5.0))
            c = _peak_sidelobe(cont_seed)
            if c < global_best:
                global_best = c
                global_best_h = cont_seed
        
        # Also try a few more continuous inits with different seeds if time allows
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed - 15.0
        for s in [123, 999]:
            if time.time() - start >= hard_deadline - 15.0:
                break
            sub = min(5.0, hard_deadline - (time.time() - start) - 15.0)
            if sub < 1.0:
                break
            cont_seed = _continuous_flat_spectrum_init(s, sub)
            c = _peak_sidelobe(cont_seed)
            if c < global_best:
                global_best = c
                global_best_h = cont_seed
        
        # Phase 1: Short anneals from multiple inits
        inits = []
        tm = _thue_morse(N)
        inits.append(('tm', tm))
        for s in [42, 12345, 99999]:
            rng = np.random.default_rng(s)
            inits.append((f'rand_{s}', rng.choice([-1.0, 1.0], size=N)))
        alt = np.ones(N)
        alt[1::2] = -1.0
        inits.append(('alt', alt))
        
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed - 10.0
        short_budget = min(4.0, remaining / max(len(inits) + 1, 1))
        if short_budget > 0.5:
            for name, init_h in inits:
                if time.time() - start >= hard_deadline - 10.0:
                    break
                sub = min(short_budget, hard_deadline - (time.time() - start) - 10.0)
                if sub < 0.5:
                    break
                h, c = _anneal(hash(name) % 100000, sub, init_h=init_h)
                if c < global_best:
                    global_best = c
                    global_best_h = h
        
        # Phase 2: Longer anneal from the best found so far
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed - 8.0
        if remaining > 3.0:
            h, c = _anneal(777777, remaining, init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Phase 3: More restarts if time allows
        restart_id = 1
        while time.time() - start < hard_deadline - 10.0:
            sub = min(8.0, hard_deadline - (time.time() - start) - 10.0)
            if sub < 1.0:
                break
            h, c = _anneal(200000 + restart_id, sub, init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1
        
        # Phase 4: Local search polish
        remaining = hard_deadline - (time.time() - start)
        if remaining > 3.0:
            h, c = _local_search(global_best_h, remaining - 2.0, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Phase 5: Final greedy polish
        remaining = hard_deadline - (time.time() - start)
        if remaining > 1.0:
            h, c = _greedy_polish(global_best_h, remaining - 0.5, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return global_best_h
    except Exception:
        return np.ones(N)

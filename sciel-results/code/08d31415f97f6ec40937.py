"""Search for a low-autocorrelation signed step function (400 equal intervals).

Combines simulated annealing with block-flip moves and fast O(N) in-place
autocorrelation updates for the peak sidelobe objective.
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
    """
    Flip h[i] and update ac in place.
    """
    val = h[i]
    
    if i + 1 < N:
        ac[1 : N-i] -= 2.0 * val * h[i+1 : N]
    if i > 0:
        ac[1 : i+1] -= 2.0 * val * h[i-1 : -1 : -1]
        
    h[i] = -val


def _flip_block(h, ac, i, L):
    """
    Flip h[i:i+L] in place and recompute ac via FFT.
    """
    h[i:i+L] *= -1.0
    ac[:] = _compute_acf(h)


def _adaptive_block_flip(h, ac, rng, T):
    """
    Adaptive block flip: larger blocks at higher temperature, smaller at lower.
    Returns the block length used.
    """
    # Normalize temperature to [0,1] range for adaptation
    T_norm = min(1.0, max(0.0, T / 2.0))
    # Block size: 2 at T=0, up to 32 at T=1
    min_L = 2
    max_L = 2 + int(30 * T_norm)
    L = int(rng.integers(min_L, max_L + 1))
    return L


def _anneal(seed, time_budget, init_h=None):
    """
    Simulated annealing with single-flip and adaptive block-flip moves.
    Tracks peak sidelobe incrementally for speed.
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
    T = 4.0  # Start hotter for more exploration
    T_min = 1e-8
    start = time.time()
    
    while time.time() - start < time_budget:
        # Choose move type: balance between single and block flips
        r = rng.random()
        if r < 0.75:  # 75% single flips, 25% block flips
            # Single flip
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
            # Adaptive block flip: larger at high T, smaller at low T
            L = _adaptive_block_flip(h, ac, rng, T)
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
        
        # Faster cooling: 0.999 instead of 0.99995
        T = max(T_min, T * 0.999)
        
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """
    Greedy single-flip polish: try each position, keep the best flip.
    Uses incremental tracking to speed up.
    """
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
    """
    More aggressive local search: try all single flips, then best few, repeat.
    Also tries small block flips around the current best.
    """
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        improved = False
        
        # Pass 1: single flips
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
        
        # Pass 2: small block flips (length 2-4) near random positions
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
        
        # Start with ones as fallback
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)
        
        # Try multiple initialization strategies
        inits = []
        
        # 1. Thue-Morse
        tm = _thue_morse(N)
        inits.append(('tm', tm))
        
        # 2. Random with different seeds
        for s in [42, 12345, 99999, 777, 31337]:
            rng = np.random.default_rng(s)
            inits.append((f'rand_{s}', rng.choice([-1.0, 1.0], size=N)))
        
        # 3. Alternating
        alt = np.ones(N)
        alt[1::2] = -1.0
        inits.append(('alt', alt))
        
        # 4. Block-random: random signs per block of 8
        for s in [111, 222]:
            rng = np.random.default_rng(s)
            block_h = np.repeat(rng.choice([-1.0, 1.0], size=N // 8), 8)
            inits.append((f'block_{s}', block_h))
        
        # Allocate time: do short anneals from each init, then longer ones from best
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed - 5.0  # reserve 5s for polish
        
        # Phase 1: Short anneals from multiple inits (4s each)
        short_budget = min(4.0, remaining / max(len(inits), 1))
        if short_budget > 0.5:
            for name, init_h in inits:
                if time.time() - start >= hard_deadline - 12.0:
                    break
                sub = min(short_budget, hard_deadline - (time.time() - start) - 12.0)
                if sub < 0.5:
                    break
                h, c = _anneal(hash(name) % 100000, sub, init_h=init_h)
                if c < global_best:
                    global_best = c
                    global_best_h = h
        
        # Phase 2: Longer anneal from the best found so far
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed - 10.0
        if remaining > 3.0:
            h, c = _anneal(777777, remaining, init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Phase 3: More restarts if time allows
        restart_id = 1
        while time.time() - start < hard_deadline - 12.0:
            sub = min(6.0, hard_deadline - (time.time() - start) - 12.0)
            if sub < 1.0:
                break
            h, c = _anneal(200000 + restart_id, sub, init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1
        
        # Phase 4: Local search polish
        remaining = hard_deadline - (time.time() - start)
        if remaining > 4.0:
            h, c = _local_search(global_best_h, remaining - 3.0, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Phase 5: Final greedy polish
        remaining = hard_deadline - (time.time() - start)
        if remaining > 1.5:
            h, c = _greedy_polish(global_best_h, remaining - 0.5, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return global_best_h
    except Exception:
        return np.ones(N)

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    h = np.asarray(h, dtype=np.float64)
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * np.conj(H), n=_FFT_SIZE)[:N]
    if N <= 1:
        return 0.0
    return float(np.max(np.abs(ac[1:])))


def _anneal(seed, time_budget):
    """Simulated annealing with block-flip moves."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 5.0
    T_min = 1e-6
    start = time.time()
    while time.time() - start < time_budget:
        L = int(rng.integers(2, 33))  # block size 2-32
        if L >= N:
            L = N // 2
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        new = _peak_sidelobe(h)
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.9998)
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """Greedy single-flip polish."""
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            if time.time() - start > time_budget:
                break
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i < 0:
            break
        h[best_i] *= -1.0
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def _block_local_search(h, time_budget, start):
    """Local search with small block flips (size 2-8)."""
    rng = np.random.default_rng(42)
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start < time_budget and no_improve < max_no_improve:
        improved = False
        # Try a few random block flips
        for _ in range(50):
            if time.time() - start > time_budget:
                break
            L = int(rng.integers(2, 9))
            if L >= N:
                L = N // 2
            i = int(rng.integers(0, N - L + 1))
            h[i:i + L] *= -1.0
            new = _peak_sidelobe(h)
            if new < cur:
                cur = new
                improved = True
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                h[i:i + L] *= -1.0
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    return best_h, best


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N, dtype=np.float64)
    best_score = _peak_sidelobe(best_h)
    
    # Phase 1: Long initial anneal
    budget1 = min(40.0, deadline - (time.time() - start_time) - 10.0)
    if budget1 > 1.0:
        h, c = _anneal(12345, budget1)
        if c < best_score:
            best_score = c
            best_h = h
    
    # Phase 2: Multiple shorter restarts for diversity
    restart_id = 1
    while time.time() < deadline - 15.0:
        sub = min(12.0, deadline - (time.time() - start_time) - 15.0)
        if sub < 1.0:
            break
        h, c = _anneal(100000 + restart_id, sub)
        if c < best_score:
            best_score = c
            best_h = h
        restart_id += 1
    
    # Phase 3: Block local search from best
    remaining = deadline - (time.time() - start_time) - 5.0
    if remaining > 2.0:
        h, c = _block_local_search(best_h, min(10.0, remaining), time.time())
        if c < best_score:
            best_score = c
            best_h = h
    
    # Phase 4: Final greedy single-flip polish
    remaining = deadline - (time.time() - start_time) - 1.0
    if remaining > 0.5:
        h, c = _greedy_polish(best_h, remaining, time.time())
        if c < best_score:
            best_score = c
            best_h = h
    
    return np.ascontiguousarray(best_h, dtype=np.float64)
# EVOLVE-BLOCK-END

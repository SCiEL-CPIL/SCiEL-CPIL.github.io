import numpy as np
import time
from itertools import product

N = 400
_start_time = time.time()
_TIME_LIMIT = 145.0

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_fft(f):
    F = np.fft.rfft(f, n=_FFT_SIZE)
    C = np.fft.irfft(F * F, n=_FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return float('inf')
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _generate_msequence(m, poly_coeffs):
    L = (1 << m) - 1
    mask = (1 << m) - 1
    state = 1
    seq = []
    taps = poly_coeffs
    
    for _ in range(L):
        seq.append(state & 1)
        fb = 0
        for t in taps:
            fb ^= (state >> t) & 1
        state = (state >> 1) | (fb << (m - 1))
        state &= mask
        
    return np.array(seq, dtype=np.int8) * 2 - 1

def _local_search(f, max_passes=10, time_budget=10.0, deadline=None):
    f = f.copy().astype(np.float64)
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, float('inf')
    
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start = time.time()
    
    for _pass in range(max_passes):
        if time.time() - start > time_budget:
            break
        if deadline and time.time() > deadline:
            break
        
        improved = False
        # Try flipping each bit
        for i in range(N):
            if deadline and time.time() > deadline:
                break
            
            v = f[i]
            f[i] = -v
            new_S = S - 2 * v
            
            C_new = C.copy()
            k_start = max(0, i)
            k_end = min(2 * N - 1, i + N)
            
            idx = np.arange(k_start, k_end)
            f_idx = idx - i
            
            center_mask = (idx == 2 * i)
            delta = -4 * v * f[f_idx]
            delta[center_mask] = 0
            
            C_new[idx] += delta
            
            if new_S == 0:
                new_score = float('inf')
            else:
                new_score = np.max(np.abs(C_new)) / (new_S * new_S)
            
            if new_score < score - 1e-10:
                S = new_S
                C = C_new
                score = new_score
                improved = True
            else:
                f[i] = v
        
        if not improved:
            break
    
    return f, score

def construct():
    deadline = _start_time + _TIME_LIMIT
    
    best_f = np.ones(N, dtype=np.int8)
    best_score = _score(best_f)
    
    def try_candidate(f):
        nonlocal best_f, best_score
        sc = _score(f)
        if sc < best_score:
            best_score = sc
            best_f = f.copy()

    # Phase 0: M-sequence initializations
    polys = {
        8: [4, 3, 2, 0],
        9: [4, 0],
        10: [7, 0],
        11: [2, 0],
    }
    
    for m, taps in polys.items():
        if time.time() > deadline:
            break
        try:
            seq = _generate_msequence(m, taps)
            L = len(seq)
            if L >= N:
                for shift in range(L):
                    if time.time() > deadline:
                        break
                    f_cand = np.roll(seq, -shift)
                    f_cand = f_cand[:N]
                    try_candidate(f_cand)
        except Exception:
            pass

    # Phase 1: Other structured initializations
    f_tm = np.zeros(N, dtype=np.int8)
    f_tm[0] = 1
    for i in range(1, N):
        n = i // 2
        r = i % 2
        f_tm[i] = f_tm[n] if r == 0 else -f_tm[n]
    try_candidate(f_tm)
    
    f_alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
    try_candidate(f_alt)

    # Phase 2: Local search from best candidate
    if time.time() < deadline - 5.0:
        f_opt, sc_opt = _local_search(best_f.astype(np.float64), max_passes=20, time_budget=30.0, deadline=deadline)
        if sc_opt < best_score:
            best_score = sc_opt
            best_f = f_opt.astype(np.int8)

    # Phase 3: Multi-start local search from random restarts
    rng = np.random.default_rng(42)
    if time.time() < deadline - 10.0:
        for restart in range(20):
            if time.time() > deadline - 10.0:
                break
            f_rand = rng.choice([-1, 1], size=N).astype(np.float64)
            f_opt, sc_opt = _local_search(f_rand, max_passes=5, time_budget=2.0, deadline=deadline)
            if sc_opt < best_score:
                best_score = sc_opt
                best_f = f_opt.astype(np.int8)

    # Phase 4: Final local search from best
    if time.time() < deadline - 5.0:
        f_opt, sc_opt = _local_search(best_f.astype(np.float64), max_passes=30, time_budget=35.0, deadline=deadline)
        if sc_opt < best_score:
            best_score = sc_opt
            best_f = f_opt.astype(np.int8)

    return best_f.astype(np.int8)

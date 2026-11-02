import numpy as np
import time

N = 100
HALF = N // 2
DEADLINE = 115.0

def _autocorr(h):
    H = np.fft.fft(h)
    return np.fft.ifft(H * np.conj(H)).real

def _max_overlap(h):
    r = _autocorr(h)
    return HALF - float(np.min(r[1:]))

def _random_bitseq(rng):
    x = rng.random(N)
    idx = np.argsort(x)
    seq = np.zeros(N)
    seq[idx[HALF:]] = 1.0
    return seq

def _zadoff_chu_bitseq(root, rng):
    """Generate a Zadoff-Chu-like binary sequence."""
    seq = np.zeros(N)
    # Zadoff-Chu: x[n] = exp(-j*pi*root*n*(n+1)/N) for N even
    # Take real part threshold
    n = np.arange(N)
    phase = -np.pi * root * n * (n + 1) / N
    vals = np.cos(phase)
    # Threshold to binary with exactly HALF ones
    idx = np.argsort(vals)[::-1]
    seq[idx[:HALF]] = 1.0
    return seq

def _barker_like_bitseq(rng):
    """Try to create a Barker-like sequence."""
    seq = np.zeros(N)
    # Start with alternating pattern with some randomness
    for i in range(N):
        if rng.random() < 0.5:
            seq[i] = 1.0 if i % 2 == 0 else 0.0
        else:
            seq[i] = 0.0 if i % 2 == 0 else 1.0
    # Ensure exactly HALF ones
    ones = np.sum(seq)
    if ones != HALF:
        idx = np.argsort(seq)[::-1]
        seq = np.zeros(N)
        seq[idx[:HALF]] = 1.0
    return seq

def _local_search(h, rng, deadline, max_iters=5000):
    """Local search with single swaps."""
    h = h.copy()
    cur = _max_overlap(h)
    improved = True
    iters = 0
    
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        # Try random swaps
        for _ in range(50):
            if time.time() > deadline:
                break
            i = ones[rng.integers(len(ones))]
            j = zeros[rng.integers(len(zeros))]
            h[i], h[j] = h[j], h[i]
            newc = _max_overlap(h)
            if newc < cur:
                cur = newc
                improved = True
                # Update lists
                ones = np.where(h == 1.0)[0]
                zeros = np.where(h == 0.0)[0]
                break
            else:
                h[i], h[j] = h[j], h[i]
    
    return h, cur

def _simulated_annealing(h_init, rng, deadline, T_start=1.0, T_end=0.01, cooling=0.995):
    """Simulated annealing with swap moves."""
    h = h_init.copy()
    cur = _max_overlap(h)
    best = h.copy()
    bestc = cur
    T = T_start
    
    while T > T_end and time.time() < deadline:
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        h[i], h[j] = h[j], h[i]
        newc = _max_overlap(h)
        
        delta = newc - cur
        if delta < 0 or rng.random() < np.exp(-delta / T):
            cur = newc
            if cur < bestc:
                bestc = cur
                best = h.copy()
        else:
            h[i], h[j] = h[j], h[i]
        
        T *= cooling
    
    return best, bestc

def _multi_scale_sa(h_init, rng, deadline):
    """Run SA at multiple temperature scales."""
    best = h_init.copy()
    bestc = _max_overlap(best)
    
    scales = [1.0, 0.5, 0.2, 0.1]
    time_per_scale = (deadline - time.time()) / len(scales)
    
    for scale in scales:
        if time.time() > deadline - 5.0:
            break
        t_start = scale
        t_end = scale * 0.01
        cooling = 0.998
        result, c = _simulated_annealing(best, rng, deadline - 5.0, T_start=t_start, T_end=t_end, cooling=cooling)
        if c < bestc:
            bestc = c
            best = result
    
    return best, bestc

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    best = None
    bestc = np.inf
    deadline = t0 + DEADLINE

    # Generate diverse initial candidates
    candidates = []
    
    # Alternating
    alt = np.zeros(N); alt[::2] = 1.0
    candidates.append(alt)
    
    # Block
    blk = np.zeros(N); blk[:HALF] = 1.0
    candidates.append(blk)
    
    # Zadoff-Chu-like
    for root in [1, 3, 7, 11]:
        candidates.append(_zadoff_chu_bitseq(root, rng))
    
    # Barker-like
    for _ in range(3):
        candidates.append(_barker_like_bitseq(rng))
    
    # Random
    for _ in range(5):
        candidates.append(_random_bitseq(rng))
    
    # Evaluate all candidates
    for c in candidates:
        c_overlap = _max_overlap(c)
        if c_overlap < bestc:
            bestc = c_overlap
            best = c
    
    # Run local search on the best candidate
    if best is not None:
        h_ls, c_ls = _local_search(best, rng, deadline - 20.0)
        if c_ls < bestc:
            bestc = c_ls
            best = h_ls
    
    # Run multi-scale SA from the best candidate
    if time.time() < deadline - 10.0:
        h_sa, c_sa = _multi_scale_sa(best, rng, deadline - 5.0)
        if c_sa < bestc:
            bestc = c_sa
            best = h_sa
    
    # Final local search polish
    if best is not None and time.time() < deadline - 2.0:
        h_final, c_final = _local_search(best, rng, deadline - 1.0, max_iters=10000)
        if c_final < bestc:
            bestc = c_final
            best = h_final
    
    # If we still have time, try SA from other good candidates
    if best is not None and time.time() < deadline - 10.0:
        # Sort candidates by score and try SA from the second best
        scored = []
        for c in candidates:
            c_overlap = _max_overlap(c)
            scored.append((c_overlap, c))
        scored.sort()
        
        for score, cand in scored[1:4]:  # Try up to 3 more
            if time.time() > deadline - 5.0:
                break
            h_sa2, c_sa2 = _simulated_annealing(cand, rng, deadline - 5.0, T_start=0.5, T_end=0.05, cooling=0.997)
            if c_sa2 < bestc:
                bestc = c_sa2
                best = h_sa2
                # Polish
                h_pol, c_pol = _local_search(best, rng, deadline - 2.0, max_iters=5000)
                if c_pol < bestc:
                    bestc = c_pol
                    best = h_pol
    
    if best is None:
        best = np.full(N, 0.5)
    
    return best

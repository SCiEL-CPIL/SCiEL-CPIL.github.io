import numpy as np
import time

N = 400
# Next power of 2 >= 2N for FFT
_NFFT = 1 << (2 * N - 1).bit_length()

def _autocorr(h):
    """Compute linear autocorrelation A[0..N-1] using FFT."""
    F = np.fft.rfft(h, n=_NFFT)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_NFFT)
    return np.real(acf[:N])

def _hard_score(h):
    """
    The actual metric: max absolute value of autocorrelation lags 1..N-1.
    Lower is better.
    """
    ac = _autocorr(h)
    # ac[0] is the energy, always N for +/- 1 sequences.
    # We want to minimize the sidelobes.
    if N <= 1:
        return 0.0
    return float(np.max(np.abs(ac[1:])))

def _smooth_score(h, beta=50.0):
    """
    Smooth surrogate using log-sum-exp on squared sidelobes.
    This guides the SA to reduce the overall energy of the sidelobes,
    which correlates with reducing the peak.
    """
    ac = _autocorr(h)
    # Normalize by energy? No, energy is constant N.
    # Just use the raw values to keep scale consistent with hard score.
    sidelobes = ac[1:]
    # Square to emphasize larger values, then log-sum-exp
    y = sidelobes ** 2
    m = np.max(y)
    if m < 1e-10:
        return 0.0
    # log-sum-exp trick: log(sum(exp(y - m))) + m
    # We want a score that is close to max(y) = m when beta is large.
    # score = m + log(sum(exp(beta*(y-m))))/beta
    # Actually, simpler: just use the log-sum-exp of the squared values directly.
    # If we use beta * max + logsumexp(beta*(y-max))/beta, it approximates max.
    # Let's use a standard soft-max:
    z = beta * (y - m)
    lse = m + np.log(np.sum(np.exp(z))) / beta
    return float(lse)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    rng = np.random.default_rng(42)

    # 1. Diverse Initializations
    # We start with a few random sequences and some structured ones.
    inits = []
    # Random
    for _ in range(5):
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=N))
    # Chirp-like (quadratic phase)
    idx = np.arange(N)
    inits.append(np.cos(2 * np.pi * idx ** 2 / (2 * N)))
    inits.append(np.cos(2 * np.pi * idx ** 2 / (N + 1)))
    # All ones (seed) - just in case
    inits.append(np.ones(N))
    
    # Evaluate initializations
    best_h = None
    best_hard = float('inf')
    for h in inits:
        h = np.sign(h) # Ensure +/- 1
        val = _hard_score(h)
        if val < best_hard:
            best_hard = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_hard = _hard_score(best_h)

    # 2. Simulated Annealing with Smooth Surrogate
    # We use the smooth score for acceptance decisions to avoid getting stuck
    # on flat plateaus of the hard max function, but we track the hard score
    # to ensure we always return the best hard score found.
    
    h = best_h.copy()
    cur_hard = best_hard
    cur_smooth = _smooth_score(h, beta=50.0)
    
    T = 5.0
    T_min = 0.01
    decay = 0.999
    beta = 50.0
    
    # Block sizes for flips
    sizes = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    
    # Main SA Loop
    while time.time() < deadline - 10.0:
        # Choose a move: single flip or block flip
        if rng.random() < 0.8:
            # Block flip
            size = int(rng.choice(sizes))
            if size >= N:
                size = N // 2
            start = int(rng.integers(0, N - size + 1))
            sl = slice(start, start + size)
            old_vals = h[sl].copy()
            h[sl] *= -1.0
        else:
            # Single flip
            i = int(rng.integers(0, N))
            sl = slice(i, i+1)
            old_vals = h[sl].copy()
            h[sl] *= -1.0
            
        # Evaluate new state
        new_hard = _hard_score(h)
        new_smooth = _smooth_score(h, beta)
        
        dS = new_smooth - cur_smooth
        
        # Acceptance criterion:
        # 1. If hard score improves, always accept.
        # 2. Else, accept with probability exp(-dS / T) based on smooth score.
        if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
            cur_hard = new_hard
            cur_smooth = new_smooth
            if new_hard < best_hard:
                best_hard = new_hard
                best_h = h.copy()
        else:
            # Reject
            h[sl] = old_vals
            
        # Cool down
        T = max(T_min, T * decay)
        
        # Periodic restart if stuck? 
        # For now, just continue cooling.
        # If T is very low and we haven't improved in a long time, we could restart,
        # but the block flips should help escape local minima.

    # 3. Greedy Polish
    # Perform a final pass of greedy local search to squeeze out the last bits.
    # Try flipping every single bit and small blocks to see if hard score improves.
    h = best_h.copy()
    cur_hard = best_hard
    
    # Single bit greedy
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            old = h[i]
            h[i] = -old
            new_hard = _hard_score(h)
            if new_hard < cur_hard - 1e-12:
                cur_hard = new_hard
                improved = True
            else:
                h[i] = old
        
        if time.time() > deadline - 2.0:
            break

    # Small block greedy (size 2, 4)
    for size in [2, 4]:
        if time.time() > deadline - 2.0:
            break
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for start in range(0, N - size + 1):
                if time.time() > deadline - 1.0:
                    break
                sl = slice(start, start + size)
                old_vals = h[sl].copy()
                h[sl] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < cur_hard - 1e-12:
                    cur_hard = new_hard
                    improved = True
                else:
                    h[sl] = old_vals
                if time.time() > deadline - 1.0:
                    break

    # Final check
    if cur_hard < best_hard:
        best_hard = cur_hard
        best_h = h.copy()

    # Convert to int8 sequence as per interface
    return best_h.astype(np.int8)

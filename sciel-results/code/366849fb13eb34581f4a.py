"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute the max absolute autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 2 * n
    H = np.fft.fft(h, n=fft_size)
    acf = np.fft.ifft(np.abs(H)**2).real
    max_val = 0.0
    for k in range(1, n):
        val = abs(acf[k])
        if val > max_val:
            max_val = val
    return max_val


def _score(h):
    """Score = upper bound on C3 (lower is better)."""
    return _autocorr_peak(h)


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n: (-1)^popcount(i)."""
    # Efficient: t[i] = (-1)^(number of 1s in binary representation of i)
    tm = np.ones(n)
    for i in range(1, n):
        # Use property: t[2i] = t[i], t[2i+1] = -t[i]
        if i % 2 == 0:
            tm[i] = tm[i // 2]
        else:
            tm[i] = -tm[(i - 1) // 2]
    return tm


def _m_sequence(n, seed=1):
    """Generate a maximal-length LFSR sequence (m-sequence) of length n."""
    # Use a simple LFSR to generate pseudo-random ±1 sequence
    rng = np.random.default_rng(seed)
    # Start with a power-of-2 length and take first n
    length = 1
    while length < n:
        length *= 2
    # Generate using linear recurrence
    seq = np.zeros(length)
    seq[0] = 1
    for i in range(1, length):
        # Simple LFSR: x[i] = x[i-1] * x[i-2] (mod 2) mapped to ±1
        # Actually use a proper m-sequence via polynomial
        pass
    # Fallback: use deterministic low-discrepancy
    return _thue_morse(n)


def construct():
    """
    Construct a signed step function with N=400 intervals
    that minimizes the autoconvolution upper bound (C3).
    
    Strategy: Use Thue-Morse and related low-autocorrelation sequences
    as base, then refine with local search.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_h = np.zeros(N)
    best_score = np.inf
    
    # Candidate base sequences
    candidates = []
    
    # 1. Thue-Morse
    tm = _thue_morse(N)
    candidates.append(tm.copy())
    
    # 2. Thue-Morse with global sign flip
    candidates.append(-tm.copy())
    
    # 3. Thue-Morse shifted by various offsets
    for shift in range(0, 10, 2):
        shifted = np.roll(tm, shift)
        candidates.append(shifted)
    
    # 4. Concatenated Thue-Morse blocks with sign variations
    # Split into blocks and flip some blocks
    for block_size in [50, 100, 200]:
        n_blocks = N // block_size
        base = np.tile(tm[:block_size], n_blocks)
        if len(base) < N:
            base = np.concatenate([base, tm[:N - len(base)]])
        candidates.append(base[:N].copy())
        
        # Flip alternating blocks
        for flip_pattern in [1, 2]:
            mod_base = base.copy()
            for b in range(n_blocks):
                if b % flip_pattern == 1:
                    mod_base[b*block_size:(b+1)*block_size] *= -1
            candidates.append(mod_base[:N].copy())
    
    # 5. Random with Thue-Morse bias
    rng = np.random.default_rng(123)
    for _ in range(5):
        h_rand = tm * rng.choice([-1, 1], size=N)
        candidates.append(h_rand)
    
    # 6. Pure random ±1
    for seed in range(5):
        rng = np.random.default_rng(seed + 100)
        candidates.append(rng.choice([-1, 1], size=N).astype(float))
    
    # Evaluate all candidates and refine the best ones
    scored = []
    for h in candidates:
        if time.time() > deadline:
            break
        s = _score(h)
        scored.append((s, h))
    
    scored.sort(key=lambda x: x[0])
    
    # Refine the top candidates with local search
    for s, h in scored[:10]:
        if time.time() > deadline:
            break
        
        current_h = h.copy()
        current_score = s
        
        # Local search: flip random positions
        rng = np.random.default_rng(42)
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Try flipping a random position
            idx = rng.integers(N)
            old_val = current_h[idx]
            current_h[idx] = -old_val
            new_score = _score(current_h)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
            else:
                current_h[idx] = old_val
        
        # Also try pair flips
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            idx1 = rng.integers(N)
            idx2 = rng.integers(N)
            if idx1 == idx2:
                continue
            
            v1, v2 = current_h[idx1], current_h[idx2]
            current_h[idx1] = -v1
            current_h[idx2] = -v2
            new_score = _score(current_h)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
            else:
                current_h[idx1] = v1
                current_h[idx2] = v2
    
    # Ensure we have a valid result
    if best_score == np.inf:
        best_h = tm
        best_score = _score(tm)
    
    return best_h
# EVOLVE-BLOCK-END

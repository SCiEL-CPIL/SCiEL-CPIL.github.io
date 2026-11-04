import numpy as np
import time

N = 400
DX = 0.5 / N

def _linear_acf(h):
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    power = np.abs(H) ** 2
    acf_full = np.fft.irfft(power, n=fft_size)
    acf = np.empty(2 * n - 1, dtype=np.float64)
    acf[n-1] = acf_full[0]
    for k in range(1, n):
        acf[n-1+k] = acf_full[k]
        acf[n-1-k] = acf_full[fft_size - k]
    return acf

def _score(h):
    acf = _linear_acf(h)
    n = len(h)
    center = n - 1
    right = acf[center+1:]
    left = acf[:center]
    max_abs = max(np.max(np.abs(right)), np.max(np.abs(left)))
    return float(DX * max_abs)

def _psl_value(acf, n):
    if n <= 1:
        return 0.0
    center = n - 1
    right = acf[center+1:]
    left = acf[:center]
    return float(max(np.max(np.abs(right)), np.max(np.abs(left))))

def _compute_acf(h):
    return _linear_acf(h)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = [1.0]
    while len(seq) < n:
        seq.extend([-x for x in seq])
    return np.array(seq[:n])

def _legendre_sequence(n, p):
    """Generate Legendre sequence of length n using prime p > n."""
    seq = np.zeros(n)
    for i in range(1, n):
        # Legendre symbol (i/p)
        ls = pow(i, (p-1)//2, p)
        if ls == p-1:
            seq[i] = -1.0
        else:
            seq[i] = 1.0
    seq[0] = 1.0
    return seq

def _barker_like(n):
    """Generate Barker-like sequence (maximal length 13 for binary)."""
    # Barker sequences of length 13
    barkers = [
        [1,1,1,-1,1,-1,1],
        [1,1,1,-1,1,1,-1],
    ]
    # Extend by repeating and sign-flipping
    base = np.array(barkers[0])
    seq = np.zeros(n)
    for i in range(n):
        seq[i] = base[i % len(base)]
        if (i // len(base)) % 2 == 1:
            seq[i] = -seq[i]
    return seq

def _m_sequence(n, degree, poly):
    """Generate m-sequence (maximal length LFSR sequence)."""
    # For degree 9, we get period 511, take first n
    # Use a known primitive polynomial
    # x^9 + x^4 + 1 (one primitive polynomial for degree 9)
    m = (1 << degree) - 1
    state = 1
    seq = np.zeros(n)
    for i in range(n):
        seq[i] = 1.0 if (state & 1) else -1.0
        # Shift and feedback
        lsb = state & 1
        state >>= 1
        if lsb:
            state ^= poly  # XOR with polynomial
        state &= m
    return seq

def _structured_candidates():
    """Generate diverse structured candidates."""
    candidates = []
    
    # Thue-Morse and variants
    tm = _thue_morse(N)
    candidates.append(tm)
    candidates.append(-tm)
    # Shifted Thue-Morse
    for shift in range(1, 8):
        candidates.append(np.roll(tm, shift))
        candidates.append(np.roll(-tm, shift))
    
    # Legendre sequences
    primes = [409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    for p in primes:
        if p > N:
            leg = _legendre_sequence(N, p)
            candidates.append(leg)
            candidates.append(-leg)
    
    # m-sequences (degree 9, period 511)
    # Primitive polynomials for degree 9
    polys = [
        0x0130,  # x^9 + x^4 + 1
        0x0160,  # x^9 + x^5 + 1
        0x0190,  # x^9 + x^6 + 1
        0x01C0,  # x^9 + x^7 + 1
        0x0230,  # x^9 + x^4 + x^5 + 1
    ]
    for poly in polys:
        mseq = _m_sequence(N, 9, poly)
        candidates.append(mseq)
        candidates.append(-mseq)
        # Shifted versions
        for shift in range(1, 5):
            candidates.append(np.roll(mseq, shift))
    
    # Barker-like
    candidates.append(_barker_like(N))
    candidates.append(-_barker_like(N))
    
    # Zadoff-Chu-like (quadratic chirp)
    t = np.arange(N, dtype=np.float64)
    for root in [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        val = np.cos(2.0 * np.pi * root * t * t / N)
        candidates.append(np.sign(val))
    
    # Mixed chirps
    rng = np.random.default_rng(123)
    for trial in range(30):
        alpha = rng.uniform(0.01, 0.5)
        beta = rng.uniform(0.001, 0.1)
        gamma = rng.uniform(0.0, 2.0 * np.pi)
        delta = rng.uniform(-0.5, 0.5)
        val = (np.cos(alpha * t * t)
               + beta * np.cos(2.0 * np.pi * t / N + gamma)
               + delta)
        candidates.append(np.sign(val))
    
    # Quasi-periodic
    alphas = [np.sqrt(2), np.sqrt(3), (1 + np.sqrt(5)) / 2, np.sqrt(5)]
    phases = [0.0, np.pi / 4, np.pi / 3, np.pi / 2]
    for alpha in alphas:
        for phase in phases:
            vals = np.sin(np.pi * alpha * t + phase)
            signs = np.sign(vals)
            signs[signs == 0] = 1.0
            candidates.append(signs)
    
    # Random
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Periodic
    for period in [3, 5, 7, 11, 13, 17]:
        pattern = np.sign(np.cos(2.0 * np.pi * np.arange(N) / period))
        candidates.append(pattern)
    
    # Ensure no all-zeros
    for i, c in enumerate(candidates):
        if np.all(c == 0):
            candidates[i] = np.ones(N)
    
    return candidates

def _local_search(h, time_budget, start_time, rng, max_iter=5000):
    """
    Local search with batch flips and momentum.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    best_h = h.copy()
    best_psl = current_psl
    
    # Momentum for recent flips
    flip_history = {}
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Try random single flip
        i = int(rng.integers(n))
        
        # Compute delta for this flip
        delta = np.zeros(2 * n - 1, dtype=np.float64)
        for k in range(1, n):
            term1 = 0.0
            term2 = 0.0
            if i + k < n:
                term1 = h[i + k]
            if i - k >= 0:
                term2 = h[i - k]
            delta[k] = -2.0 * h[i] * (term1 + term2)
            delta[-k] = delta[k]
        
        new_acf = acf + delta
        new_psl = _psl_value(new_acf, n)
        delta_psl = new_psl - current_psl
        
        # Accept if improvement or with some probability
        if delta_psl < 0 or (current_psl > 0 and rng.random() < np.exp(-delta_psl / max(0.1, current_psl * 0.1))):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        # Every 100 iterations, try a batch of flips
        if iteration % 100 == 50:
            # Try flipping 3-5 random positions simultaneously
            num_flips = int(rng.integers(3, 6))
            indices = rng.choice(n, size=num_flips, replace=False)
            
            # Compute combined delta
            batch_delta = np.zeros(2 * n - 1, dtype=np.float64)
            for idx in indices:
                for k in range(1, n):
                    term1 = 0.0
                    term2 = 0.0
                    if idx + k < n:
                        term1 = h[idx + k]
                    if idx - k >= 0:
                        term2 = h[idx - k]
                    batch_delta[k] += -2.0 * h[idx] * (term1 + term2)
                    batch_delta[-k] += batch_delta[k]
            
            new_acf_batch = acf + batch_delta
            new_psl_batch = _psl_value(new_acf_batch, n)
            
            if new_psl_batch < current_psl:
                for idx in indices:
                    h[idx] = -h[idx]
                acf = _compute_acf(h)
                current_psl = _psl_value(acf, n)
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    rng = np.random.default_rng(42)
    
    # Generate structured candidates
    candidates = _structured_candidates()
    
    # Score all candidates
    scored = []
    for h in candidates:
        s = _score(h)
        scored.append((s, h))
    scored.sort(key=lambda x: x[0])
    
    # Refine top candidates with local search
    top_k = 15
    for i, (s, h) in enumerate(scored[:top_k]):
        if time.time() > deadline - 20:
            break
        time_limit = deadline - 15
        h_refined, s_refined = _local_search(
            h, time_budget=time_limit, start_time=start_time, rng=rng)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Random restarts with local search
    while time.time() < deadline - 10:
        # Mix of random and structured initializations
        if rng.random() < 0.5:
            h_init = rng.choice([-1.0, 1.0], size=N)
        else:
            # Perturb a good structured candidate
            base = scored[rng.integers(min(top_k, len(scored)))][1]
            h_init = base.copy()
            num_flips = rng.integers(5, 20)
            flip_idx = rng.choice(N, size=num_flips, replace=False)
            h_init[flip_idx] *= -1
        
        time_limit = min(deadline - 5, time.time() + 5)
        h_refined, s_refined = _local_search(
            h_init, time_budget=time_limit, start_time=start_time, rng=rng)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Final polish on best found
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(15.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 3.0:
            h_polished, s_polished = _local_search(
                best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_polished < best_score:
                best_score = s_polished
                best_h = h_polished.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h

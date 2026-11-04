import numpy as np
import time

N = 400

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return np.max(np.abs(acf[1:N]))

def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p)."""
    if n % p == 0:
        return 0
    return 1 if pow(n, (p-1)//2, p) == 1 else -1

def _legendre_sequence(N, p):
    """Generate a Legendre sequence of length N using prime p."""
    # Map indices 0..N-1 to 1..N (avoiding 0)
    # Actually, standard Legendre sequences use 1..p-1.
    # If N < p, we can just take the first N non-zero residues? Or map 0..N-1 to 1..N?
    # Let's map i in 0..N-1 to i+1. If i+1 >= p, we can't use Legendre directly for that index if we strictly follow field elements.
    # However, for p close to N, we can just evaluate for x=1..N.
    seq = np.zeros(N)
    for i in range(N):
        x = i + 1
        if x < p:
            seq[i] = _legendre_symbol(x, p)
        else:
            # If x >= p, this index is outside the field.
            # We can wrap around or set to 0?
            # Better to choose p > N.
            # Since N=400, p=401 is perfect.
            pass
    return seq

def _m_sequence(N, m):
    """Generate an m-sequence (maximal length LFSR sequence) of length N."""
    # We need a primitive polynomial.
    # For N=400, we don't have a single m-sequence of length 400.
    # m-sequences have length 2^m - 1.
    # 2^8 - 1 = 255, 2^9 - 1 = 511.
    # We can use a 9-bit LFSR and take the first 400 bits.
    # Primitive polynomial for GF(2^9): x^9 + x^4 + 1 (0x111 in hex, bits 9, 4, 0)
    # Or x^9 + x^5 + 1? Let's use a known one.
    # x^9 + x^4 + 1 is primitive.
    # Feedback taps: 9, 4, 0.
    # State is 9 bits.
    # Initial state: all 1s (non-zero).
    state = (1 << 9) - 1
    seq = np.zeros(N)
    for i in range(N):
        # Output bit is LSB
        seq[i] = 1.0 if (state & 1) == 1 else -1.0
        # Feedback bit
        fb = ((state >> 8) ^ (state >> 3)) & 1  # Taps at 9 (index 8) and 4 (index 3)
        state = (state >> 1) | (fb << 8)
    return seq

def _local_search(h, time_limit):
    """Greedy local search: flip the sign of the element that most reduces the score."""
    h = h.copy()
    n = len(h)
    current_score = _score(h)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_delta = 0
        best_pos = -1
        best_new_score = current_score
        
        # Randomize order to escape symmetry/local minima slightly
        indices = np.random.permutation(n)
        
        for j in indices:
            h[j] = -h[j]
            new_score = _score(h)
            if new_score < best_new_score - 1e-9:
                best_new_score = new_score
                best_pos = j
                best_delta = new_score - current_score
            h[j] = h[j]  # revert
        
        if best_pos != -1:
            h[best_pos] = -h[best_pos]
            current_score = best_new_score
            improved = True
            
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # A. Number Theoretic Sequences (Structural Idea from Directive)
    # Legendre Sequence with p=401
    p = 401
    leg_seq = _legendre_sequence(N, p)
    candidates.append(leg_seq)
    
    # M-Sequence (truncated)
    m_seq = _m_sequence(N, 9)
    candidates.append(m_seq)
    
    # B. Random and Quasi-periodic (from Current Program)
    # Random
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # Quasi-periodic (Irrational frequencies)
    for trial in range(5):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # Block structure
    block_size = 10
    for i in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)

    # 2. Evaluate and sort candidates
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # 3. Refine top candidates with local search
    # Prioritize the number-theoretic ones if they are in the top 5, but ensure we try them.
    # Let's force the Legendre and M-sequences to be refined if they are among the top 10, 
    # or just take the top 5 overall.
    
    # Ensure Legendre and M-seq are considered for refinement if they are competitive
    # But the sort handles that.
    
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 20:
            break
        time_limit = deadline - 10
        h_refined, s_refined = _local_search(h, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # 4. If time remains, do a few random restarts with local search
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 5, time.time() + 5)
        h_refined, s_refined = _local_search(h_init, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf(seq):
    """Compute autocorrelation using FFT."""
    n = len(seq)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    f = np.fft.fft(seq, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return acf

def _psl(seq):
    """Compute peak sidelobe level: max |acf[1:]|."""
    acf = _compute_acf(seq)
    return float(np.max(np.abs(acf[1:])))

def _block_flip(seq, start, length):
    """Flip a block of signs in seq (in-place)."""
    seq[start:start+length] *= -1

def _simulated_annealing(seq, deadline, rng):
    """Simulated annealing with block-flip moves."""
    current_seq = seq.copy()
    current_psl = _psl(current_seq)
    best_seq = current_seq.copy()
    best_psl = current_psl
    
    # Annealing schedule
    t_start = time.time()
    total_time = deadline - t_start
    t_max = 1.0
    t_min = 0.001
    cooling_rate = 0.995
    
    current_temp = t_max
    block_sizes = [4, 8, 16, 32, 64]
    
    while time.time() < deadline:
        if current_temp <= t_min:
            break
        
        # Choose a random block size and position
        size = block_sizes[rng.integers(0, len(block_sizes))]
        start = rng.integers(0, N - size + 1)
        
        # Apply block flip
        _block_flip(current_seq, start, size)
        new_psl = _psl(current_seq)
        
        # Accept or reject
        delta = new_psl - current_psl
        if delta < 0 or rng.random() < np.exp(-delta / current_temp):
            current_psl = new_psl
            if new_psl < best_psl:
                best_psl = new_psl
                best_seq = current_seq.copy()
        else:
            # Revert
            _block_flip(current_seq, start, size)
        
        # Cool down
        current_temp *= cooling_rate
    
    return best_seq, best_psl

def _hill_climb_block(seq, deadline, rng):
    """Hill climbing with block flips."""
    current_seq = seq.copy()
    current_psl = _psl(current_seq)
    best_seq = current_seq.copy()
    best_psl = current_psl
    
    block_sizes = [4, 8, 16, 32]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_psl_local = current_psl
        best_flip = None
        
        # Try several random block flips
        for _ in range(200):
            if time.time() > deadline:
                break
            size = block_sizes[rng.integers(0, len(block_sizes))]
            start = rng.integers(0, N - size + 1)
            
            _block_flip(current_seq, start, size)
            new_psl = _psl(current_seq)
            
            if new_psl < best_psl_local - 1e-12:
                best_psl_local = new_psl
                best_flip = (start, size)
            else:
                _block_flip(current_seq, start, size)
        
        if best_flip is not None:
            start, size = best_flip
            _block_flip(current_seq, start, size)
            current_psl = best_psl_local
            if current_psl < best_psl:
                best_psl = current_psl
                best_seq = current_seq.copy()
            improved = True
    
    return best_seq, best_psl

def _sine_sequence(freq, phase):
    """Generate sign sequence from sine wave zero crossings."""
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    vals = np.sin(2 * np.pi * freq * t + phase)
    signs = np.sign(vals)
    signs[signs == 0] = 1
    return signs.astype(np.float64)

def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound C3."""
    t_start = time.time()
    deadline = t_start + 140  # Safety margin before 150s limit
    
    rng = np.random.default_rng(12345)
    best_seq = None
    best_psl = float('inf')
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Sine-based sequences with various frequencies and phases
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        for phase in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]:
            if time.time() > deadline - 30:
                break
            seq = _sine_sequence(freq, phase)
            candidates.append(seq)
    
    # Random sequences
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        seq = rng.choice([-1.0, 1.0], size=N)
        candidates.append(seq)
    
    # Block-structured sequences (alternating blocks)
    for block_len in [4, 8, 16, 32]:
        for num_blocks in [N // block_len]:
            if time.time() > deadline - 30:
                break
            # Random block pattern
            blocks = rng.choice([-1.0, 1.0], size=num_blocks)
            seq = np.repeat(blocks, block_len)
            candidates.append(seq)
    
    # Evaluate all candidates and keep the best few
    evaluated = []
    for seq in candidates:
        if time.time() > deadline - 20:
            break
        psl = _psl(seq)
        evaluated.append((psl, seq))
    
    evaluated.sort(key=lambda x: x[0])
    
    # Take top 5 candidates
    top_candidates = [seq for _, seq in evaluated[:5]] if evaluated else [np.ones(N)]
    
    # Phase 2: Simulated annealing from each top candidate
    for i, seq in enumerate(top_candidates):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining / len(top_candidates)
        sa_deadline = min(sa_deadline, deadline - 5)
        
        sa_seq, sa_psl = _simulated_annealing(seq, sa_deadline, rng)
        if sa_psl < best_psl:
            best_psl = sa_psl
            best_seq = sa_seq
    
    # Phase 3: Hill climbing from the best
    if best_seq is not None and time.time() < deadline - 3:
        hc_deadline = deadline - 3
        hc_seq, hc_psl = _hill_climb_block(best_seq, hc_deadline, rng)
        if hc_psl < best_psl:
            best_psl = hc_psl
            best_seq = hc_seq
    
    # Phase 4: Final SA polish
    if best_seq is not None and time.time() < deadline - 1:
        sa_seq, sa_psl = _simulated_annealing(best_seq, deadline - 1, rng)
        if sa_psl < best_psl:
            best_psl = sa_psl
            best_seq = sa_seq
    
    if best_seq is None:
        best_seq = np.ones(N)
    
    return best_seq
# EVOLVE-BLOCK-END

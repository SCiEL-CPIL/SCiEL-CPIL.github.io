"""Combined program: PSD-based convex optimization + time-domain refinement for signed step functions."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _construct_zadoff_chu_like(N, root=2, offset=0):
    """Generate a Zadoff-Chu-like sequence with low peak autoconvolution."""
    n = N
    x = np.zeros(n, dtype=float)
    for i in range(n):
        x[i] = np.exp(2j * np.pi * root * i * (i + offset) / (2 * n + 1))
    return x.real

def _construct_mersenne_like(N, seed=12345):
    """Generate a pseudo-random ±1 sequence with good autoconvolution properties."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=N)

def _construct_fourier_based(N):
    """Construct sequence from Fourier basis with optimized phases."""
    n = N
    # Use first few Fourier modes with optimized phases
    x = np.zeros(n)
    for k in range(1, min(20, n // 2)):
        phase = 2 * np.pi * k / n * k  # quadratic phase
        x += np.cos(2 * np.pi * k * np.arange(n) / n + phase)
    x = np.sign(x)
    return x

def _construct_binary_search_peak(N, max_iter=50000, seed=42):
    """
    Binary search over peak autoconvolution value using feasibility check.
    Uses simulated annealing on ±1 sequences.
    """
    rng = np.random.default_rng(seed)
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    # Initialize with various structured sequences
    candidates = [
        np.ones(N),
        _construct_mersenne_like(N, seed=1),
        _construct_mersenne_like(N, seed=2),
        _construct_mersenne_like(N, seed=3),
        _construct_fourier_based(N),
    ]
    
    for c in candidates:
        v = _objective(c)
        if v < best_val:
            best_val = v
            best_x = c.copy()
    
    if _check_deadline():
        return best_x
    
    # Simulated annealing on ±1 sequences
    x = best_x.copy()
    current_val = best_val
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    for iteration in range(max_iter):
        if _check_deadline():
            break
        
        # Pick a random position to flip
        pos = rng.integers(0, N)
        old_val = x[pos]
        x[pos] = -old_val
        
        new_val = _objective(x)
        
        if new_val < current_val or rng.random() < np.exp(-(new_val - current_val) / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[pos] = old_val  # revert
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_x

def _coordinate_descent(N, max_iter=100000, seed=42):
    """
    Coordinate descent: flip one bit at a time, keep if it improves.
    """
    rng = np.random.default_rng(seed)
    x = np.ones(N)
    current_val = _objective(x)
    best_x = x.copy()
    best_val = current_val
    
    # Precompute for efficiency
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Try multiple initializations
    for init_seed in range(10):
        if _check_deadline():
            break
        
        x = _construct_mersenne_like(N, seed=init_seed * 1000 + 7)
        current_val = _objective(x)
        best_x = x.copy()
        best_val = current_val
        
        for iteration in range(max_iter // 10):
            if _check_deadline():
                break
            
            # Pick a random position
            pos = rng.integers(0, N)
            old_val = x[pos]
            x[pos] = -old_val
            
            new_val = _objective(x)
            
            if new_val < current_val:
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_x = x.copy()
            else:
                x[pos] = old_val  # revert
    
    return best_x

def _multi_scale_search(N):
    """
    Multi-scale approach: start with coarse sequence, refine.
    """
    # Coarse: use N//4 length, then upsample
    N_coarse = N // 4
    x_coarse = _construct_mersenne_like(N_coarse, seed=999)
    x = np.repeat(x_coarse, 4)[:N]
    
    best_x = x.copy()
    best_val = _objective(x)
    
    if _check_deadline():
        return best_x
    
    # Refine with local search
    rng = np.random.default_rng(777)
    current_val = best_val
    
    for iteration in range(50000):
        if _check_deadline():
            break
        
        pos = rng.integers(0, N)
        old_val = x[pos]
        x[pos] = -old_val
        
        new_val = _objective(x)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[pos] = old_val
    
    return best_x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Strategy 1: Structured sequences
    structured_seqs = [
        _construct_mersenne_like(N, seed=1),
        _construct_mersenne_like(N, seed=2),
        _construct_mersenne_like(N, seed=3),
        _construct_mersenne_like(N, seed=4),
        _construct_mersenne_like(N, seed=5),
        _construct_fourier_based(N),
    ]
    
    for seq in structured_seqs:
        v = _objective(seq)
        if v < best_val:
            best_val = v
            best_x = seq.copy()
    
    if _check_deadline():
        return best_x
    
    # Strategy 2: Coordinate descent from best structured
    x_cd = _coordinate_descent(N, max_iter=200000, seed=42)
    v_cd = _objective(x_cd)
    if v_cd < best_val:
        best_val = v_cd
        best_x = x_cd.copy()
    
    if _check_deadline():
        return best_x
    
    # Strategy 3: Simulated annealing
    x_sa = _construct_binary_search_peak(N, max_iter=30000, seed=123)
    v_sa = _objective(x_sa)
    if v_sa < best_val:
        best_val = v_sa
        best_x = x_sa.copy()
    
    if _check_deadline():
        return best_x
    
    # Strategy 4: Multi-scale search
    x_ms = _multi_scale_search(N)
    v_ms = _objective(x_ms)
    if v_ms < best_val:
        best_val = v_ms
        best_x = x_ms.copy()
    
    if _check_deadline():
        return best_x
    
    # Strategy 5: Final refinement on best solution
    rng = np.random.default_rng(2024)
    x = best_x.copy()
    current_val = best_val
    
    for iteration in range(50000):
        if _check_deadline():
            break
        
        pos = rng.integers(0, N)
        old_val = x[pos]
        x[pos] = -old_val
        
        new_val = _objective(x)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[pos] = old_val
    
    return best_x
# EVOLVE-BLOCK-END

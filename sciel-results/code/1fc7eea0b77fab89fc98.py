"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autoconv_fast(x):
    """Compute linear autoconvolution using FFT, optimized."""
    n_fft = 1 << (2 * N - 1).bit_length()
    fx = np.fft.rfft(x, n_fft)
    ac_full = np.fft.irfft(fx * fx.conj(), n_fft)
    return ac_full[:N]

def _score_candidate(x):
    """Compute max|autoconv| / (sum x)^2."""
    S = x.sum()
    if abs(S) < 1e-12:
        return np.inf
    ac = _compute_autoconv_fast(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (S * S)

def _local_search_multi_flip(x, time_budget, max_flips=5):
    """Local search with multi-bit flips for faster exploration."""
    x = x.copy()
    current_score = _score_candidate(x)
    best_x = x.copy()
    best_score = current_score
    t_start = time.time()
    no_improve_count = 0
    
    while time.time() < time_budget:
        # Choose number of flips (1-3)
        n_flip = np.random.randint(1, max_flips + 1)
        indices = np.random.choice(N, size=n_flip, replace=False)
        
        x_new = x.copy()
        x_new[indices] *= -1.0
        s_new = _score_candidate(x_new)
        
        if s_new < current_score - 1e-12:
            x = x_new
            current_score = s_new
            no_improve_count = 0
            if s_new < best_score:
                best_score = s_new
                best_x = x.copy()
        else:
            no_improve_count += 1
            if no_improve_count > 200:
                break
    
    return best_x, best_score

def _structured_initialization():
    """Generate diverse initial patterns."""
    patterns = []
    
    # All ones
    patterns.append(np.ones(N))
    
    # Random ±1
    patterns.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Alternating
    patterns.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Block structure: +1 for first half, -1 for second
    x = np.ones(N)
    x[N//2:] = -1.0
    patterns.append(x)
    
    # Multi-frequency: sin-based
    for freq in [1, 3, 7, 13]:
        x = np.sign(np.sin(2 * np.pi * freq * np.arange(N) / N))
        x[x == 0] = 1.0
        patterns.append(x)
    
    # Tent-like: +1 in center, -1 on edges
    x = np.ones(N)
    center = N // 2
    width = N // 4
    x[:center - width] = -1.0
    x[center + width:] = -1.0
    patterns.append(x)
    
    return patterns

def _refine_with_pressure(x, time_budget):
    """Refine using pressure-based sign flips targeting worst lags."""
    x = x.copy()
    current_score = _score_candidate(x)
    t_start = time.time()
    
    for _ in range(2000):
        if time.time() > time_budget:
            break
        
        ac = _compute_autoconv_fast(x)
        worst_lag = np.argmax(np.abs(ac))
        worst_val = ac[worst_lag]
        
        if abs(worst_val) < 1e-10:
            break
        
        k = worst_lag
        if k == 0:
            break
        
        # Compute pressure for the worst lag
        pressure = np.zeros(N)
        for j in range(N - k):
            pressure[j] += -2.0 * x[j] * x[j + k]
        for j in range(k, N):
            pressure[j] += -2.0 * x[j - k] * x[j]
        
        # Find best single flip
        if worst_val > 0:
            candidates = np.where(pressure < -1e-10)[0]
        else:
            candidates = np.where(pressure > 1e-10)[0]
        
        if len(candidates) == 0:
            break
        
        # Try the best candidate
        if worst_val > 0:
            best_idx = candidates[np.argmin(pressure[candidates])]
        else:
            best_idx = candidates[np.argmax(pressure[candidates])]
        
        x_new = x.copy()
        x_new[best_idx] *= -1.0
        s_new = _score_candidate(x_new)
        
        if s_new < current_score - 1e-12:
            x = x_new
            current_score = s_new
    
    return x, current_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin before 150s kill
    
    np.random.seed(42)
    
    # Generate diverse initializations
    initializations = _structured_initialization()
    
    best_score = np.inf
    best_x = np.ones(N)
    
    # Evaluate each initialization
    for x_init in initializations:
        s = _score_candidate(x_init)
        if s < best_score:
            best_score = s
            best_x = x_init.copy()
    
    # Phase 1: Pressure-based refinement
    time_per_phase = (deadline - t_start) / 4.0
    x_ref, s_ref = _refine_with_pressure(best_x.copy(), t_start + time_per_phase)
    if s_ref < best_score:
        best_score = s_ref
        best_x = x_ref.copy()
    
    # Phase 2: Local search with multi-flips
    remaining = deadline - time.time()
    x_ls, s_ls = _local_search_multi_flip(best_x.copy(), time.time() + remaining * 0.6, max_flips=3)
    if s_ls < best_score:
        best_score = s_ls
        best_x = x_ls.copy()
    
    # Phase 3: More pressure refinement
    remaining = deadline - time.time()
    if remaining > 5.0:
        x_ref2, s_ref2 = _refine_with_pressure(best_x.copy(), time.time() + remaining * 0.8)
        if s_ref2 < best_score:
            best_score = s_ref2
            best_x = x_ref2.copy()
    
    # Phase 4: Final local search
    remaining = deadline - time.time()
    if remaining > 2.0:
        x_ls2, s_ls2 = _local_search_multi_flip(best_x.copy(), time.time() + remaining * 0.9, max_flips=2)
        if s_ls2 < best_score:
            best_score = s_ls2
            best_x = x_ls2.copy()
    
    return best_x
# EVOLVE-BLOCK-END

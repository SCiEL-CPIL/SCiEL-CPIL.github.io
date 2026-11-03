import numpy as np
import time
from scipy.fft import fft, ifft

N = 400

def _fft_size(n):
    """Return the next power of 2 >= n."""
    m = 1
    while m < n:
        m *= 2
    return m

def _autoconv_fft(f):
    """Compute full autoconvolution using FFT."""
    n = len(f)
    size = _fft_size(2 * n - 1)
    F = fft(f, size)
    C = ifft(F * F, size)
    return C[:2 * n - 1].real

def _score(f):
    """Compute the score: max |autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_fft(f)
    return np.max(np.abs(c)) / (s * s)

def _soft_max_abs(c, alpha):
    """Smooth approximation of max(|c|) using log-sum-exp."""
    return (1.0 / alpha) * np.log(np.sum(np.exp(alpha * np.abs(c))))

def _objective_smooth(f, alpha):
    """Smooth objective: soft_max(|autoconv|) / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf'), np.zeros(len(f))
    c = _autoconv_fft(f)
    obj = _soft_max_abs(c, alpha) / (s * s)
    # Gradient via finite differences or analytic
    # d/d f_i of max|c|: use chain rule
    # For smooth version, gradient is computable
    # Use finite difference for simplicity in early stages
    return obj, None

def _local_search_perturb(f_init, time_budget):
    """Local search with perturbations using FFT-based scoring."""
    f = f_init.copy()
    s = np.sum(f)
    if abs(s) < 1e-12:
        f = f_init / np.sum(f_init)
        s = np.sum(f)
    else:
        f = f / s
    
    current_score = _score(f)
    best_f = f.copy()
    best_score = current_score
    
    start = time.time()
    step_size = 0.02
    T = 0.01
    
    while time.time() - start < time_budget:
        if time.time() - start > time_budget:
            break
        
        # Generate perturbation
        noise = np.random.normal(0, step_size, N)
        new_f = f + noise
        new_s = np.sum(new_f)
        if abs(new_s) < 1e-6:
            continue
        new_f = new_f / new_s
        
        new_score = _score(new_f)
        
        if new_score < current_score:
            f = new_f
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
        else:
            delta = new_score - current_score
            if np.random.rand() < np.exp(-delta / T):
                f = new_f
                current_score = new_score
        
        # Cool down
        T *= 0.999
        if T < 1e-6:
            T = 1e-6
            step_size *= 0.9
        if step_size < 1e-5:
            break
    
    return best_f, best_score

def _gen_chirp():
    """Generate a chirp-like signal."""
    x = np.arange(N)
    f = np.cos(2 * np.pi * (x**2) / (2 * N))
    return f

def _gen_barker_like():
    """Generate a Barker-like sequence with random signs."""
    f = np.random.choice([-1, 1], size=N)
    return f.astype(float)

def _gen_golden_ratio():
    """Generate signal based on golden ratio phase."""
    x = np.arange(N)
    phi = (1 + np.sqrt(5)) / 2
    f = np.cos(2 * np.pi * phi * x / N)
    return f

def construct():
    start_time = time.time()
    deadline = 130.0  # 20s margin from 150s limit
    
    f_best = None
    score_best = float('inf')
    
    # Multiple initializations with local search
    initializations = []
    
    # 1. Constant
    initializations.append(np.ones(N))
    
    # 2. Chirp
    initializations.append(_gen_chirp())
    
    # 3. Barker-like
    initializations.append(_gen_barker_like())
    
    # 4. Golden ratio
    initializations.append(_gen_golden_ratio())
    
    # 5. Random
    initializations.append(np.random.randn(N))
    
    # 6. Sine
    x = np.arange(N)
    initializations.append(1 + np.cos(2 * np.pi * x / N))
    
    # 7. Two-tone
    initializations.append(np.cos(2 * np.pi * x / N) + 0.5 * np.cos(2 * np.pi * 2 * x / N))
    
    # 8. Sparse random
    f_sparse = np.zeros(N)
    idx = np.random.choice(N, size=N // 4, replace=False)
    f_sparse[idx] = np.random.choice([-1, 1], size=len(idx))
    initializations.append(f_sparse)
    
    # Allocate time to each initialization
    n_inits = len(initializations)
    time_per_init = (deadline - (time.time() - start_time)) / n_inits - 5  # 5s buffer
    
    for i, f_init in enumerate(initializations):
        if time.time() - start_time > deadline - 10:
            break
        
        # Normalize
        s = np.sum(f_init)
        if abs(s) < 1e-12:
            f_init = np.ones(N)
        else:
            f_init = f_init / s
        
        # Run local search
        f, s_score = _local_search_perturb(f_init, max(time_per_init, 1.0))
        
        if s_score < score_best:
            score_best = s_score
            f_best = f.copy()
    
    # If we have time left, do additional refinement on best solution
    if f_best is not None and time.time() - start_time < deadline - 15:
        remaining = deadline - 15 - (time.time() - start_time)
        if remaining > 5:
            f_refined, s_refined = _local_search_perturb(f_best, remaining)
            if s_refined < score_best:
                score_best = s_refined
                f_best = f_refined
    
    if f_best is None:
        f_best = np.ones(N)
    
    return f_best

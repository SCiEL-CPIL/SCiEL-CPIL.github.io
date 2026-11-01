"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 15.0


def _compute_autocorr(x):
    """Compute linear autocorrelation of x using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    return R


def _max_sidelobe(R):
    """Return maximum absolute sidelobe value from autocorrelation R."""
    R0 = R[0]
    if R0 == 0:
        return 0.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return np.max(np.abs(sidelobes)) / R0


def _compute_c3(x):
    """Compute C3 = max|sidelobe|/R0 for signed step function x."""
    R = _compute_autocorr(x)
    return _max_sidelobe(R)


def _m_sequence_like(N, seed=0):
    """Generate an m-sequence-like pseudo-random sequence with good autocorrelation properties."""
    rng = np.random.RandomState(seed)
    # Use a simple LFSR-like approach for better initial properties
    x = np.zeros(N)
    x[0] = 1.0
    for i in range(1, N):
        # Simple feedback: XOR of previous positions
        feedback = x[i-1]
        if i >= 3:
            feedback = np.sign(x[i-1] * x[i-3])
        elif i >= 2:
            feedback = np.sign(x[i-1] * x[i-2])
        x[i] = feedback * (1 if rng.random() > 0.1 else -1)
    return np.sign(x)


def _chaotic_sequence(N, seed=0):
    """Generate a chaotic sequence using logistic map."""
    rng = np.random.RandomState(seed)
    x = np.zeros(N)
    val = rng.uniform(0.1, 0.9)
    for i in range(N):
        val = 4.0 * val * (1 - val)
        x[i] = 1.0 if val > 0.5 else -1.0
    return x


def _spectrally_shaped_random(N, seed=0):
    """Generate random sequence with spectral shaping to reduce sidelobes."""
    rng = np.random.RandomState(seed)
    x = rng.choice([-1.0, 1.0], size=N)
    # Apply spectral shaping: modify in frequency domain to flatten
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    # Smooth the spectrum
    freq = np.arange(len(F))
    # Reduce high-frequency components
    mask = np.ones(len(F))
    cutoff = int(len(F) * 0.7)
    mask[cutoff:] = np.linspace(1.0, 0.3, len(F) - cutoff)
    F_shaped = F * mask
    x_shaped = np.fft.irfft(F_shaped, n=n_fft)[:N]
    return np.sign(x_shaped)


def _local_search_fast(x, max_iter=5000, time_limit=None, batch_size=8):
    """
    Fast local search: evaluate multiple flips in parallel and pick the best.
    """
    start_time = time.time()
    x = np.sign(x).astype(float)
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 100
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Try batch of random flips, pick the best improving one
        candidates = []
        for _ in range(batch_size):
            idx = random.randint(0, N - 1)
            x_new = x.copy()
            x_new[idx] = -x_new[idx]
            candidates.append((idx, x_new))
        
        best_delta = current_c3
        best_new_x = None
        for idx, x_new in candidates:
            new_c3 = _compute_c3(x_new)
            if new_c3 < best_delta:
                best_delta = new_c3
                best_new_x = x_new
        
        if best_new_x is not None:
            x = best_new_x
            current_c3 = best_delta
            no_improve_count = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3


def _simulated_annealing(x, max_iter=30000, T_start=0.3, T_end=0.005, time_limit=None):
    """
    Simulated annealing with adaptive temperature and batch moves.
    """
    start_time = time.time()
    x = np.sign(x).astype(float)
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    T = T_start
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Adaptive temperature
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        # Random single flip
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _compute_c3(x_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-12)):
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3


def _block_flip_search(x, max_iter=1000, time_limit=None, block_sizes=[2, 3, 4, 5]):
    """
    Try flipping contiguous blocks of varying sizes.
    """
    start_time = time.time()
    x = np.sign(x).astype(float)
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 50
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        block_size = random.choice(block_sizes)
        if block_size >= N:
            continue
        start = random.randint(0, N - block_size)
        
        x_new = x.copy()
        x_new[start:start+block_size] = -x_new[start:start+block_size]
        new_c3 = _compute_c3(x_new)
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
            current_c3 = new_c3
            no_improve_count = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3


def _soft_max_sidelobe(x, beta=10.0):
    """
    Compute soft-max approximation of max|sidelobe|/R0 using log-sum-exp.
    This provides a smooth surrogate for the nonsmooth max/abs objective.
    """
    R = _compute_autocorr(x)
    R0 = R[0]
    if R0 <= 1e-12:
        return 1e6
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes) / R0
    
    # Log-sum-exp for numerical stability
    max_val = np.max(abs_sidelobes)
    if max_val < 1e-12:
        return max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * (abs_sidelobes - max_val)))) / beta
    return log_sum_exp


def _objective_and_gradient(x, beta=10.0):
    """
    Compute the soft-max objective and its gradient with respect to x.
    Uses FFT-based computation for efficiency.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 <= 1e-12:
        return 1e6, np.zeros(N)
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes) / R0
    
    # Log-sum-exp for soft-max
    max_val = np.max(abs_sidelobes)
    if max_val < 1e-12:
        return max_val, np.zeros(N)
    
    # Weights for soft-max
    exp_terms = np.exp(beta * (abs_sidelobes - max_val))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of soft-max w.r.t. sidelobes
    # d(soft_max)/d(sidelobe_k) = beta * weights[k] * sign(sidelobe_k)
    signs = np.sign(sidelobes)
    grad_sidelobes = beta * weights * signs / R0
    
    # Now compute gradient w.r.t. x
    # R[k] = sum_j x[j] * x[j-k] for k>=0 (with appropriate indexing)
    # This is complex; use automatic differentiation via finite differences as fallback
    # But for efficiency, we can use the FFT relationship
    
    # Actually, let's use a simpler approach: finite differences for gradient
    grad_x = np.zeros(N)
    eps = 1e-6
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        obj_plus = _soft_max_sidelobe(x_plus, beta=beta)
        
        x_minus = x.copy()
        x_minus[i] -= eps
        obj_minus = _soft_max_sidelobe(x_minus, beta=beta)
        
        grad_x[i] = (obj_plus - obj_minus) / (2 * eps)
    
    objective = max_val + np.log(np.sum(exp_terms)) / beta
    return objective, grad_x


def _gradient_based_search(x, max_iter=500, lr=0.1, beta=10.0, time_limit=None):
    """
    Gradient-based search using smooth approximation of the objective.
    """
    start_time = time.time()
    x = np.sign(x).astype(float)
    current_obj = _soft_max_sidelobe(x, beta=beta)
    best_x = x.copy()
    best_c3 = _compute_c3(x)
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Compute gradient
        obj, grad = _objective_and_gradient(x, beta=beta)
        
        # Gradient descent step
        x_new = x - lr * grad
        
        # Project back to {-1, 1}
        x_new = np.sign(x_new)
        x_new[x_new == 0] = 1
        
        new_obj = _soft_max_sidelobe(x_new, beta=beta)
        new_c3 = _compute_c3(x_new)
        
        if new_obj < obj:
            x = x_new
            if new_c3 < best_c3:
                best_x = x.copy()
                best_c3 = new_c3
    
    return best_x, best_c3


def _high_precision_polish(x, max_iter=20000, time_limit=None):
    """
    Final high-precision polish pass with tighter tolerance and more iterations.
    Uses exhaustive local search over all single flips.
    """
    start_time = time.time()
    x = np.sign(x).astype(float)
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 200
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Try all single flips and pick the best improving one
        best_delta = current_c3
        best_new_x = None
        best_idx = -1
        
        for idx in range(N):
            x_new = x.copy()
            x_new[idx] = -x_new[idx]
            new_c3 = _compute_c3(x_new)
            if new_c3 < best_delta - 1e-14:
                best_delta = new_c3
                best_new_x = x_new
                best_idx = idx
        
        if best_new_x is not None:
            x = best_new_x
            current_c3 = best_delta
            no_improve_count = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3


def _construct():
    """
    Main construction function with multiple diverse strategies.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # Initialize with several diverse starting points
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    inits = [
        np.ones(N),
        _m_sequence_like(N, seed=1),
        _m_sequence_like(N, seed=2),
        _chaotic_sequence(N, seed=3),
        _chaotic_sequence(N, seed=4),
        _spectrally_shaped_random(N, seed=5),
        _spectrally_shaped_random(N, seed=6),
        np.random.choice([-1.0, 1.0], size=N),
        np.random.choice([-1.0, 1.0], size=N),
    ]
    
    # Phase 1: Quick local search from each initialization
    for i, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining < 5:
            break
        
        x_ls, c3_ls = _local_search_fast(x_init, max_iter=3000, time_limit=min(10, remaining))
        if c3_ls < best_c3:
            best_x = x_ls
            best_c3 = c3_ls
    
    # Phase 2: Gradient-based search from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining > 30:
            x_gb, c3_gb = _gradient_based_search(best_x, max_iter=200, lr=0.1, beta=10.0,
                                                  time_limit=min(30, remaining))
            if c3_gb < best_c3:
                best_x = x_gb
                best_c3 = c3_gb
    
    # Phase 3: Simulated annealing from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining > 20:
            x_sa, c3_sa = _simulated_annealing(best_x, max_iter=20000, T_start=0.3, T_end=0.005,
                                                time_limit=min(30, remaining))
            if c3_sa < best_c3:
                best_x = x_sa
                best_c3 = c3_sa
    
    # Phase 4: Block flip search to escape local minima
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining > 10:
            x_bf, c3_bf = _block_flip_search(best_x, max_iter=500, time_limit=min(15, remaining))
            if c3_bf < best_c3:
                best_x = x_bf
                best_c3 = c3_bf
    
    # Phase 5: Another gradient-based search pass
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining > 20:
            x_gb2, c3_gb2 = _gradient_based_search(best_x, max_iter=150, lr=0.05, beta=15.0,
                                                    time_limit=min(20, remaining))
            if c3_gb2 < best_c3:
                best_x = x_gb2
                best_c3 = c3_gb2
    
    # Phase 6: Final intensive local search
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining > 5:
            x_final, c3_final = _local_search_fast(best_x, max_iter=10000, time_limit=remaining)
            if c3_final < best_c3:
                best_x = x_final
                best_c3 = c3_final
    
    # Phase 7: High-precision polish pass
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        if remaining > 10:
            x_polish, c3_polish = _high_precision_polish(best_x, max_iter=15000, time_limit=remaining)
            if c3_polish < best_c3:
                best_x = x_polish
                best_c3 = c3_polish
    
    return best_x


def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    x = np.sign(x)
    x[x == 0] = 1
    return x.astype(int)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autoconv(x):
    """Compute linear autoconvolution using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    fx = np.fft.rfft(x, n_fft)
    ac_full = np.fft.irfft(fx * np.conj(fx), n_fft)
    # ac_full[k] for k=0..N-1 gives c_k = sum_i x_i x_{i+k}
    # By symmetry, c_{-k} = c_k, so max over all lags = max over k=0..N-1
    return ac_full[:N]

def _score_candidate(x):
    """Compute max|autoconv| / (sum x)^2."""
    S = x.sum()
    if abs(S) < 1e-12:
        return np.inf
    ac = _compute_autoconv(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (S * S)

def _spectral_refine(x, time_budget):
    """
    Iterative spectral refinement: 
    1. Compute autoconvolution
    2. Find lag with maximum |c_k|
    3. Try to reduce it by flipping signs in a structured way
    """
    current_score = _score_candidate(x)
    x = x.copy()
    t_start = time.time()
    
    for iteration in range(500):
        if time.time() > time_budget:
            break
        
        ac = _compute_autoconv(x)
        # Find the worst lag
        worst_lag = np.argmax(np.abs(ac))
        worst_val = ac[worst_lag]
        
        if abs(worst_val) < 1e-10:
            break
        
        # Try flipping a subset of signs to reduce the worst lag
        # The contribution to c_k from position i is x_i * x_{i+k}
        # If we flip x_i, it changes c_k by -2*x_i*x_{i+k}
        # We want to flip indices where x_i*x_{i+k} has the same sign as worst_val
        
        # Build a "pressure" vector: for each i, how much does flipping i affect the worst lag
        k = worst_lag
        if k == 0:
            # c_0 = N always for ±1, can't change
            break
        
        # For lag k, c_k = sum_{i=0}^{N-1-k} x_i * x_{i+k}
        # Flipping x_j affects c_k if j is in [0, N-1-k] or j is in [k, N-1]
        # Specifically, flipping x_j changes c_k by:
        # - If j <= N-1-k: term x_j*x_{j+k} becomes -x_j*x_{j+k}, change = -2*x_j*x_{j+k}
        # - If j >= k: term x_{j-k}*x_j becomes -x_{j-k}*x_j, change = -2*x_{j-k}*x_j
        
        pressure = np.zeros(N)
        for j in range(N - k):
            pressure[j] += -2.0 * x[j] * x[j + k]
        for j in range(k, N):
            pressure[j] += -2.0 * x[j - k] * x[j]
        
        # We want to flip signs where pressure has opposite sign to worst_val
        # i.e., where flipping reduces |worst_val|
        if worst_val > 0:
            # Want pressure < 0 (flipping makes c_k more negative, reducing positive peak)
            flip_candidates = np.where(pressure < -1e-10)[0]
        else:
            # Want pressure > 0
            flip_candidates = np.where(pressure > 1e-10)[0]
        
        if len(flip_candidates) == 0:
            break
        
        # Try flipping a small random subset of candidates
        n_flip = min(3, len(flip_candidates))
        chosen = np.random.choice(flip_candidates, size=n_flip, replace=False)
        
        x_new = x.copy()
        x_new[chosen] *= -1.0
        s_new = _score_candidate(x_new)
        
        if s_new < current_score - 1e-12:
            x = x_new
            current_score = s_new
    
    return x, current_score

def _simulated_annealing(x, time_budget):
    """Simulated annealing on sign flips."""
    x = x.copy()
    current_score = _score_candidate(x)
    best_x = x.copy()
    best_score = current_score
    t_start = time.time()
    
    T = 0.1
    T_min = 0.001
    cooling = 0.995
    
    while time.time() < time_budget:
        if T < T_min:
            break
        
        # Random sign flip
        i = np.random.randint(N)
        x[i] *= -1.0
        s_new = _score_candidate(x)
        
        delta = s_new - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = s_new
            if s_new < best_score:
                best_score = s_new
                best_x = x.copy()
        else:
            x[i] *= -1.0  # revert
        
        T *= cooling
    
    return best_x, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin
    
    np.random.seed(42)
    
    # Start with all +1 (maximizes S, gives baseline)
    x = np.ones(N)
    best_score = _score_candidate(x)
    best_x = x.copy()
    
    # Phase 1: Spectral refinement from all-ones
    x_ref, s_ref = _spectral_refine(x.copy(), deadline - 60.0)
    if s_ref < best_score:
        best_score = s_ref
        best_x = x_ref.copy()
    
    # Phase 2: Simulated annealing from best
    x_sa, s_sa = _simulated_annealing(best_x.copy(), deadline - 10.0)
    if s_sa < best_score:
        best_score = s_sa
        best_x = x_sa.copy()
    
    # Phase 3: More spectral refinement from SA result
    if time.time() < deadline - 5.0:
        x_ref2, s_ref2 = _spectral_refine(best_x.copy(), deadline - 5.0)
        if s_ref2 < best_score:
            best_score = s_ref2
            best_x = x_ref2.copy()
    
    return best_x
# EVOLVE-BLOCK-END

import numpy as np
import time
import math

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _fast_acf_delta(s, flip_start, flip_len, A_current):
    """
    Efficiently compute the change in max|ACF| when flipping a block.
    Uses the fact that flipping s[start:start+len] changes the autocorrelation
    by a known delta. For small flips, we can approximate.
    For now, just recompute but with a faster path for single-bit flips.
    """
    # For single-bit flip, we can use a fast delta computation
    if flip_len == 1:
        i = flip_start
        # Flipping s[i] changes ACF[k] for all k
        # A[k] = sum_j s[j]*s[j+k]
        # After flipping s[i]: A[k] changes by -2*s[i]*s[i+k] for k where i+k is valid,
        # and -2*s[i]*s[i-k] for negative lags (but we only track positive)
        # Actually it's more complex. Let's just recompute for safety.
        pass
    
    s_f = s.astype(np.float64)
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _ising_mean_field_init(beta, power_law_exp=1.5, rng=None, n_iter=50):
    """
    Generate initial sign sequence using Ising model with power-law interactions.
    Faster version with fewer iterations.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist_safe = np.where(dist > 0, dist, 1.0)
    J = np.where(dist > 0, dist_safe ** (-power_law_exp), 0.0)
    
    J_max = np.max(J)
    if J_max > 0:
        J = J / J_max
    
    s = rng.choice([-1.0, 1.0], size=N)
    
    for iteration in range(n_iter):
        h = J @ s
        energy_change = -2.0 * s * h
        accept_prob = 1.0 / (1.0 + np.exp(energy_change * beta))
        flip_mask = rng.random(N) < accept_prob
        s[flip_mask] *= -1.0
    
    return s

def _single_bit_local_search(s, deadline, max_passes=20):
    """
    Local search using single-bit flips.
    Much faster per iteration than block flips since we can do incremental updates.
    """
    A = _calc_acf(s)
    current_val = _max_abs_acf_from_array(A)
    
    improved = True
    passes = 0
    local_start = time.time()
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        
        # Try all single-bit flips in random order
        order = np.random.permutation(N)
        for i in order:
            if time.time() > deadline:
                break
            
            s[i] *= -1.0
            A_new = _calc_acf(s)
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val - 1e-12:
                current_val = val_new
                A = A_new
                improved = True
            else:
                s[i] *= -1.0
    
    return s, current_val

def _block_flip_local_search(s, deadline, max_block_sizes=None):
    """
    Local search using block flips of varying sizes.
    """
    if max_block_sizes is None:
        max_block_sizes = [1, 2, 4, 8, 16, 32]
    
    A = _calc_acf(s)
    current_val = _max_abs_acf_from_array(A)
    
    improved = True
    passes = 0
    max_passes = 30
    local_start = time.time()
    
    while improved and passes < max_passes and time.time() - local_start < 8.0 and time.time() < deadline:
        improved = False
        passes += 1
        
        for block_size in max_block_sizes:
            if time.time() > deadline:
                break
            
            n_blocks = N // block_size
            if n_blocks < 1:
                continue
            
            n_attempts = min(30, n_blocks)
            for _ in range(n_attempts):
                if time.time() > deadline:
                    break
                
                block_start = np.random.randint(0, N - block_size + 1)
                block_end = block_start + block_size
                
                s[block_start:block_end] *= -1.0
                A_new = _calc_acf(s)
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val:
                    current_val = val_new
                    A = A_new
                    improved = True
                else:
                    s[block_start:block_end] *= -1.0
    
    return s, current_val

def _simulated_annealing(s, deadline, t_start=0.5, t_end=0.01, cooling=0.95):
    """
    Simulated annealing with block-flip moves.
    """
    A = _calc_acf(s)
    current_val = _max_abs_acf_from_array(A)
    best_val = current_val
    best_s = s.copy()
    
    local_start = time.time()
    t = t_start
    max_block_sizes = [2, 4, 8, 16]
    
    while time.time() < deadline and t > t_end:
        if time.time() - local_start > 10.0:
            break
        
        block_size = int(np.random.choice(max_block_sizes))
        block_start = np.random.randint(0, N - block_size + 1)
        block_end = block_start + block_size
        
        s[block_start:block_end] *= -1.0
        A_new = _calc_acf(s)
        val_new = _max_abs_acf_from_array(A_new)
        
        delta = val_new - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(t, 1e-10)):
            current_val = val_new
            if val_new < best_val:
                best_val = val_new
                best_s = s.copy()
        else:
            s[block_start:block_end] *= -1.0
        
        t *= cooling
    
    return best_s, best_val

def _polish_single_bit(s, deadline, max_passes=50):
    """
    Final high-precision polish: iterate single-bit flips until convergence.
    Use a tighter tolerance and more passes.
    """
    A = _calc_acf(s)
    current_val = _max_abs_acf_from_array(A)
    
    improved = True
    passes = 0
    local_start = time.time()
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        
        order = np.random.permutation(N)
        for i in order:
            if time.time() > deadline:
                break
            
            s[i] *= -1.0
            A_new = _calc_acf(s)
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val - 1e-14:
                current_val = val_new
                A = A_new
                improved = True
            else:
                s[i] *= -1.0
    
    return s, current_val

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Quick Ising initializations (fewer iterations, fewer configs)
    beta_values = [1.0, 2.0, 4.0]
    power_law_exps = [1.5, 2.0]
    
    for beta in beta_values:
        if time.time() > deadline - 80.0:
            break
        
        for alpha in power_law_exps:
            if time.time() > deadline - 80.0:
                break
            
            s = _ising_mean_field_init(beta, alpha, rng, n_iter=30)
            
            # Quick block-flip local search
            s, current_val = _block_flip_local_search(s, deadline - 50.0)
            
            if current_val < best_val:
                best_val = current_val
                best_seq = s.copy()
    
    # Phase 2: Simulated annealing from best
    if time.time() < deadline - 25.0:
        s_anneal, val_anneal = _simulated_annealing(best_seq.copy(), deadline - 15.0)
        if val_anneal < best_val:
            best_val = val_anneal
            best_seq = s_anneal.copy()
    
    # Phase 3: Random restarts with quick local search
    while time.time() < deadline - 10.0:
        s = rng.choice([-1.0, 1.0], size=N)
        s, current_val = _block_flip_local_search(s, deadline - 10.0)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Phase 4: Final polish with single-bit flips
    if time.time() < deadline - 2.0:
        s_polished, val_polished = _polish_single_bit(best_seq.copy(), deadline - 1.0, max_passes=30)
        if val_polished < best_val:
            best_val = val_polished
            best_seq = s_polished.copy()
    
    return best_seq

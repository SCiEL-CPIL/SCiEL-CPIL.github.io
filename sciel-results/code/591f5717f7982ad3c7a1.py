"""Signed step function for third autocorrelation inequality - spin-glass Ising model approach."""
import time
import numpy as np
from scipy.fft import rfft, irfft

N = 400

# EVOLVE-BLOCK-START

def _compute_autocorr(signs):
    """Compute linear autocorrelation of the sign sequence via FFT."""
    n = len(signs)
    size = 2 * n
    f = np.zeros(size)
    f[:n] = signs
    F = rfft(f)
    corr = irfft(F * np.conj(F), size)
    return corr

def _max_abs_autocorr(signs):
    """Return max absolute value of the linear autocorrelation."""
    corr = _compute_autocorr(signs)
    return np.max(np.abs(corr))

def _local_search(signs, time_limit=10.0):
    """Greedy coordinate descent: flip individual signs to reduce max |autocorr|."""
    start = time.time()
    n = len(signs)
    current = signs.copy()
    current_obj = _max_abs_autocorr(current)
    best = current.copy()
    best_obj = current_obj
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(n):
            current[i] = -current[i]
            new_obj = _max_abs_autocorr(current)
            if new_obj < current_obj:
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best = current.copy()
                improved = True
            else:
                current[i] = -current[i]
    
    return best, best_obj

def _block_flips(signs, time_limit=15.0, block_size=8):
    """Simulated annealing with long-range block flips to escape local minima.
    
    Treats the sign sequence as a 1D spin-glass Ising model where the
    autoconvolution energy has competing interactions. Block flips of
    size block_size allow the system to traverse rugged energy landscapes.
    """
    start = time.time()
    n = len(signs)
    current = signs.copy()
    current_obj = _max_abs_autocorr(current)
    best = current.copy()
    best_obj = current_obj
    
    T = 5.0
    T_min = 0.001
    cooling = 0.995
    
    while T > T_min and (time.time() - start) < time_limit:
        # Choose a random block to flip
        max_start = n - block_size
        if max_start <= 0:
            block_start = 0
        else:
            block_start = np.random.randint(max_start)
        
        old_block = current[block_start:block_start + block_size].copy()
        current[block_start:block_start + block_size] = -old_block
        
        new_obj = _max_abs_autocorr(current)
        delta = new_obj - current_obj
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best = current.copy()
        else:
            current[block_start:block_start + block_size] = old_block
        
        T *= cooling
    
    return best, best_obj

def _multi_scale_search(signs, time_limit=30.0):
    """Multi-scale search: combine block flips of different sizes with local polish."""
    start = time.time()
    n = len(signs)
    current = signs.copy()
    current_obj = _max_abs_autocorr(current)
    best = current.copy()
    best_obj = current_obj
    
    block_sizes = [4, 8, 16, 32]
    T = 3.0
    T_min = 0.001
    cooling = 0.997
    
    while T > T_min and (time.time() - start) < time_limit:
        # Alternate between different block sizes
        bs = block_sizes[np.random.randint(len(block_sizes))]
        if bs >= n:
            bs = n // 2
        
        max_start = n - bs
        if max_start <= 0:
            block_start = 0
        else:
            block_start = np.random.randint(max_start)
        
        old_block = current[block_start:block_start + bs].copy()
        current[block_start:block_start + bs] = -old_block
        
        new_obj = _max_abs_autocorr(current)
        delta = new_obj - current_obj
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best = current.copy()
        else:
            current[block_start:block_start + bs] = old_block
        
        T *= cooling
    
    return best, best_obj

def _simulated_annealing_ising(signs, time_limit=40.0):
    """Full Ising-model simulated annealing with mixed move types.
    
    Moves:
    1. Single spin flip (probability 0.3)
    2. Block flip of size 4-16 (probability 0.5)
    3. Block flip of size 32-64 (probability 0.2)
    
    This mimics thermal fluctuations in a spin glass, allowing escape
    from local minima via correlated spin rearrangements.
    """
    start = time.time()
    n = len(signs)
    current = signs.copy()
    current_obj = _max_abs_autocorr(current)
    best = current.copy()
    best_obj = current_obj
    
    T = 4.0
    T_min = 0.0005
    cooling = 0.996
    
    while T > T_min and (time.time() - start) < time_limit:
        r = np.random.rand()
        
        if r < 0.3:
            # Single spin flip
            i = np.random.randint(n)
            old_val = current[i]
            current[i] = -old_val
            new_obj = _max_abs_autocorr(current)
            delta = new_obj - current_obj
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best = current.copy()
            else:
                current[i] = old_val
        
        elif r < 0.8:
            # Medium block flip (size 4-16)
            bs = np.random.choice([4, 8, 16])
            max_start = n - bs
            block_start = np.random.randint(max_start) if max_start > 0 else 0
            old_block = current[block_start:block_start + bs].copy()
            current[block_start:block_start + bs] = -old_block
            new_obj = _max_abs_autocorr(current)
            delta = new_obj - current_obj
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best = current.copy()
            else:
                current[block_start:block_start + bs] = old_block
        
        else:
            # Large block flip (size 32-64)
            bs = np.random.choice([32, 48, 64])
            if bs >= n:
                bs = n // 2
            max_start = n - bs
            block_start = np.random.randint(max_start) if max_start > 0 else 0
            old_block = current[block_start:block_start + bs].copy()
            current[block_start:block_start + bs] = -old_block
            new_obj = _max_abs_autocorr(current)
            delta = new_obj - current_obj
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best = current.copy()
            else:
                current[block_start:block_start + bs] = old_block
        
        T *= cooling
    
    return best, best_obj

def construct():
    """Construct a signed step function with N intervals to minimize the
    autoconvolution upper bound C3 using spin-glass Ising model approach."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    np.random.seed(12345)
    
    # Initialize with random signs
    signs = np.random.choice([-1, 1], size=N)
    
    best_signs = signs.copy()
    best_obj = _max_abs_autocorr(best_signs)
    
    # Phase 1: Quick local search to get a good starting point
    t1 = time.time()
    if t1 < deadline - 20.0:
        signs, obj = _local_search(signs, time_limit=min(10.0, deadline - t1 - 20.0))
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()
    
    # Phase 2: Ising-model simulated annealing with mixed moves
    t2 = time.time()
    remaining = deadline - t2 - 5.0
    if remaining > 10.0:
        signs, obj = _simulated_annealing_ising(best_signs, time_limit=min(40.0, remaining))
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()
    
    # Phase 3: Multi-scale search for further refinement
    t3 = time.time()
    remaining = deadline - t3 - 5.0
    if remaining > 10.0:
        signs, obj = _multi_scale_search(best_signs, time_limit=min(30.0, remaining))
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()
    
    # Phase 4: Final local polish
    t4 = time.time()
    remaining = deadline - t4 - 2.0
    if remaining > 2.0:
        signs, obj = _local_search(best_signs, time_limit=min(8.0, remaining))
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()
    
    # Ensure we return within time limit
    if time.time() > deadline - 1.0:
        return best_signs
    
    return best_signs

# EVOLVE-BLOCK-END

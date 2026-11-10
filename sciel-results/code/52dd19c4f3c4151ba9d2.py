# EVOLVE-BLOCK-START

import numpy as np
import time
import random

_KNOWN_GOOD = [
    [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1],
    [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
    [1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
    [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
    [1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1],
    [1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1],
    [1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1],
]

_N = 24
_SQRT_N = np.sqrt(_N)

def _sup_norm_fast(coeffs):
    """Compute sup norm of polynomial with given coefficients on unit circle."""
    n = len(coeffs)
    # Dense sampling
    npts = 4096
    angles = 2 * np.pi * np.arange(npts) / npts
    zs = np.exp(1j * angles)
    
    # Evaluate polynomial using Horner's method
    vals = np.zeros(npts, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    
    mag = np.abs(vals)
    
    # Refine top peaks with golden section
    top_k = 8
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = float(np.max(mag))
    window = 2 * np.pi / npts
    
    for idx in peak_indices:
        center = angles[idx]
        # Golden section search in a small window
        a = center - window
        b = center + window
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
        
        def f(theta):
            z = np.exp(1j * theta)
            val = 0
            for c in reversed(coeffs):
                val = val * z + c
            return abs(val)
        
        # Golden section
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        
        for _ in range(20):
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = f(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = f(x2)
        
        peak_val = max(f1, f2)
        if peak_val > best_val:
            best_val = peak_val
    
    return best_val

def _score(coeffs):
    """Return the score: sup_norm / sqrt(n)."""
    return _sup_norm_fast(coeffs) / _SQRT_N

def _greedy_local_search(coeffs, max_iter=10000, time_limit=None, start_time=None):
    """
    Greedy local search with single and pair flips.
    Returns the best sequence found.
    """
    if start_time is None:
        start_time = time.time()
    
    current = list(coeffs)
    current_score = _score(current)
    best = list(current)
    best_score = current_score
    
    n = len(current)
    no_improve = 0
    
    for iteration in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        improved = False
        
        # Try single flips
        best_flip_pos = -1
        best_flip_score = current_score
        
        for i in range(n):
            old_val = current[i]
            current[i] = -old_val
            s = _score(current)
            current[i] = old_val
            
            if s < best_flip_score - 1e-12:
                best_flip_score = s
                best_flip_pos = i
        
        if best_flip_pos >= 0:
            current[best_flip_pos] = -current[best_flip_pos]
            current_score = best_flip_score
            improved = True
            no_improve = 0
        else:
            # Try pair flips
            best_pair = (-1, -1)
            best_pair_score = current_score
            
            for i in range(n):
                if time_limit is not None and (time.time() - start_time) > time_limit:
                    break
                for j in range(i + 1, n):
                    old_i = current[i]
                    old_j = current[j]
                    current[i] = -old_i
                    current[j] = -old_j
                    s = _score(current)
                    current[i] = old_i
                    current[j] = old_j
                    
                    if s < best_pair_score - 1e-12:
                        best_pair_score = s
                        best_pair = (i, j)
            
            if best_pair[0] >= 0:
                i, j = best_pair
                current[i] = -current[i]
                current[j] = -current[j]
                current_score = best_pair_score
                improved = True
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 50:
                    break
        
        if current_score < best_score - 1e-12:
            best_score = current_score
            best = list(current)
    
    return best, best_score

def _simulated_annealing(coeffs, time_limit=60, start_time=None):
    """
    Simulated annealing with single and pair flips.
    """
    if start_time is None:
        start_time = time.time()
    
    current = list(coeffs)
    current_score = _score(current)
    best = list(current)
    best_score = current_score
    
    n = len(current)
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    while time.time() - start_time < start_time + time_limit:
        if T < T_min:
            break
        
        # Choose a random move: single flip (70%) or pair flip (30%)
        if random.random() < 0.7:
            i = random.randint(0, n - 1)
            old_i = current[i]
            current[i] = -old_i
            new_score = _score(current)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current[i] = old_i
        else:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            old_i = current[i]
            old_j = current[j]
            current[i] = -old_i
            current[j] = -old_j
            new_score = _score(current)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best = list(current)
            else:
                current[i] = old_i
                current[j] = old_j
        
        T *= cooling
    
    return best, best_score

def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_seq = None
    best_score = float('inf')
    
    # Evaluate known good sequences
    for seq in _KNOWN_GOOD:
        if time.time() > deadline:
            break
        s = _score(seq)
        if s < best_score:
            best_score = s
            best_seq = list(seq)
    
    # Add GRS sequence
    grs = _grs_sequence(_N)
    s = _score(grs)
    if s < best_score:
        best_score = s
        best_seq = list(grs)
    
    # Run local search from best known
    if time.time() < deadline - 30:
        ls_best, ls_score = _greedy_local_search(
            best_seq, max_iter=20000, 
            time_limit=min(30, deadline - time.time()),
            start_time=start_time
        )
        if ls_score < best_score:
            best_score = ls_score
            best_seq = ls_best
    
    # Run simulated annealing with multiple restarts
    remaining = deadline - time.time()
    if remaining > 5:
        n_restarts = max(1, int(remaining / 15))
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random restart
            random_seed = 42 + r
            random.seed(random_seed)
            random_seq = [random.choice([1, -1]) for _ in range(_N)]
            
            sa_best, sa_score = _simulated_annealing(
                random_seq, 
                time_limit=min(10, deadline - time.time() - 5),
                start_time=start_time
            )
            if sa_score < best_score:
                best_score = sa_score
                best_seq = sa_best
    
    # Final polish with local search
    if time.time() < deadline - 5 and best_seq is not None:
        final_best, final_score = _greedy_local_search(
            best_seq, max_iter=5000,
            time_limit=min(5, deadline - time.time()),
            start_time=start_time
        )
        if final_score < best_score:
            best_score = final_score
            best_seq = final_best
    
    return list(best_seq)

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _thue_morse(n):
    t = [0]
    for _ in range(n - 1):
        t += [1 - x for x in t]
    return np.array(t[:n])

def _adjust_sum(h, target_sum):
    """Adjust binary h to have sum exactly target_sum."""
    current_sum = np.sum(h)
    diff = int(target_sum) - int(current_sum)
    if diff > 0:
        zeros = np.where(h == 0)[0]
        for i in range(diff):
            h[zeros[i]] = 1
    elif diff < 0:
        ones = np.where(h == 1)[0]
        for i in range(-diff):
            h[ones[i]] = 0
    return h

def _compute_all_corrs(h):
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _sa_move(h, corrs, i, j, n):
    """
    Update corrs after flipping h[i] 1->0 and h[j] 0->1.
    Returns updated corrs array.
    """
    # We modify h in place by the caller, but we need old h for delta calc?
    # No, delta depends on neighbors.
    # h[i] is 1, h[j] is 0.
    # After flip: h[i]=0, h[j]=1.
    # Delta C_k = - (h[i+k] + h[i-k]) + (h[j+k] + h[j-k])
    # Note: h[i] and h[j] themselves are not in the product for shift k != 0.
    # But for the terms where i or j is the second element, e.g., h[m]h[m+k] where m+k=i.
    # This is covered by h[i-k].
    
    new_corrs = corrs.copy()
    factor = 2.0 / n
    
    for k in range(1, n):
        delta = 0.0
        # Remove contributions from i
        if i + k < n:
            delta -= h[i + k]
        if i - k >= 0:
            delta -= h[i - k]
        # Add contributions from j
        if j + k < n:
            delta += h[j + k]
        if j - k >= 0:
            delta += h[j - k]
        
        new_corrs[k-1] += delta * factor
        
    return new_corrs

def _simulated_annealing(h0, n, target_sum, max_iters=50000, T_start=1.0, T_end=0.01):
    h = h0.copy()
    h = _adjust_sum(h, target_sum)
    corrs = _compute_all_corrs(h)
    current_max = np.max(corrs)
    best_h = h.copy()
    best_max = current_max
    
    ones = list(np.where(h == 1)[0])
    zeros = list(np.where(h == 0)[0])
    
    alpha = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
            
        if not ones or not zeros:
            break
            
        i = random.choice(ones)
        j = random.choice(zeros)
        
        # Perform flip
        h[i] = 0
        h[j] = 1
        
        # Update corrs
        new_corrs = _sa_move(h, corrs, i, j, n)
        new_max = np.max(new_corrs)
        
        delta = new_max - current_max
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_max = new_max
            corrs = new_corrs
            # Update lists
            ones.remove(i)
            ones.append(j)
            zeros.remove(j)
            zeros.append(i)
            
            if new_max < best_max:
                best_max = new_max
                best_h = h.copy()
        else:
            # Revert
            h[i] = 1
            h[j] = 0
            # Revert lists
            ones.remove(j)
            ones.append(i)
            zeros.remove(i)
            zeros.append(j)
            # corrs remains unchanged
            
        T *= alpha
        
    return best_h, best_max

def construct():
    n = 100
    target_sum = n / 2.0
    
    # Initial candidates
    candidates = []
    
    # Thue-Morse
    tm = _thue_morse(n)
    tm = _adjust_sum(tm, target_sum)
    candidates.append(tm)
    
    # Alternating
    alt = np.array([1 if i % 2 == 0 else 0 for i in range(n)])
    alt = _adjust_sum(alt, target_sum)
    candidates.append(alt)
    
    # Random
    random.seed(42)
    for _ in range(5):
        idx = list(range(n))
        random.shuffle(idx)
        r = np.zeros(n)
        r[idx[:int(target_sum)]] = 1
        candidates.append(r)
        
    best_h = None
    best_val = float('inf')
    
    # Run SA on each candidate
    # Time budget: ~140s.
    # Each SA run ~ 50k iters.
    # Let's estimate time.
    
    for h0 in candidates:
        if _time_remaining() < 15.0:
            break
            
        h_opt, val = _simulated_annealing(h0, n, target_sum, max_iters=50000)
        
        if val < best_val:
            best_val = val
            best_h = h_opt
            
    # If time remains, polish the best
    if best_h is not None and _time_remaining() > 10.0:
        h_opt, val = _simulated_annealing(best_h, n, target_sum, max_iters=50000, T_start=0.5, T_end=0.005)
        if val < best_val:
            best_val = val
            best_h = h_opt
            
    return best_h
# EVOLVE-BLOCK-END

import numpy as np
import time
from itertools import combinations

N = 50
START_TIME = time.time()
TIME_LIMIT = 135.0  # Leave margin before 150s kill

def _compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for step function on [-1/4, 1/4].
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    dx = 1.0 / N
    l1 = dx * np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = dx * np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_scaled = conv * dx
    conv_inf = np.max(conv_scaled)
    if conv_inf <= 0:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _find_maximal_sidon_set(n, max_size=12):
    """
    Find a maximal Sidon set (B2-sequence) of size max_size within {0, ..., n-1}.
    A Sidon set has all pairwise sums distinct (or at least, minimal collisions).
    We use a greedy approach with backtracking if needed.
    """
    # Greedy construction of Sidon set
    S = []
    used_sums = set()
    
    for i in range(n):
        if len(S) >= max_size:
            break
        # Check if adding i creates any sum collision
        valid = True
        for j in S:
            s = i + j
            if s in used_sums:
                valid = False
                break
        if valid:
            # Also check i+i
            s2 = 2 * i
            if s2 in used_sums:
                valid = False
        
        if valid:
            # Add all new sums
            for j in S:
                used_sums.add(i + j)
            used_sums.add(2 * i)
            S.append(i)
    
    return S

def _local_search_sidon_support(S, h, n_iters=5000):
    """
    Local search that swaps elements within the Sidon support to minimize max convolution.
    We keep the support fixed but can swap heights between positions.
    Actually, let's allow moving elements in/out of the support too.
    
    Better approach: Start with support S, all heights = 1.
    Then try to improve by:
    1. Swapping which indices are in the support
    2. Adjusting heights
    """
    h = np.array(h, dtype=float)
    best_val = _compute_ratio(h)
    best_h = h.copy()
    
    support = set(S)
    
    for it in range(n_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Random move: pick two indices, one in support, one not
        # Or adjust a height
        move_type = np.random.randint(3)
        
        if move_type == 0:
            # Swap: move element from support to outside, and vice versa
            if len(support) < 2:
                continue
            in_support = list(support)
            out_support = [i for i in range(N) if i not in support]
            if not out_support:
                continue
            i_in = in_support[np.random.randint(len(in_support))]
            i_out = out_support[np.random.randint(len(out_support))]
            
            # Try swapping
            new_h = best_h.copy()
            new_h[i_in] = 0.0
            new_h[i_out] = best_h[i_in]
            new_val = _compute_ratio(new_h)
            
            if new_val > best_val:
                best_val = new_val
                best_h = new_h
                support.remove(i_in)
                support.add(i_out)
        
        elif move_type == 1:
            # Adjust height of an element in support
            if not support:
                continue
            i = list(support)[np.random.randint(len(support))]
            old_h = best_h[i]
            for factor in [0.5, 0.7, 0.85, 1.15, 1.3, 1.5, 0.0, 2.0, 3.0]:
                new_h = best_h.copy()
                new_h[i] = max(0.0, old_h * factor)
                new_val = _compute_ratio(new_h)
                if new_val > best_val:
                    best_val = new_val
                    best_h = new_h
                    break
        
        else:
            # Add a new element to support with small height
            out_support = [i for i in range(N) if i not in support]
            if not out_support:
                continue
            i = out_support[np.random.randint(len(out_support))]
            for height in [0.1, 0.3, 0.5, 0.7, 1.0]:
                new_h = best_h.copy()
                new_h[i] = height
                new_val = _compute_ratio(new_h)
                if new_val > best_val:
                    best_val = new_val
                    best_h = new_h
                    support.add(i)
                    break
    
    return best_h, best_val

def construct():
    np.random.seed(42)
    best_overall = None
    best_overall_val = 0.0
    
    # Try different sizes of Sidon sets
    for sidon_size in [8, 10, 12, 14, 16]:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        # Find a Sidon set
        S = _find_maximal_sidon_set(N, sidon_size)
        
        if len(S) < 4:
            continue
        
        # Initialize with all ones on support
        h = np.zeros(N)
        for i in S:
            h[i] = 1.0
        
        # Local search
        opt_h, opt_val = _local_search_sidon_support(S, h, n_iters=3000)
        
        if opt_val > best_overall_val:
            best_overall_val = opt_val
            best_overall = opt_h.copy()
    
    # Also try a few non-Sidon initializations for comparison
    for init_type in ['uniform', 'gaussian', 'random']:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        if init_type == 'uniform':
            h = np.ones(N)
        elif init_type == 'gaussian':
            x = np.linspace(-5, 5, N)
            h = np.exp(-x**2 / 2)
        else:
            h = np.random.rand(N)
        
        # Simple coordinate ascent
        best_val = _compute_ratio(h)
        best_h = h.copy()
        
        for it in range(2000):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            i = np.random.randint(N)
            old_val = best_h[i]
            for factor in [0.5, 0.7, 0.85, 1.15, 1.3, 1.5, 0.0, 2.0, 3.0]:
                new_h = best_h.copy()
                new_h[i] = max(0.0, old_val * factor)
                new_val = _compute_ratio(new_h)
                if new_val > best_val:
                    best_val = new_val
                    best_h = new_h
                    break
        
        if best_val > best_overall_val:
            best_overall_val = best_val
            best_overall = best_h.copy()
    
    if best_overall is None:
        # Fallback: single spike
        h = [0.0] * N
        h[0] = 1.0
        return h
    
    # Normalize to unit L1
    s = np.sum(best_overall)
    if s > 0:
        result = best_overall / s
    else:
        result = best_overall
    
    result = np.maximum(0, result)
    return [float(x) for x in result]

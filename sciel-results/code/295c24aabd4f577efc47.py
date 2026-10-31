"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconv_max(h):
    """Compute supremum norm of autoconvolution."""
    c = np.correlate(h, h, mode='full')
    return np.max(c)

def _ratio(h):
    """Compute the objective ratio."""
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    max_conv = _autoconv_max(h)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _neg_ratio(h):
    return -_ratio(h)

def _find_sidon_set(n, size, seed=42):
    """Find a Sidon set (B2 sequence) of given size within [0, n)."""
    rng = np.random.default_rng(seed)
    
    # Try to construct a Sidon set greedily
    for attempt in range(100):
        # Start with a random subset and try to build up
        support = set()
        # Add elements in a way that avoids additive collisions
        candidates = list(range(n))
        rng.shuffle(candidates)
        
        for idx in candidates:
            if len(support) >= size:
                break
            # Check if adding idx creates any new additive collisions
            # For a Sidon set, all pairwise sums (with repetition) must be unique
            # Actually for B2, all sums a+b = c+d with {a,b} != {c,d} should not happen
            # For simplicity, let's check if idx can be added
            can_add = True
            new_support = support | {idx}
            s = sorted(new_support)
            sums = {}
            for i in range(len(s)):
                for j in range(i, len(s)):
                    total = s[i] + s[j]
                    if total in sums:
                        # Check if it's the same pair
                        if sums[total] != (i, j):
                            can_add = False
                            break
                    else:
                        sums[total] = (i, j)
                if not can_add:
                    break
            if can_add:
                support.add(idx)
        
        if len(support) >= size:
            return sorted(list(support))[:size]
    
    # Fallback: just use evenly spaced indices
    return list(range(0, n, n // size))[:size]

def _local_search_sidon(h, support, deadline, max_iter=5000):
    """Local search: swap elements in support to improve ratio."""
    best_h = h.copy()
    best_val = _ratio(best_h)
    n = len(h)
    
    # Generate all possible indices
    all_indices = set(range(n))
    current_support = set(support)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try swapping one element in support with one outside
        for i in list(current_support):
            for j in list(all_indices - current_support):
                new_support = (current_support - {i}) | {j}
                new_h = np.zeros(n)
                for idx in new_support:
                    new_h[idx] = best_h[i] if idx == j else best_h[idx]
                # Preserve the weight from the removed element to the new position
                new_h[j] = best_h[i]
                
                val = _ratio(new_h)
                if val > best_val + 1e-12:
                    best_h = new_h
                    best_val = val
                    current_support = new_support
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try perturbing weights slightly
            if len(current_support) > 0:
                idx = list(current_support)[0]
                for delta in [0.1, 0.5, -0.1, -0.5]:
                    new_h = best_h.copy()
                    new_h[idx] = max(0, best_h[idx] + delta)
                    val = _ratio(new_h)
                    if val > best_val + 1e-12:
                        best_h = new_h
                        best_val = val
                        improved = True
                        break
            if not improved:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Start with a single spike (known good: ratio=1)
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_val = _ratio(best_h)
    
    # Try sparse Sidon-like supports with optimized weights
    # A Sidon set minimizes additive collisions in autoconvolution
    for support_size in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 10:
            break
        
        # Find a Sidon set of this size
        support = _find_sidon_set(N, support_size)
        if len(support) < support_size:
            continue
        
        # Initialize with equal weights
        h = np.zeros(N)
        for idx in support:
            h[idx] = 1.0
        
        val = _ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        
        # Local search to optimize
        if time.time() < deadline - 5:
            h_opt, val_opt = _local_search_sidon(h, support, deadline - 5, max_iter=2000)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
    
    # Also try Nelder-Mead from best found
    if time.time() < deadline - 5:
        result = minimize(_neg_ratio, best_h, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < -best_val:
            best_val = -result.fun
            best_h = np.maximum(result.x, 0)
    
    # Final polish
    if time.time() < deadline - 2:
        result = minimize(_neg_ratio, best_h, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < -best_val:
            best_val = -result.fun
            best_h = np.maximum(result.x, 0)
    
    return list(best_h)
# EVOLVE-BLOCK-END

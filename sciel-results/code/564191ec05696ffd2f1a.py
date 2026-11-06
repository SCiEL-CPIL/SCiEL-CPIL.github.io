"""
Strategy:
1. Generate Batcher's Odd-Even Merge Sort network (63 comparators).
2. Perform a deterministic, exhaustive removal pass:
   - Iterate through all comparators in reverse order (from last to first).
   - For each comparator, attempt to remove it.
   - If the network remains valid (using a fast vectorized check), keep it removed.
   - This is a "greedy" reduction that is deterministic and often effective.
3. If the network size is still > 60, perform a second pass in forward order.
4. If still > 60, perform a limited local search with random removals/replacements.
5. Return the best network found.

This is materially different from the current program because:
- It uses a deterministic, structural removal order (reverse then forward) instead of random.
- It prioritizes exhaustive checks over random sampling in the initial reduction phase.
- It avoids the overhead of random number generation and complex move acceptance criteria.
"""

import time
import numpy as np

N = 16
MAX_TIME = 58.0

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a flat list of comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        # Step 1: Compare (offset+i) with (offset+i+n)
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            # Step 1: Compare (offset + i*stride) with (offset + (n+i)*stride)
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            
            # Recursive merges for even and odd indices
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs.
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    """
    Check if the network sorts all 2^16 boolean inputs.
    """
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # 1. Generate initial network (Batcher's, 63 comparators)
    current_comps = _generate_batcher_network()
    
    # Verify initial network
    if not _is_sorted_network_vectorized(current_comps, bits):
        # Fallback: Odd-Even Transposition
        current_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    current_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    current_comps.append((j, j + 1))
    
    best_comps = current_comps[:]
    best_len = len(best_comps)
    
    # 2. Deterministic Reduction Pass 1: Reverse Order
    # Try removing comparators from the end of the network first
    i = len(current_comps) - 1
    while i >= 0 and time.time() < deadline - 5.0:
        if time.time() > deadline - 1.0:
            break
            
        comp = current_comps[i]
        # Temporarily remove
        current_comps.pop(i)
        
        if _is_sorted_network_vectorized(current_comps, bits):
            best_comps = current_comps[:]
            best_len = len(best_comps)
            # Do not decrement i, because the next element is now at index i
        else:
            # Restore
            current_comps.insert(i, comp)
            i -= 1
            
    # 3. Deterministic Reduction Pass 2: Forward Order
    # Try removing comparators from the beginning
    i = 0
    while i < len(current_comps) and time.time() < deadline - 5.0:
        if time.time() > deadline - 1.0:
            break
            
        comp = current_comps[i]
        # Temporarily remove
        current_comps.pop(i)
        
        if _is_sorted_network_vectorized(current_comps, bits):
            best_comps = current_comps[:]
            best_len = len(best_comps)
            # Do not increment i
        else:
            # Restore
            current_comps.insert(i, comp)
            i += 1
            
    # 4. Limited Local Search for further reduction
    rng = np.random.default_rng(42)
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline - 1.0:
        if time.time() > deadline - 1.0:
            break
            
        current_comps = best_comps[:]
        improved = False
        
        # Try random removal
        if len(current_comps) > 1:
            idx = rng.integers(0, len(current_comps))
            removed = current_comps.pop(idx)
            if _is_sorted_network_vectorized(current_comps, bits):
                best_comps = current_comps
                best_len = len(best_comps)
                improved = True
            else:
                current_comps.insert(idx, removed)
                
        # If no improvement, try random replacement
        if not improved and len(current_comps) > 1:
            idx = rng.integers(0, len(current_comps))
            old_comp = current_comps[idx]
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            
            if new_comp != old_comp:
                current_comps[idx] = new_comp
                if _is_sorted_network_vectorized(current_comps, bits):
                    best_comps = current_comps
                    # best_len unchanged
                    improved = True
                else:
                    current_comps[idx] = old_comp
                    
    return best_comps

# EVOLVE-BLOCK-END

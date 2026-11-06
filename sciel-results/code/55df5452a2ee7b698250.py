"""
Strategy:
Simulated Annealing over the space of comparator networks.
Unlike the previous "validity-preserving" local search, this approach allows
transitioning through invalid states (where V > 0) or larger states (C > current)
if the temperature is high enough. This helps escape local minima where no single
move preserves validity but a sequence of moves could.

Objective Function: F = C + lambda * V
where C is the number of comparators, and V is the number of Boolean inputs 
(0..2^16-1) that are NOT sorted by the network.
We minimize F. A valid network has V=0, so F=C.

Moves:
1. Remove a random comparator.
2. Add a random comparator.
3. Replace a random comparator with a different one.
4. Swap the position of two comparators.

Acceptance:
Metropolis criterion with a cooling schedule.
"""

import time
import numpy as np
import math

N = 16
MAX_TIME = 58.0
LAMBDA = 1000.0  # Heavy penalty for violations to prioritize validity

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a flat list of comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
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

def _evaluate_network(comps, bits):
    """
    Evaluate the network: returns (C, V, F).
    C: number of comparators.
    V: number of inputs not sorted.
    F: objective function value.
    """
    C = len(comps)
    if C == 0:
        # If no comparators, only sorted if input is already sorted.
        # But we need to check V.
        # Actually, just run the vectorized check.
        pass
        
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    # An input is unsorted if any diff is negative.
    # Since bits are 0/1, diffs can be -1, 0, 1.
    # Unsorted if min(diff) < 0.
    unsorted_mask = np.any(diffs < 0, axis=1)
    V = int(np.sum(unsorted_mask))
    
    F = C + LAMBDA * V
    return C, V, F

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # 1. Generate initial network (Batcher's, 63 comparators)
    initial_comps = _generate_batcher_network()
    
    # Verify initial network and get initial score
    C, V, F = _evaluate_network(initial_comps, bits)
    
    best_comps = initial_comps[:]
    best_C, best_V, best_F = C, V, F
    
    # Current state
    curr_comps = initial_comps[:]
    curr_C, curr_V, curr_F = C, V, F
    
    rng = np.random.default_rng(42)
    
    # Pre-compute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    # We don't know the number of steps in advance, so we use a time-based cooling.
    # Or we can estimate steps. 
    # Let's say we can do ~5000 evaluations in 58s (rough estimate, depends on speed).
    # Let's use a cooling factor per step.
    # alpha = (T_end / T_start) ** (1 / max_steps)
    # But max_steps is unknown. 
    # Alternative: Cool based on time elapsed.
    # T(t) = T_start * exp(-k * t)
    
    k = 0.05 # Cooling constant
    
    while time.time() < deadline:
        if time.time() > deadline - 1.0:
            break
            
        t_elapsed = time.time() - start_time
        T = T_start * math.exp(-k * t_elapsed)
        
        # Generate a move
        move_type = rng.integers(0, 4)
        new_comps = curr_comps[:]
        
        if move_type == 0: # Remove
            if len(curr_comps) > 1:
                idx = rng.integers(0, len(curr_comps))
                new_comps.pop(idx)
            else:
                continue # Can't remove if empty or 1 (keep at least 1 for safety? No, 0 is allowed but bad)
                
        elif move_type == 1: # Add
            # Avoid adding duplicates if possible? 
            # If we add a duplicate, it's a waste of C but doesn't change sorting.
            # Let's pick a random comparator.
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            idx = rng.integers(0, len(new_comps) + 1)
            new_comps.insert(idx, new_comp)
            
        elif move_type == 2: # Replace
            if len(curr_comps) > 0:
                idx = rng.integers(0, len(curr_comps))
                new_comp = all_comps[rng.integers(0, len(all_comps))]
                new_comps[idx] = new_comp
            else:
                continue
                
        elif move_type == 3: # Swap
            if len(curr_comps) > 1:
                idx1 = rng.integers(0, len(curr_comps))
                idx2 = rng.integers(0, len(curr_comps))
                if idx1 != idx2:
                    new_comps[idx1], new_comps[idx2] = new_comps[idx2], new_comps[idx1]
                else:
                    continue
            else:
                continue
                
        # Evaluate new state
        new_C, new_V, new_F = _evaluate_network(new_comps, bits)
        
        delta_F = new_F - curr_F
        
        # Acceptance
        if delta_F < 0:
            accept = True
        else:
            # Avoid overflow in exp
            if T > 0:
                prob = math.exp(-delta_F / T)
            else:
                prob = 0.0
            accept = rng.random() < prob
            
        if accept:
            curr_comps = new_comps
            curr_C, curr_V, curr_F = new_C, new_V, new_F
            
            # Update best
            if curr_F < best_F:
                best_comps = curr_comps[:]
                best_C, best_V, best_F = curr_C, curr_V, curr_F
                
            # If we found a valid network (V=0), we can potentially lower T faster?
            # Or just let the schedule run.
            
    return best_comps

# EVOLVE-BLOCK-END

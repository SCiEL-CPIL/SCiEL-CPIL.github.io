"""Size-efficient sorting networks on 18 wires.
Uses a vectorized NumPy approach for fast verification and an aggressive
local search to reduce comparator count while maintaining correctness.
"""

import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N  # 262144
DEADLINE = 85.0  # seconds (5s safety margin before 90s kill)


def _precompute_sorted_mask():
    """Precompute a boolean array: True if the state index is a sorted configuration."""
    mask = np.zeros(NUM_BOOLS, dtype=np.bool_)
    for k in range(N + 1):
        # k ones in the top k positions (wires N-k through N-1)
        val = ((1 << k) - 1) << (N - k)
        mask[val] = True
    return mask


_SORTED_MASK = _precompute_sorted_mask()


def _odd_even_transposition_network():
    """Generate the odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _verify_network(comparators):
    """Verify that the network sorts all 2^18 Boolean inputs.
    Returns True if valid, False otherwise.
    """
    current = np.arange(NUM_BOOLS, dtype=np.int32)
    for i, j in comparators:
        bit_i = (current >> i) & 1
        bit_j = (current >> j) & 1
        # Where bit_i=1 and bit_j=0, swap bits i and j
        mask_swap = (bit_i == 1) & (bit_j == 0)
        if np.any(mask_swap):
            # Swap: clear bit_i, set bit_j for those where swap happens
            clear_i = ~(1 << i) & 0xFFFF
            set_j = (1 << j)
            current = np.where(mask_swap, (current & clear_i) | set_j, current)
    
    # Check if all are sorted
    return np.all(_SORTED_MASK[current])


def _random_comparator():
    """Generate a random comparator (i, j) with i < j."""
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    return (i, j)


def _greedy_add_comparators(target_size):
    """Start with an empty network and add comparators until we reach target_size,
    choosing the comparator that reduces the number of unsorted states the most."""
    # Start with identity (no comparators) - all states are "unsorted" except the N+1 sorted ones
    # We track which states are currently sorted
    # For efficiency, track the current state of each input
    current = np.arange(NUM_BOOLS, dtype=np.int32)
    comparators = []
    
    # Check initial sorted count
    sorted_count = np.sum(_SORTED_MASK[current])
    print(f"Initial sorted count: {sorted_count}/{NUM_BOOLS}")
    
    while len(comparators) < target_size:
        best_comp = None
        best_score = -1
        
        # Try all possible comparators
        for i in range(N - 1):
            for j in range(i + 1, N):
                # Test adding this comparator
                trial_current = current.copy()
                bit_i = (trial_current >> i) & 1
                bit_j = (trial_current >> j) & 1
                mask_swap = (bit_i == 1) & (bit_j == 0)
                if np.any(mask_swap):
                    clear_i = ~(1 << i) & 0xFFFF
                    set_j = (1 << j)
                    trial_current = np.where(mask_swap, (trial_current & clear_i) | set_j, trial_current)
                
                new_sorted_count = np.sum(_SORTED_MASK[trial_current])
                score = new_sorted_count - sorted_count
                
                if score > best_score:
                    best_score = score
                    best_comp = (i, j)
        
        if best_comp is None or best_score <= 0:
            break
        
        # Apply the best comparator
        i, j = best_comp
        bit_i = (current >> i) & 1
        bit_j = (current >> j) & 1
        mask_swap = (bit_i == 1) & (bit_j == 0)
        if np.any(mask_swap):
            clear_i = ~(1 << i) & 0xFFFF
            set_j = (1 << j)
            current = np.where(mask_swap, (current & clear_i) | set_j, current)
        
        comparators.append(best_comp)
        sorted_count = np.sum(_SORTED_MASK[current])
        print(f"Added {best_comp}, size={len(comparators)}, sorted={sorted_count}/{NUM_BOOLS}")
        
        if sorted_count == NUM_BOOLS:
            break
    
    return comparators


def _local_search(initial_comparators, max_time=75.0):
    """Local search to reduce comparator count while maintaining validity."""
    start_time = time.time()
    best_comparators = list(initial_comparators)
    best_len = len(best_comparators)
    
    # Phase 1: Try removing each comparator (from end to start for efficiency)
    current = list(best_comparators)
    changed = True
    while changed and time.time() - start_time < max_time:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() - start_time > max_time:
                break
            if len(current) <= 68:  # Can't go below lower bound
                break
            
            # Try removing comparator at idx
            trial = current[:idx] + current[idx+1:]
            if _verify_network(trial):
                current = trial
                if len(current) < best_len:
                    best_len = len(current)
                    best_comparators = list(current)
                changed = True
                break  # Restart the loop since indices changed
    
    # Phase 2: Try moving comparators to enable further removals
    # Try swapping adjacent comparators and see if it enables removals
    for restart in range(10):
        if time.time() - start_time > max_time:
            break
        
        curr = list(best_comparators)
        
        # Perturb: swap two random comparators
        if len(curr) > 1:
            i = random.randint(0, len(curr) - 2)
            j = random.randint(0, len(curr) - 2)
            if i != j:
                curr[i], curr[j] = curr[j], curr[i]
        
        # Try removal on perturbed network
        changed = True
        while changed and time.time() - start_time < max_time:
            changed = False
            for idx in range(len(curr) - 1, -1, -1):
                if time.time() - start_time > max_time:
                    break
                trial = curr[:idx] + curr[idx+1:]
                if _verify_network(trial):
                    curr = trial
                    changed = True
                    break
        
        if len(curr) < best_len:
            best_len = len(curr)
            best_comparators = list(curr)
    
    return best_comparators


def construct():
    """Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    
    # Start with odd-even transposition network (known good, 153 comparators)
    comparators = _odd_even_transposition_network()
    
    # Verify baseline
    if not _verify_network(comparators):
        # Should not happen, but fallback
        return comparators
    
    # Try local search to reduce size
    remaining_time = DEADLINE - (time.time() - start_time) - 5
    if remaining_time > 10:
        comparators = _local_search(comparators, max_time=remaining_time)
    
    # Final verification
    if not _verify_network(comparators):
        # Fallback to known good network
        comparators = _odd_even_transposition_network()
    
    return comparators

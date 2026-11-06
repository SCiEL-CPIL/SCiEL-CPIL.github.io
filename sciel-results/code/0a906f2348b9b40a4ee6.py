import numpy as np
import time
import random

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

# ---------- Batcher Bitonic Sort Network ----------
def _bitonic_network(n):
    """Generate Batcher bitonic sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    half = n // 2
    # Sort first half ascending, second half descending, then merge
    left = _bitonic_network(half)
    right = _bitonic_network(half)
    # Shift right comparators
    right = [(i + half, j + half) for (i, j) in right]
    # Merge: odd-even merge for bitonic
    merge = _bitonic_merge(n)
    return left + right + merge

def _bitonic_merge(n):
    """Bitonic merge for n wires (n power of 2). Assumes input is bitonic."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # First do cross comparisons
    cross = [(i, i + half) for i in range(half)]
    # Then recursively merge both halves
    left_merge = _bitonic_merge(half)
    right_merge = _bitonic_merge(half)
    right_merge = [(i + half, j + half) for (i, j) in right_merge]
    return cross + left_merge + right_merge

def _make_bitonic_network():
    """Build full bitonic sort: split, sort halves opposite, merge."""
    half = N // 2
    left = _bitonic_network(half)
    right = _bitonic_network(half)
    right = [(i + half, j + half) for (i, j) in right]
    merge = _bitonic_merge(N)
    return left + right + merge

# ---------- Verification ----------
def _verify(comparators):
    """Check if network sorts all 2^16 zero-one inputs. Returns True if valid."""
    # Precomputed popcount table for 16-bit values
    if not hasattr(_verify, '_popcount'):
        _verify._popcount = np.array([bin(v).count('1') for v in range(65536)], dtype=np.uint8)
    
    all_inputs = np.arange(65536, dtype=np.uint32)
    
    for (i, j) in comparators:
        bit_i = (all_inputs >> i) & 1
        bit_j = (all_inputs >> j) & 1
        # min goes to i, max goes to j
        new_i = np.minimum(bit_i, bit_j)
        new_j = np.maximum(bit_i, bit_j)
        
        mask_i = np.uint32(1 << i)
        mask_j = np.uint32(1 << j)
        
        # Clear bits i and j
        all_inputs = all_inputs & (~mask_i & ~mask_j)
        # Set new bits
        all_inputs = all_inputs | (new_i.astype(np.uint32) * mask_i) | (new_j.astype(np.uint32) * mask_j)
        
    ones = _verify._popcount[all_inputs]
    
    # Expected sorted form: if there are k ones, the result must have the top k bits set.
    # i.e., value == ((1 << k) - 1) << (16 - k)
    
    targets = np.zeros(17, dtype=np.uint32)
    for k in range(17):
        if k == 0:
            targets[k] = 0
        elif k == 16:
            targets[k] = 65535
        else:
            targets[k] = ((1 << k) - 1) << (16 - k)
            
    expected = targets[ones]
    
    return np.array_equal(all_inputs, expected)

# ---------- Local Search ----------
def _try_remove(comparators, idx):
    """Try removing comparator at index idx. Return new list if still valid, else None."""
    new_comp = comparators[:idx] + comparators[idx+1:]
    if _verify(new_comp):
        return new_comp
    return None

def _try_swap(comparators, idx1, idx2):
    """Try swapping comparators at idx1 and idx2."""
    new_comp = list(comparators)
    new_comp[idx1], new_comp[idx2] = new_comp[idx2], new_comp[idx1]
    if _verify(new_comp):
        return new_comp
    return None

def _try_replace(comparators, idx, new_comp):
    """Try replacing comparator at idx with new_comp."""
    new_comp_list = list(comparators)
    new_comp_list[idx] = new_comp
    if _verify(new_comp_list):
        return new_comp_list
    return None

def _local_search(comparators, time_limit):
    """Aggressive local search to reduce comparator count."""
    best = list(comparators)
    start_time = time.time()
    
    # All possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # First pass: try to remove each comparator
    i = 0
    while i < len(best):
        if time.time() - start_time > time_limit:
            break
        new_best = _try_remove(best, i)
        if new_best is not None:
            best = new_best
            # Don't increment i, re-check same position
        else:
            i += 1
    
    # Second pass: try swapping adjacent comparators to enable further removals
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(len(best) - 1):
            if time.time() - start_time > time_limit:
                break
            new_best = _try_swap(best, i, i+1)
            if new_best is not None and new_best != best:
                best = new_best
                improved = True
                break  # Restart scan
    
    # Third pass: try replacing comparators with different ones to enable removals
    # This is more expensive, so do it selectively
    while time.time() - start_time < time_limit:
        # Pick a random position and try a random replacement
        idx = random.randint(0, len(best) - 1)
        new_pair = random.choice(all_pairs)
        if new_pair == best[idx]:
            continue
        new_best = _try_replace(best, idx, new_pair)
        if new_best is not None:
            best = new_best
            # After replacement, try to remove some comparators
            j = 0
            while j < len(best) and time.time() - start_time < time_limit:
                nb = _try_remove(best, j)
                if nb is not None:
                    best = nb
                else:
                    j += 1
    return best

def _anneal(comparators, time_limit):
    """Simulated annealing-style search."""
    best = list(comparators)
    current = list(comparators)
    start_time = time.time()
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    temperature = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        temperature *= cooling
        if temperature < 0.01:
            break
        
        # Random move: remove, swap, or replace
        move_type = random.random()
        if move_type < 0.5 and len(current) > 1:
            # Try to remove a random comparator
            idx = random.randint(0, len(current) - 1)
            new_comp = current[:idx] + current[idx+1:]
            cost = -1  # reduction in length is good
        elif move_type < 0.8:
            # Try to swap two random comparators
            if len(current) < 2:
                continue
            idx1, idx2 = random.sample(range(len(current)), 2)
            new_comp = list(current)
            new_comp[idx1], new_comp[idx2] = new_comp[idx2], new_comp[idx1]
            cost = 0  # same length
        else:
            # Try to replace a random comparator
            idx = random.randint(0, len(current) - 1)
            new_pair = random.choice(all_pairs)
            if new_pair == current[idx]:
                continue
            new_comp = list(current)
            new_comp[idx] = new_pair
            cost = 0  # same length
        
        if _verify(new_comp):
            # Accept if valid
            current = new_comp
            if len(current) < len(best):
                best = current
        else:
            # Reject invalid moves
            pass
    
    return best

def construct():
    deadline = time.time() + DEADLINE
    
    # Start with bitonic sort (63 comparators for N=16)
    network = _make_bitonic_network()
    
    if not _verify(network):
        # Fallback to odd-even transposition (120 comparators)
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    best_network = network[:]
    best_len = len(network)
    
    # Phase 1: Local search
    remaining = deadline - time.time()
    if remaining > 5:
        network = _local_search(network, min(remaining - 5, 40))
        if len(network) < best_len:
            best_network = network[:]
            best_len = len(network)
    
    # Phase 2: Annealing
    remaining = deadline - time.time()
    if remaining > 5:
        network = _anneal(network, min(remaining - 5, 25))
        if len(network) < best_len:
            best_network = network[:]
            best_len = len(network)
    
    # Final verification
    if not _verify(best_network):
        # Should not happen, but safety fallback
        best_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    return best_network

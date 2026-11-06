"""Size-efficient sorting networks on 16 wires.
Uses Batcher bitonic sort (63 comparators) as seed, then aggressive local search
to reduce comparator count toward the known optimum of 60.
"""
import numpy as np
import time
import random
from collections import deque

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
    # We can create this once globally to save time, but for simplicity and thread-safety 
    # in case of parallel evals (though unlikely here), we'll make it static.
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
    # We can construct the expected array vectorially.
    # k ranges from 0 to 16.
    # For k=0, expected is 0.
    # For k=16, expected is 65535.
    # For k in 1..15, expected is ((1<<k)-1) << (16-k).
    
    # Create an array of expected values for each possible count of ones (0..16)
    # But we have 65536 entries, each with a specific count.
    # It's faster to check: 
    # Is all_inputs equal to the sorted version of itself?
    # Actually, the condition is:
    # For each input x, let k = popcount(x).
    # The sorted output must be y such that popcount(y)=k and y has the highest bits set.
    # y = ((1 << k) - 1) << (16 - k)
    
    # Let's compute the target value for each input based on its popcount.
    # We can't easily vectorize "shift by k" where k varies per element in a simple way 
    # without a loop or fancy indexing, but 65536 is small.
    # However, we can precompute the target for each possible popcount value 0..16.
    
    targets = np.zeros(17, dtype=np.uint32)
    for k in range(17):
        if k == 0:
            targets[k] = 0
        elif k == 16:
            targets[k] = 65535
        else:
            targets[k] = ((1 << k) - 1) << (16 - k)
            
    # Map each input's popcount to its target value
    expected = targets[ones]
    
    return np.array_equal(all_inputs, expected)

def _verify_fast(comparators):
    """Faster verification using the property that we only need to check
    inputs that are 'critical'. Actually for 16 wires, 65536 checks is fast enough
    with numpy. Let's just use _verify."""
    return _verify(comparators)

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
        
        # Random move: add, remove, swap, or replace
        move_type = random.choice(['remove', 'swap', 'replace', 'add'])
        
        new_comp = list(current)
        valid = False
        
        if move_type == 'remove' and len(new_comp) > 0:
            idx = random.randint(0, len(new_comp) - 1)
            new_comp.pop(idx)
            valid = _verify(new_comp)
        
        elif move_type == 'swap' and len(new_comp) >= 2:
            i, j = random.sample(range(len(new_comp)), 2)
            new_comp[i], new_comp[j] = new_comp[j], new_comp[i]
            valid = _verify(new_comp)
        
        elif move_type == 'replace' and len(new_comp) > 0:
            idx = random.randint(0, len(new_comp) - 1)
            new_comp[idx] = random.choice(all_pairs)
            valid = _verify(new_comp)
        
        elif move_type == 'add' and len(new_comp) < 100:
            new_comp.append(random.choice(all_pairs))
            valid = _verify(new_comp)
        
        if not valid:
            continue
        
        # Accept if better or with probability exp(-delta/T)
        delta = len(new_comp) - len(current)
        if delta < 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
            current = new_comp
            if len(current) < len(best):
                best = list(current)
    
    return best

# ---------- Main construct function ----------
def construct():
    start_time = time.time()
    # Safety margin: finish by 55s to avoid 60s kill
    deadline = start_time + 55.0
    
    # Start with Batcher bitonic sort network (63 comparators)
    network = _make_bitonic_network()
    
    # Verify it's correct
    if not _verify(network):
        # Fallback to odd-even transposition
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Phase 1: Greedy removal (fast with vectorized verify)
    network = _local_search(network, deadline - time.time() - 10)
    
    # Phase 2: More aggressive search with restarts
    best_network = network
    best_len = len(network)
    
    while time.time() < deadline - 5:
        # Restart from best with random perturbation
        perturbed = list(best_network)
        # Remove a random comparator
        if len(perturbed) > 0:
            idx = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
        
        # Local search from perturbed state
        remaining_time = deadline - time.time() - 5
        if remaining_time < 1:
            break
        
        candidate = _local_search(perturbed, remaining_time)
        if len(candidate) < best_len:
            best_network = candidate
            best_len = len(candidate)
    
    # Final: try one more round of removals on the best
    if time.time() < deadline:
        best_network = _local_search(best_network, deadline - time.time())
    
    # Verify final result
    if not _verify(best_network):
        # Should not happen, but fallback
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return best_network

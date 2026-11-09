"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 18
MAX_COMPARATORS = 4096
TIME_LIMIT = 85.0  # Safety margin before 90s kill

# EVOLVE-BLOCK-START
def _verify_sorts_fast(nets, n_wires, timeout=None):
    """Verify that a sorting network sorts all 2^n Boolean inputs using vectorized operations."""
    n = n_wires
    total = 1 << n
    start_time = time.time()
    
    # Create array of all 2^n inputs
    inputs = np.arange(total, dtype=np.uint32)
    
    # Apply each comparator
    for (i, j) in nets:
        bi = (inputs >> i) & 1
        bj = (inputs >> j) & 1
        # If bi > bj, we need to swap: clear bit i, set bit j
        mask = (bi > bj).astype(np.uint32)
        inputs = (inputs & ~(mask << i)) | (mask << j)
        
        if timeout and time.time() - start_time > timeout:
            return None  # Timeout
    
    # Check if sorted: all 0s come before all 1s
    # x is sorted iff the complement y = ~x & ((1<<n)-1) is of form (1<<k)-1
    # which means y & (y+1) == 0
    y = (~inputs) & ((1 << n) - 1)
    result = (y & (y + 1)) == 0
    return bool(np.all(result))

def _depth(nets, n_wires):
    """Compute parallel depth of a sorting network."""
    depth = [0] * n_wires
    for (i, j) in nets:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network."""
    nets = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                nets.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                nets.append((i, i + 1))
    return nets

def _bitonic_sort(n):
    """Generate a bitonic sorting network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _bitonic_sort(half)
    right = [((i + half), (j + half)) for (i, j) in left]
    merge = _bitonic_merge(half)
    return left + right + merge

def _bitonic_merge(n):
    """Bitonic merge for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    half = n // 2
    comps = [(i, i + half) for i in range(half)]
    left = _bitonic_merge(half)
    right = [((i + half), (j + half)) for (i, j) in left]
    return comps + left + right

def _generate_bitonic_for_n(n):
    """Generate bitonic sort network for arbitrary n by padding to next power of 2."""
    if n <= 1:
        return []
    p2 = 1
    while p2 < n:
        p2 *= 2
    full_net = _bitonic_sort(p2)
    filtered = [(i, j) for (i, j) in full_net if i < n and j < n]
    return filtered

def _akamai_network(n):
    """
    Generate a good starting network using a combination of approaches.
    For n=18, we can use a modified bitonic or other known constructions.
    """
    base = _generate_bitonic_for_n(16)
    extra = []
    for i in range(16):
        extra.append((i, 16))
    for i in range(17):
        extra.append((i, 17))
    return base + extra

def _greedy_construct(n_wires, time_budget):
    """
    Greedy constructive search guided by the 0-1 principle.
    We build the network layer by layer. At each step, we pick the comparator
    that reduces the number of unsorted Boolean inputs the most.
    """
    start_time = time.time()
    n = n_wires
    total = 1 << n
    
    # State: bitmask of which inputs are currently unsorted
    # An input x is "sorted" if it has the form 0...01...1 (zeros then ones)
    # i.e., y = ~x & ((1<<n)-1) satisfies y & (y+1) == 0
    # We track which inputs are NOT sorted.
    
    # Precompute sorted check for all inputs
    # x is sorted iff y = (~x) & ((1<<n)-1) has y & (y+1) == 0
    all_inputs = np.arange(total, dtype=np.uint32)
    y_all = (~all_inputs) & ((1 << n) - 1)
    is_sorted_mask = (y_all & (y_all + 1)) == 0  # True where input is already sorted
    
    # Current state: which inputs are unsorted
    # Start: all inputs except those already sorted (0 and all-ones)
    # Actually, we need to track the *current* value of each input as we apply comparators.
    # This is expensive to track for all 2^n inputs simultaneously in a greedy manner
    # because we need to know which specific inputs are unsorted at each step.
    
    # Alternative approach:
    # We maintain an array `current` of size 2^n, where current[x] is the current value
    # of input x after applying the comparators so far.
    # We also maintain a set (or bitmask) of indices x where current[x] is not sorted.
    
    # This is memory intensive: 2^18 * 4 bytes = 1MB, which is fine.
    # But updating it for each comparator is O(2^n) per comparator.
    # If we do ~100 comparators, that's 100 * 262144 = 26M operations, which is fast in NumPy.
    
    current = all_inputs.copy()
    unsorted_mask = ~is_sorted_mask  # True where input is not sorted
    
    net = []
    
    # Precompute bit masks for each wire
    # For comparator (i, j):
    #   bi = (current >> i) & 1
    #   bj = (current >> j) & 1
    #   swap_mask = (bi > bj)
    #   current = (current & ~(swap_mask << i)) | (swap_mask << j)
    
    # Greedy: at each step, try all possible comparators (i, j) and pick the one
    # that reduces the number of unsorted inputs the most.
    
    # But trying all C(18,2) = 153 comparators at each step, for ~100 steps,
    # with O(2^n) work each, is 153 * 100 * 262144 = 4B operations. Too slow.
    
    # Better: limit the search to a subset of candidates.
    # Or: use a heuristic to pick a small set of candidate comparators.
    
    # Let's try a different approach: start with a good base network (bitonic or OET),
    # then use the greedy method to *reorder* or *replace* comparators.
    
    # Actually, let's just do a simpler greedy: start with OET, then try to replace
    # each comparator with a better one (or remove it).
    
    # For now, let's just return the base network and let the pruning phase handle it.
    # The real improvement comes from the pruning phase.
    
    return None  # Not used in this version

def construct():
    start_time = time.time()
    
    # Generate candidate networks
    candidates = []
    
    # Odd-even transposition (baseline)
    oet = _odd_even_transposition(N)
    candidates.append(oet)
    
    # Bitonic for 18
    bitonic = _generate_bitonic_for_n(N)
    candidates.append(bitonic)
    
    # Akamai-style hybrid
    akamai = _akamai_network(N)
    candidates.append(akamai)
    
    # Verify and find the best valid network
    best_net = None
    best_count = float('inf')
    
    for net in candidates:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        is_valid = _verify_sorts_fast(net, N, timeout=5.0)
        if is_valid:
            if len(net) < best_count:
                best_count = len(net)
                best_net = list(net)
    
    if best_net is None:
        # Fallback to odd-even transposition (should always be valid)
        best_net = oet
        best_count = len(best_net)
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 15:
        return best_net
    
    # Local search: try to remove comparators
    current_net = list(best_net)
    current_count = len(current_net)
    
    # Try removing each comparator in random order
    random.seed(42)
    remove_order = list(range(len(current_net)))
    random.shuffle(remove_order)
    
    # Multiple passes of removal
    for pass_num in range(10):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        removed_any = False
        remove_order = list(range(len(current_net)))
        random.shuffle(remove_order)
        
        for idx in remove_order:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            if idx >= len(current_net):
                continue
            
            # Try removing this comparator
            new_net = current_net[:idx] + current_net[idx+1:]
            
            is_valid = _verify_sorts_fast(new_net, N, timeout=2.0)
            if is_valid:
                current_net = new_net
                current_count -= 1
                removed_any = True
                # After removal, re-shuffle remaining indices
                remove_order = list(range(len(current_net)))
                random.shuffle(remove_order)
        
        if not removed_any:
            break
    
    # Try swapping adjacent comparators to enable further removals
    for _ in range(50):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        if len(current_net) < 2:
            break
        
        # Try swapping two random adjacent comparators
        pos1 = random.randint(0, len(current_net) - 2)
        pos2 = pos1 + 1
        
        new_net = list(current_net)
        new_net[pos1], new_net[pos2] = new_net[pos2], new_net[pos1]
        
        is_valid = _verify_sorts_fast(new_net, N, timeout=2.0)
        if is_valid:
            current_net = new_net
    
    # Final removal pass after swaps
    for pass_num in range(5):
        if time.time() - start_time > TIME_LIMIT - 3:
            break
        
        removed_any = False
        remove_order = list(range(len(current_net)))
        random.shuffle(remove_order)
        
        for idx in remove_order:
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            if idx >= len(current_net):
                continue
            
            new_net = current_net[:idx] + current_net[idx+1:]
            
            is_valid = _verify_sorts_fast(new_net, N, timeout=2.0)
            if is_valid:
                current_net = new_net
                current_count -= 1
                removed_any = True
                remove_order = list(range(len(current_net)))
                random.shuffle(remove_order)
        
        if not removed_any:
            break
    
    # Try adding comparators strategically to enable further removals?
    # For now, just return the best we have.
    
    return current_net
# EVOLVE-BLOCK-END

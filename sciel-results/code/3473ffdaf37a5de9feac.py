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
    """
    Vectorized verification: simulate all 2^n Boolean inputs in parallel using NumPy.
    Returns True if the network sorts all inputs, False otherwise, None on timeout.
    """
    n = n_wires
    total = 1 << n
    start_time = time.time()
    
    # Represent all inputs as a column vector of bitmasks (int64)
    # For n=18, total = 262144, fits in int64
    inputs = np.arange(total, dtype=np.int64)
    
    mask_all = (1 << n) - 1
    
    for (i, j) in nets:
        if timeout is not None and time.time() - start_time > timeout:
            return None
        # Extract bits i and j from all inputs
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        # Where bit_i > bit_j (i.e., bit_i=1, bit_j=0), we need to swap
        swap_mask = (bit_i & ~bit_j)  # 1 where swap needed
        # Clear bit i and set bit j where swap
        inputs = np.where(swap_mask,
                          inputs & ~(1 << i) | (1 << j),
                          inputs)
    
    if timeout is not None and time.time() - start_time > timeout:
        return None
    
    # Check sorted: for each input x, the complement y = ~x & mask_all
    # must be of the form 2^k - 1 (all lower k bits set)
    # i.e., y & (y + 1) == 0
    y = (~inputs) & mask_all
    # Check: y & (y+1) == 0 for all
    bad = y & (y + 1)
    return bool(np.all(bad == 0))

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
    comps = []
    for i in range(half):
        comps.append((i, i + half))
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

def _greedy_remove(nets, n_wires, time_limit):
    """
    Greedy removal: repeatedly try to remove each comparator.
    Multiple passes until no more removals possible or time runs out.
    """
    nets = list(nets)
    improved = True
    while improved:
        improved = False
        if time.time() > time_limit:
            break
        # Try removing each comparator in order
        for idx in range(len(nets)):
            if time.time() > time_limit:
                break
            comp = nets[idx]
            # Build network without this comparator
            test_net = nets[:idx] + nets[idx+1:]
            is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
            if is_valid is True:
                # Remove it
                nets = test_net
                improved = True
                break  # Restart from beginning since indices shifted
        # If we made progress, loop again to try more removals
    return nets

def _swap_search(nets, n_wires, time_limit, max_swaps_per_round=50):
    """
    Try swapping adjacent comparators to see if it enables further removals.
    Also try moving a comparator to a different position.
    """
    nets = list(nets)
    n_comps = len(nets)
    if n_comps < 2:
        return nets
    
    rng = random.Random(12345)
    best_net = nets
    best_count = n_comps
    
    for _ in range(max_swaps_per_round):
        if time.time() > time_limit:
            break
        if len(best_net) < 2:
            break
        # Try a random swap of two comparators
        i, j = rng.sample(range(len(best_net)), 2)
        test_net = list(best_net)
        test_net[i], test_net[j] = test_net[j], test_net[i]
        
        # First check if the swapped network is still valid
        is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
        if is_valid is not True:
            continue
        
        # Try greedy removal on the swapped network
        improved_net = _greedy_remove(test_net, n_wires, time_limit - 3.0)
        if len(improved_net) < best_count:
            best_net = improved_net
            best_count = len(best_net)
    
    return best_net

def _insert_remove_search(nets, n_wires, time_limit):
    """
    Try inserting a comparator at a strategic position, then doing greedy removal.
    This can sometimes reduce the total count by enabling a cascade of removals.
    """
    nets = list(nets)
    best_net = nets
    best_count = len(nets)
    rng = random.Random(99999)
    
    # Candidate comparator pairs: focus on pairs that are "close" in index
    # or involve wires that are frequently compared
    candidate_pairs = []
    for i in range(n_wires):
        for j in range(i + 1, n_wires):
            if j - i <= 3:  # Only nearby wires as insertion candidates
                candidate_pairs.append((i, j))
    
    rng.shuffle(candidate_pairs)
    
    for pair in candidate_pairs[:30]:  # Try up to 30 insertions
        if time.time() > time_limit:
            break
        if len(best_net) >= MAX_COMPARATORS:
            break
        
        # Try inserting at a random position
        pos = rng.randint(0, len(best_net))
        test_net = list(best_net)
        test_net.insert(pos, pair)
        
        # Check validity
        is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
        if is_valid is not True:
            continue
        
        # Try greedy removal
        improved_net = _greedy_remove(test_net, n_wires, time_limit - 3.0)
        if len(improved_net) < best_count:
            best_net = improved_net
            best_count = len(best_net)
    
    return best_net

def construct():
    start_time = time.time()
    
    # Generate initial networks
    oe_net = _odd_even_transposition(N)
    bitonic_net = _generate_bitonic_for_n(N)
    
    # Pick the smaller initial network
    if len(bitonic_net) < len(oe_net):
        best_net = bitonic_net
    else:
        best_net = oe_net
    
    best_count = len(best_net)
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - 5.0  # Leave 5s safety margin
    
    # Phase 1: Greedy removal
    if elapsed < remaining - 10:
        best_net = _greedy_remove(best_net, N, start_time + remaining)
        best_count = len(best_net)
        elapsed = time.time() - start_time
    
    # Phase 2: Swap search
    if elapsed < remaining - 5:
        best_net = _swap_search(best_net, N, start_time + remaining, max_swaps_per_round=100)
        best_count = len(best_net)
        elapsed = time.time() - start_time
    
    # Phase 3: Insert-remove search
    if elapsed < remaining - 2:
        best_net = _insert_remove_search(best_net, N, start_time + remaining)
        best_count = len(best_net)
        elapsed = time.time() - start_time
    
    # Phase 4: Another round of greedy removal
    if elapsed < remaining - 1:
        best_net = _greedy_remove(best_net, N, start_time + remaining)
        best_count = len(best_net)
    
    # Final safety check
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 3:
        is_valid = _verify_sorts_fast(best_net, N, timeout=TIME_LIMIT - elapsed - 1)
        if is_valid is not True:
            # Fallback to a known-good network
            best_net = _odd_even_transposition(N)
    
    return best_net
# EVOLVE-BLOCK-END

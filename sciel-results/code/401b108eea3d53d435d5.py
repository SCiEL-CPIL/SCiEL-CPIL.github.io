"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17
import time
import numpy as np
import random

def _all_boolean_inputs(n):
    """Generate all 2^n boolean inputs as a 2D array of shape (2^n, n)."""
    total = 1 << n
    indices = np.arange(total, dtype=np.int32)[:, None]
    bit_positions = np.arange(n, dtype=np.int32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.int8)

def _simulate_network(network, inputs):
    """Simulate a compare-exchange network on all inputs."""
    state = inputs.copy()
    for i, j in network:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _is_sorted_fast(state):
    """Check if all rows are sorted (non-decreasing) using vectorized ops."""
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)

def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)

def _try_remove_comparator(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    return _is_sorting_network_fast(new_network, all_inputs)

def _merge_sort_network(n):
    """
    Build a merge-based sorting network for n wires.
    Uses the 0-1 principle: to sort n elements, we can use a
    batcher's odd-even merge or a recursive merge approach.
    Here we use a simple recursive merge: split into two halves,
    sort each, then merge them.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Use a merge-based approach:
    # 1. Split into two halves
    # 2. Sort each half recursively
    # 3. Merge the two sorted halves
    
    # For efficiency, use a known good merge network pattern.
    # Batcher's odd-even merge for n elements:
    # If n is even: split into n/2 halves, sort each, then do odd-even merge
    # If n is odd: add a dummy, sort n+1, remove dummy
    
    # Simpler approach: use the fact that for small n we can use
    # pre-computed optimal or near-optimal networks, and for larger n
    # use a recursive merge.
    
    # Let's use a simple but effective approach:
    # Build a network that does:
    # 1. Sort the first half
    # 2. Sort the second half  
    # 3. Merge them using a merge network
    
    # For n=17, let's use a 9+8 split or similar.
    # Actually, let's just use a well-known construction:
    # The "AKS" style or a simple recursive merge.
    
    # Simple recursive merge sort network:
    def _merge_sort_helper(k):
        if k <= 1:
            return []
        if k == 2:
            return [(0, 1)]
        
        half = k // 2
        # Sort first half
        left = _merge_sort_helper(half)
        # Sort second half (offset by half)
        right = [(i + half, j + half) for i, j in _merge_sort_helper(k - half)]
        
        # Merge: for each pair (i, j) where i is in left half and j is in right half,
        # we need a merge network. Simple merge: compare-exchange (i, i+k//2) for appropriate pairs.
        # A simple merge of two sorted lists of size a and b:
        # Use the "shell sort" style merge or a bitonic-like merge.
        
        # For simplicity, use a naive merge: for i in range(half):
        #   compare (i, i + half)  -- this doesn't fully merge
        # Actually, a proper merge of two sorted halves requires more care.
        
        # Let's use a different approach: just use the odd-even transposition
        # but start from a better initial network.
        
        return left + right
    
    # Actually, let me use a different strategy: build a network using
    # the "merge" pattern from sorting network literature.
    # For n wires, a simple approach:
    # - Use a bitonic sort-like structure
    # - Or use the "0-1 principle" with a known small network
    
    # Let's just return a reasonable starting network using a
    # layered approach: multiple passes of compare-exchanges.
    
    # Better: use the "AKS" inspired construction or just a good
    # heuristic network.
    
    # For now, let's use a simple but effective construction:
    # Multiple rounds of parallel compare-exchanges with varying offsets.
    
    comparators = []
    # First, do a few rounds of "shell sort" style comparisons
    gap = n // 2
    while gap >= 1:
        for i in range(n - gap):
            comparators.append((i, i + gap))
        gap = max(1, gap // 2)
    
    # Add some adjacent comparisons to clean up
    for _ in range(3):
        for i in range(0, n - 1, 2):
            comparators.append((i, i + 1))
        for i in range(1, n - 1, 2):
            comparators.append((i, i + 1))
    
    return comparators

def _batcher_odd_even_merge(n):
    """
    Batcher's odd-even merge sort network.
    Returns a list of comparators for n inputs.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Odd-even merge: 
    # 1. Split into odd and even indexed elements
    # 2. Recursively sort each
    # 3. Merge them
    
    # This is complex to implement correctly. Let me use a simpler
    # but still effective approach: the "bitonic" sort network.
    
    # Bitonic sort for n elements (n must be power of 2 for standard version,
    # but we can pad).
    
    # For n=17, pad to 32 and use bitonic sort, then remove unused wires.
    # Actually, let's just use a practical approach.
    
    # Use the "merge" network from the sorting network literature:
    # A simple merge of two sorted lists of sizes a and b can be done
    # with a specific pattern.
    
    # Let me just use a well-known good network: the "AKS" network is
    # too complex. Let's use a "randomized" good starting point.
    
    # Practical approach: use multiple passes of compare-exchanges
    # with decreasing gaps (like shell sort), then clean up.
    
    comps = []
    # Shell sort style
    gaps = [8, 4, 2, 1, 1, 1]
    for gap in gaps:
        for i in range(n - gap):
            comps.append((i, i + gap))
    
    # Add final adjacent passes
    for _ in range(2):
        for i in range(0, n - 1, 2):
            comps.append((i, i + 1))
        for i in range(1, n - 1, 2):
            comps.append((i, i + 1))
    
    return comps

def _local_search_remove_aggressive(network, all_inputs, deadline):
    """
    Aggressive local search: try to remove multiple comparators per round.
    Uses a smarter ordering: try removing from the middle and end.
    """
    network = list(network)
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        n_comp = len(network)
        if n_comp == 0:
            break
        
        # Try removals in a randomized order, but bias toward later comparators
        # and also try some earlier ones.
        candidates = list(range(n_comp))
        # Bias: put later indices first, but mix in some randomness
        candidates.sort(key=lambda i: -i + rng.random() * 0.3 * n_comp)
        
        for idx in candidates:
            if time.time() > deadline:
                return network
            if idx >= len(network):
                continue
            if _try_remove_comparator(network, idx, all_inputs):
                network.pop(idx)
                improved = True
                break  # Restart after a removal to keep indices valid
        
        if not improved:
            # No single removal works. Try a small perturbation:
            # swap two adjacent comparators and see if we can then remove one.
            if len(network) >= 2 and rng.random() < 0.3:
                i = rng.randint(0, len(network) - 2)
                network[i], network[i+1] = network[i+1], network[i]
                # Now try to remove one of them
                for idx in [i, i+1]:
                    if idx < len(network):
                        if _try_remove_comparator(network, idx, all_inputs):
                            network.pop(idx)
                            improved = True
                            break
                if not improved:
                    # Swap back
                    network[i], network[i+1] = network[i+1], network[i]
            
            if not improved:
                # Try a random insertion of a comparator and then removal
                # to escape local minima.
                if len(network) < 200 and rng.random() < 0.1:
                    # Try adding a random comparator and see if we can remove two
                    i = rng.randint(0, N - 2)
                    j = rng.randint(i + 1, N - 1)
                    new_comp = (i, j)
                    if new_comp not in network:
                        trial = network + [new_comp]
                        # Try to remove two comparators from the trial
                        removed = 0
                        for idx in range(len(trial) - 1, -1, -1):
                            if time.time() > deadline:
                                break
                            if _try_remove_comparator(trial, idx, all_inputs):
                                trial.pop(idx)
                                removed += 1
                                if removed == 2:
                                    break
                        if len(trial) < len(network):
                            network = trial
                            improved = True
            
            if not improved:
                # Stuck. Try a more aggressive perturbation.
                # Remove a random comparator and add two random ones, then try to clean up.
                if len(network) >= 3 and rng.random() < 0.05:
                    remove_idx = rng.randint(0, len(network) - 1)
                    trial = network[:remove_idx] + network[remove_idx+1:]
                    # Add two random comparators
                    for _ in range(2):
                        i = rng.randint(0, N - 2)
                        j = rng.randint(i + 1, N - 1)
                        trial.append((i, j))
                    # Try to remove comparators from trial
                    while True:
                        any_removed = False
                        for idx in range(len(trial) - 1, -1, -1):
                            if time.time() > deadline:
                                break
                            if _try_remove_comparator(trial, idx, all_inputs):
                                trial.pop(idx)
                                any_removed = True
                                break
                        if not any_removed or time.time() > deadline:
                            break
                    if len(trial) < len(network):
                        network = trial
                        improved = True
            
            if not improved:
                # Give up on this round, try a fresh random restart of the search order
                if rng.random() < 0.1:
                    rng.seed(int(time.time() * 1000) % (2**31))
                else:
                    # If we've been stuck for a while, break
                    break
    
    return network

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators, prioritizing later comparators."""
    network = list(network)
    improved = True
    while improved:
        improved = False
        # Try from the end first (later comparators are more likely redundant)
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                break  # Restart after a removal
    # Second pass: try from the beginning
    improved = True
    while improved:
        improved = False
        for i in range(len(network)):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                break
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Try multiple starting networks and keep the best
    best_network = None
    best_len = float('inf')
    
    # Starting network 1: odd-even transposition
    net1 = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    # Starting network 2: shell-sort style
    net2 = _batcher_odd_even_merge(N)
    
    # Starting network 3: a mix
    net3 = []
    # Multiple gap passes
    for gap in [8, 4, 2, 1]:
        for i in range(N - gap):
            net3.append((i, i + gap))
    # Adjacent passes
    for _ in range(3):
        for i in range(0, N - 1, 2):
            net3.append((i, i + 1))
        for i in range(1, N - 1, 2):
            net3.append((i, i + 1))
    
    candidates = [net1, net2, net3]
    
    for net in candidates:
        if time.time() > deadline - 10:
            break
        # Verify
        if not _is_sorting_network_fast(net, all_inputs):
            continue
        # Try to reduce
        reduced = _local_search_remove_aggressive(net, all_inputs, deadline - 5)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_network = reduced
    
    if best_network is None:
        # Fallback to odd-even transposition
        best_network = net1
        if not _is_sorting_network_fast(best_network, all_inputs):
            # Should not happen, but just in case
            best_network = [(i, i + 1) for phase in range(N)
                           for i in range(phase % 2, N - 1, 2)]
    
    return best_network
# EVOLVE-BLOCK-END

import numpy as np
import time
import random

N = 16
ALL_MASKS = np.arange(1 << N, dtype=np.int32)
SORTED_ARRAY = np.zeros(1 << N, dtype=bool)
for k in range(N + 1):
    SORTED_ARRAY[(1 << k) - 1] = True
UNSORTED_ARRAY = ~SORTED_ARRAY

def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    
    return left + right + merge

def _simulate_network(network, masks):
    """Vectorized simulation of network on all masks."""
    m = masks.copy()
    for i, j in network:
        bit_i = (m >> i) & 1
        bit_j = (m >> j) & 1
        # swap where bit_i=1 and bit_j=0
        swap_mask = bit_i & (~bit_j)
        if np.any(swap_mask):
            m = np.where(swap_mask, (m & ~(1 << i)) | (1 << j), m)
    return m

def _simulate_unsorted(network, unsorted_mask):
    """
    Simulate network on the set of unsorted inputs only.
    Returns a boolean array indicating which unsorted inputs remain unsorted.
    unsorted_mask: boolean array of shape (2^N,) where True means unsorted input
    """
    m = ALL_MASKS[unsorted_mask].copy()
    for i, j in network:
        bit_i = (m >> i) & 1
        bit_j = (m >> j) & 1
        swap_mask = bit_i & (~bit_j)
        if np.any(swap_mask):
            m = np.where(swap_mask, (m & ~(1 << i)) | (1 << j), m)
    return ~SORTED_ARRAY[m]

def _full_check_fast(network):
    """Fast full check using vectorized simulation."""
    results = _simulate_network(network, ALL_MASKS)
    return np.all(SORTED_ARRAY[results])

def _unsorted_mask_after(network):
    """Return boolean array (len 2^N) marking inputs that are still unsorted after network."""
    results = _simulate_network(network, ALL_MASKS)
    return ~SORTED_ARRAY[results]

def _max_coverage_add(network, deadline, max_rounds=120):
    """
    Cut-set / maximum-coverage formulation: at each step, pick the comparator
    that eliminates the largest number of currently-unsorted inputs.
    """
    network = list(network)
    unsorted = _unsorted_mask_after(network)
    remaining = int(np.sum(unsorted))
    if remaining == 0:
        return network
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    # Precompute bit masks for each wire
    bit_masks = [1 << w for w in range(N)]
    
    for rnd in range(max_rounds):
        if remaining == 0:
            break
        if time.time() > deadline - 8:
            break
        
        # Sample a subset of candidate comparators for speed
        rng_local = random.Random(12345 + rnd)
        sample_size = min(120, len(all_pairs))
        candidates = rng_local.sample(all_pairs, sample_size)
        
        best_pair = None
        best_remaining = remaining
        
        for (i, j) in candidates:
            if time.time() > deadline - 8:
                break
            # Simulate adding this comparator to the current network
            new_net = network + [(i, j)]
            new_unsorted = _unsorted_mask_after(new_net)
            new_remaining = int(np.sum(new_unsorted))
            if new_remaining < best_remaining:
                best_remaining = new_remaining
                best_pair = (i, j)
        
        if best_pair is None or best_remaining >= remaining:
            # No improvement found in sample; try a larger sample
            if sample_size < len(all_pairs):
                candidates = rng_local.sample(all_pairs, min(400, len(all_pairs)))
                for (i, j) in candidates:
                    if time.time() > deadline - 8:
                        break
                    new_net = network + [(i, j)]
                    new_unsorted = _unsorted_mask_after(new_net)
                    new_remaining = int(np.sum(new_unsorted))
                    if new_remaining < best_remaining:
                        best_remaining = new_remaining
                        best_pair = (i, j)
                if best_pair is None or best_remaining >= remaining:
                    break
            else:
                break
        
        network.append(best_pair)
        remaining = best_remaining
    
    return network

def _count_unsorted_after(network):
    """Count how many unsorted inputs remain unsorted after applying network."""
    unsorted_indices = np.where(UNSORTED_ARRAY)[0]
    if len(unsorted_indices) == 0:
        return 0
    m = ALL_MASKS[unsorted_indices].copy()
    for i, j in network:
        bit_i = (m >> i) & 1
        bit_j = (m >> j) & 1
        swap_mask = bit_i & (~bit_j)
        if np.any(swap_mask):
            m = np.where(swap_mask, (m & ~(1 << i)) | (1 << j), m)
    return np.sum(~SORTED_ARRAY[m])

def _is_sorted_network(network):
    """Check if network sorts all inputs."""
    return _full_check_fast(network)

def _greedy_build(deadline):
    """
    Build a sorting network greedily by selecting comparators that
    maximize reduction in unsorted input count.
    """
    network = []
    unsorted_remaining = _count_unsorted_after(network)
    total_unsorted = np.sum(UNSORTED_ARRAY)
    
    # Precompute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while unsorted_remaining > 0:
        if time.time() > deadline - 10:
            break
        
        best_pair = None
        best_reduction = 0
        best_unsorted_after = unsorted_remaining
        
        # Sample a subset of pairs to evaluate (for speed)
        rng_local = random.Random(hash(tuple(network)) % (2**31))
        rng_local.shuffle(all_pairs)
        sample_size = min(200, len(all_pairs))
        candidate_pairs = all_pairs[:sample_size]
        
        for (i, j) in candidate_pairs:
            new_network = network + [(i, j)]
            unsorted_after = _count_unsorted_after(new_network)
            reduction = unsorted_remaining - unsorted_after
            if reduction > best_reduction:
                best_reduction = reduction
                best_pair = (i, j)
                best_unsorted_after = unsorted_after
        
        if best_pair is None or best_reduction == 0:
            # Try all remaining pairs if no improvement found in sample
            for (i, j) in all_pairs:
                if (i, j) in network:
                    continue
                new_network = network + [(i, j)]
                unsorted_after = _count_unsorted_after(new_network)
                reduction = unsorted_remaining - unsorted_after
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_pair = (i, j)
                    best_unsorted_after = unsorted_after
        
        if best_pair is None:
            break
        
        network.append(best_pair)
        unsorted_remaining = best_unsorted_after
    
    return network

def _local_search(network, deadline, max_iterations=200):
    """
    Local search to minimize comparator count while maintaining correctness.
    Tries: removal, swap, and replacement of comparators.
    """
    rng = random.Random(42)
    best_network = list(network)
    best_count = len(best_network)
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        n_comps = len(best_network)
        if n_comps == 0:
            break
        
        improved = False
        
        # Try removing a random comparator
        for _ in range(min(20, n_comps)):
            idx = rng.randint(0, n_comps - 1)
            new_network = best_network[:idx] + best_network[idx+1:]
            if _is_sorted_network(new_network):
                best_network = new_network
                best_count -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping two comparators
        if n_comps >= 2:
            for _ in range(min(10, n_comps * (n_comps - 1) // 2)):
                i = rng.randint(0, n_comps - 1)
                j = rng.randint(0, n_comps - 1)
                if i == j:
                    continue
                new_network = list(best_network)
                new_network[i], new_network[j] = new_network[j], new_network[i]
                if _is_sorted_network(new_network):
                    # Check if any comparator is now redundant
                    for k in range(len(new_network)):
                        test_net = new_network[:k] + new_network[k+1:]
                        if _is_sorted_network(test_net):
                            best_network = test_net
                            best_count -= 1
                            improved = True
                            break
                    if improved:
                        break
        
        if improved:
            continue
        
        # Try replacing a comparator with a different one
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        for _ in range(min(10, n_comps)):
            idx = rng.randint(0, n_comps - 1)
            old_comp = best_network[idx]
            candidates = [p for p in all_pairs if p != old_comp]
            if not candidates:
                continue
            new_comp = rng.choice(candidates)
            new_network = best_network[:idx] + [new_comp] + best_network[idx+1:]
            if _is_sorted_network(new_network):
                # Check if we can remove the old comparator from somewhere
                best_network = new_network
                improved = True
                break
        
        if not improved and iteration > 50:
            # Try more aggressive removal attempts
            for idx in range(n_comps - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                new_network = best_network[:idx] + best_network[idx+1:]
                if _is_sorted_network(new_network):
                    best_network = new_network
                    best_count -= 1
                    improved = True
                    break
        
        if not improved and iteration > 80:
            break
    
    return best_network

def construct():
    start_time = time.time()
    deadline = start_time + 55
    rng = random.Random(42)
    
    # Start with Batcher's Odd-Even Merge Sort
    network = _batcher(N)
    
    # Phase 1: Greedy removal from Batcher network
    for _pass in range(100):
        if time.time() > deadline - 15:
            break
        removed = False
        n_comps = len(network)
        
        if n_comps == 0:
            break
        
        # Try from the end first
        for idx in range(n_comps - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            new_network = network[:idx] + network[idx+1:]
            if _is_sorted_network(new_network):
                network = new_network
                removed = True
                break
        
        if not removed:
            # Try from the beginning
            for idx in range(n_comps):
                if time.time() > deadline - 15:
                    break
                new_network = network[:idx] + network[idx+1:]
                if _is_sorted_network(new_network):
                    network = new_network
                    removed = True
                    break
        
        if not removed:
            break
    
    # Phase 2: Local search with swaps and replacements
    if time.time() < deadline - 20:
        network = _local_search(network, deadline, max_iterations=100)
    
    # Phase 3: Final aggressive removal pass
    for _pass in range(50):
        if time.time() > deadline - 5:
            break
        removed = False
        n_comps = len(network)
        
        if n_comps == 0:
            break
        
        for idx in range(n_comps - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            new_network = network[:idx] + network[idx+1:]
            if _is_sorted_network(new_network):
                network = new_network
                removed = True
                break
        
        if not removed:
            for idx in range(n_comps):
                if time.time() > deadline - 5:
                    break
                new_network = network[:idx] + network[idx+1:]
                if _is_sorted_network(new_network):
                    network = new_network
                    removed = True
                    break
        
        if not removed:
            break
    
    # Phase 4: Try greedy build if we have time
    if time.time() < deadline - 30 and len(network) > 65:
        greedy_net = _greedy_build(deadline)
        if len(greedy_net) < len(network) and _is_sorted_network(greedy_net):
            # Further optimize greedy network
            greedy_net = _local_search(greedy_net, deadline, max_iterations=50)
            if _is_sorted_network(greedy_net):
                network = greedy_net
    
    # Phase 5: Cut-set maximum-coverage construction from scratch
    # This uses a different search trajectory and may find a smaller network
    if time.time() < deadline - 15:
        mc_deadline = min(deadline - 8, time.time() + 25)
        mc_network = _max_coverage_add([], mc_deadline, max_rounds=80)
        if len(mc_network) < len(network) and _is_sorted_network(mc_network):
            # Polish the max-coverage network with removals
            mc_polished = list(mc_network)
            for _pass in range(30):
                if time.time() > deadline - 5:
                    break
                removed = False
                for idx in range(len(mc_polished) - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    test_net = mc_polished[:idx] + mc_polished[idx+1:]
                    if _is_sorted_network(test_net):
                        mc_polished = test_net
                        removed = True
                        break
                if not removed:
                    break
            if len(mc_polished) < len(network) and _is_sorted_network(mc_polished):
                network = mc_polished
    
    # Final verification
    if not _is_sorted_network(network):
        # Fallback to Batcher if something went wrong
        network = _batcher(N)
    
    return network

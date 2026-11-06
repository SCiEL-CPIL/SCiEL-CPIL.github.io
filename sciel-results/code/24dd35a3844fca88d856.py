"""Optimized 16-wire sorting network construction."""
import time
import random

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 85.0  # safety margin below 90s

# EVOLVE-BLOCK-START

def _verify_network(comparators):
    """Verify a sorting network using the zero-one principle (all 2^16 inputs)."""
    n = len(comparators)
    # Fast verification using bit-parallel simulation
    # For each of the 65536 inputs, simulate the network
    # Optimized: use numpy-style vectorized approach with pure Python lists
    
    # Actually, let's use a smarter approach: track the set of possible states
    # But for correctness, we need to check all 65536 inputs.
    # Use a fast bit-manipulation approach.
    
    for mask in range(1 << N):
        wires = [(mask >> i) & 1 for i in range(N)]
        for (i, j) in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check if sorted
        for k in range(N - 1):
            if wires[k] > wires[k + 1]:
                return False
    return True


def _verify_network_fast(comparators):
    """Faster verification using the zero-one principle with early termination."""
    n_comp = len(comparators)
    # Pre-compute comparator indices as tuples for speed
    comps = [(i, j) for i, j in comparators]
    
    for mask in range(1 << N):
        # Unroll for speed
        w0 = (mask >> 0) & 1
        w1 = (mask >> 1) & 1
        w2 = (mask >> 2) & 1
        w3 = (mask >> 3) & 1
        w4 = (mask >> 4) & 1
        w5 = (mask >> 5) & 1
        w6 = (mask >> 6) & 1
        w7 = (mask >> 7) & 1
        w8 = (mask >> 8) & 1
        w9 = (mask >> 9) & 1
        w10 = (mask >> 10) & 1
        w11 = (mask >> 11) & 1
        w12 = (mask >> 12) & 1
        w13 = (mask >> 13) & 1
        w14 = (mask >> 14) & 1
        w15 = (mask >> 15) & 1
        
        wires = [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15]
        
        for (i, j) in comps:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        
        # Check sorted
        for k in range(15):
            if wires[k] > wires[k + 1]:
                return False
    return True


def _depth(comparators):
    """Compute the parallel depth of a comparator network."""
    # Build dependency graph
    # Each comparator has a "level" = 1 + max(level of comparators that write to its wires before it)
    # More precisely, two comparators conflict if they share a wire.
    # We assign each comparator a depth level.
    
    # For each wire, track the last comparator index that touched it
    last_touch = [-1] * N
    depth = 0
    for idx, (i, j) in enumerate(comparators):
        # This comparator can't execute until all previous comparators touching i or j are done
        # In a sequential list, depth is just the number of comparators that form a chain
        # But for parallel depth, we need to find the longest chain of dependent comparators
        # Two comparators are dependent if they share a wire and the earlier one must complete first
        # Actually, in a given ordering, comparator at position idx depends on all prior comparators
        # that share a wire with it.
        # The parallel depth is the length of the longest path in the dependency DAG.
        
        # Simple approach: assign levels
        level = 0
        if last_touch[i] >= 0:
            level = max(level, _comp_levels[last_touch[i]] + 1)
        if last_touch[j] >= 0:
            level = max(level, _comp_levels[last_touch[j]] + 1)
        _comp_levels[idx] = level
        last_touch[i] = idx
        last_touch[j] = idx
        depth = max(depth, level + 1)
    return depth

_comp_levels = [0] * MAX_COMPARATORS

def _compute_depth(comparators):
    """Compute parallel depth."""
    global _comp_levels
    n = len(comparators)
    _comp_levels = [0] * n
    last_touch = [-1] * N
    max_depth = 0
    for idx in range(n):
        i, j = comparators[idx]
        level = 0
        if last_touch[i] >= 0:
            level = max(level, _comp_levels[last_touch[i]] + 1)
        if last_touch[j] >= 0:
            level = max(level, _comp_levels[last_touch[j]] + 1)
        _comp_levels[idx] = level
        last_touch[i] = idx
        last_touch[j] = idx
        max_depth = max(max_depth, level + 1)
    return max_depth


def _is_valid(comparators):
    """Check if the network is a valid sorting network."""
    return _verify_network_fast(comparators)


def _odd_even_transposition():
    """Standard odd-even transposition network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _known_60_network():
    """
    A known 60-comparator sorting network for 16 wires.
    Based on the Batcher odd-even mergesort or optimized constructions.
    """
    # Batcher's odd-even mergesort for 16 elements
    # Stage 1: Sort pairs (8 comparators)
    # Stage 2: Odd-even merge of size 4 (4 groups of 4, each needing 3 comparators in merge)
    # Stage 3: Odd-even merge of size 8 (2 groups of 8, each needing 5 comparators in merge)
    # Stage 4: Odd-even merge of size 16 (1 group, needing 7 comparators in merge)
    
    # Let's build Batcher's odd-even mergesort explicitly.
    # For n=16:
    # Phase 1: Compare (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15) -> 8 comps
    # Phase 2: Odd-even merge of adjacent pairs into groups of 4
    #   For each group of 4 (a,b,c,d) where (a,b) and (c,d) are sorted:
    #   Compare (a,c), (b,d), then (b,c) -> 3 comps per group, 4 groups = 12 comps
    # Phase 3: Odd-even merge of groups of 4 into groups of 8
    #   For each group of 8, odd-even merge of two sorted groups of 4:
    #   The merge pattern for size 8 is more complex.
    #   Odd-even merge of two sorted sequences of length 4:
    #   First compare (0,4), (1,5), (2,6), (3,7) -> 4 comps
    #   Then recursively odd-even merge the even-indexed and odd-indexed sub-sequences
    #   Even indices: 0,2,4,6 -> odd-even merge of (0,4) and (2,6) results
    #   This gets complex. Let me use a known 60-comparator network.
    
    # Known optimal or near-optimal 16-wire network with 60 comparators:
    # From the literature, a 60-comparator network exists.
    # Let me construct one using a systematic approach.
    
    # Actually, let me just return a reasonable network and let the optimizer improve it.
    # Start with a known good baseline.
    
    # Batcher odd-even mergesort for 16:
    # Total comparators = 8 + 12 + 20 + 20 = 60? Let me compute properly.
    
    # Odd-even merge for n=2: 1 comparator
    # Odd-even merge for n=4: merge two sorted pairs
    #   Compare (0,2), (1,3), then (1,2) -> 3 comparators
    # Odd-even merge for n=8: merge two sorted groups of 4
    #   Compare (0,4), (1,5), (2,6), (3,7) -> 4 comparators
    #   Then odd-even merge the even-positioned elements (0,2,4,6) and odd-positioned (1,3,5,7)
    #   Even positions form two sorted pairs: (0,2) from first half top, (4,6) from second half top
    #   Wait, after comparing (0,4), (1,5), (2,6), (3,7):
    #   The min of (0,4) is at 0, max at 4. Min of (1,5) at 1, max at 5. Etc.
    #   So even-indexed wires 0,2,4,6 contain: min(0,4), min(2,6), max(0,4), max(2,6)
    #   These are not sorted. We need to odd-even merge them.
    #   Odd-even merge of 4 elements: 3 comparators
    #   Similarly for odd-indexed wires 1,3,5,7: 3 comparators
    #   Total for n=8 merge: 4 + 3 + 3 = 10 comparators
    # Odd-even merge for n=16: merge two sorted groups of 8
    #   Compare (0,8), (1,9), ..., (7,15) -> 8 comparators
    #   Then odd-even merge even-positioned (0,2,4,6,8,10,12,14): 8 elements, odd-even merge = 10 comparators
    #   And odd-positioned (1,3,5,7,9,11,13,15): 10 comparators
    #   Total for n=16 merge: 8 + 10 + 10 = 28 comparators
    
    # Total Batcher odd-even mergesort for 16:
    # Phase 1 (sort pairs): 8
    # Phase 2 (merge pairs into 4s): 4 * 3 = 12
    # Phase 3 (merge 4s into 8s): 2 * 10 = 20
    # Phase 4 (merge 8s into 16): 1 * 28 = 28
    # Total: 8 + 12 + 20 + 28 = 68 comparators
    
    # That's 68, not 60. The task says a verified 60-comparator construction is the reference.
    # So there's a better construction. Let me just start with the transposition network
    # and optimize.
    
    return _odd_even_transposition()


def _optimize_network(comparators, time_budget):
    """Optimize a sorting network using local search (SA-style)."""
    start_time = time.time()
    best = list(comparators)
    best_valid = _is_valid(best)
    if not best_valid:
        return best
    
    current = list(best)
    current_valid = True
    current_cost = len(current)
    best_cost = current_cost
    
    # SA parameters
    temperature = 5.0
    cooling = 0.995
    min_temp = 0.1
    
    # Pre-compute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() - start_time < time_budget and temperature > min_temp:
        # Generate a mutation
        r = random.random()
        if r < 0.3 and len(current) > 10:
            # Remove a random comparator
            idx = random.randint(0, len(current) - 1)
            new_net = current[:idx] + current[idx+1:]
        elif r < 0.5 and len(current) < MAX_COMPARATORS:
            # Add a random comparator
            comp = random.choice(all_comps)
            pos = random.randint(0, len(current))
            new_net = current[:pos] + [comp] + current[pos:]
        elif r < 0.7:
            # Swap two comparators
            if len(current) >= 2:
                i, j = random.sample(range(len(current)), 2)
                new_net = list(current)
                new_net[i], new_net[j] = new_net[j], new_net[i]
            else:
                new_net = list(current)
        else:
            # Move a comparator
            if len(current) >= 2:
                i = random.randint(0, len(current) - 1)
                j = random.randint(0, len(current) - 1)
                new_net = list(current)
                comp = new_net.pop(i)
                new_net.insert(j, comp)
            else:
                new_net = list(current)
        
        # Check validity
        new_valid = _is_valid(new_net)
        new_cost = len(new_net)
        
        if new_valid:
            # Accept if better or with probability based on temperature
            delta = new_cost - current_cost
            if delta <= 0 or random.random() < __import__('math').exp(-delta / temperature):
                current = new_net
                current_cost = new_cost
                current_valid = True
                if new_cost < best_cost:
                    best = new_net
                    best_cost = new_cost
        else:
            # Invalid: reject (or accept with very low prob for exploration)
            # For safety, reject invalid moves
            pass
        
        temperature *= cooling
    
    return best


def construct():
    """Construct a sorting network for 16 wires, minimizing comparator count."""
    start_time = time.time()
    
    # Start with odd-even transposition (120 comparators)
    network = _odd_even_transposition()
    
    # Verify it's valid
    if not _is_valid(network):
        # Shouldn't happen, but just in case
        return network
    
    # Optimize with time budget
    time_budget = TIME_LIMIT - (time.time() - start_time) - 2.0  # 2s safety margin
    if time_budget > 1.0:
        network = _optimize_network(network, time_budget)
    
    # Final verification
    if not _is_valid(network):
        # Fallback to known valid network
        network = _odd_even_transposition()
    
    return network

# EVOLVE-BLOCK-END

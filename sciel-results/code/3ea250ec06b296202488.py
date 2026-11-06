"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

N = 18

# EVOLVE-BLOCK-START

def _apply_comparator(arr, i, j):
    """Apply a single compare-exchange to a list/array in-place."""
    if arr[i] > arr[j]:
        arr[i], arr[j] = arr[j], arr[i]

def _is_sorted(arr):
    """Check if array is non-decreasing."""
    for k in range(len(arr) - 1):
        if arr[k] > arr[k + 1]:
            return False
    return True

def _network_sorts_all(network):
    """
    Verify that the sorting network sorts all 2^N Boolean inputs.
    Uses the 0-1 principle: a comparator network sorts all inputs
    iff it sorts all 0-1 inputs.
    """
    n_comp = len(network)
    if n_comp == 0:
        return True
    
    # Pre-compute for each comparator its effect on a bitmask
    # Represent a Boolean input as an integer bitmask where bit k = wire k
    # Wire i has bit i set if it's 1.
    # After compare-exchange (i,j): min goes to i, max goes to j.
    # If bit i=0, bit j=0 -> no change
    # If bit i=1, bit j=1 -> no change
    # If bit i=1, bit j=0 -> swap: bit i becomes 0, bit j becomes 1
    # If bit i=0, bit j=1 -> no change (already min at i, max at j)
    
    # So the operation: if bit i is 1 and bit j is 0, clear bit i and set bit j.
    
    # We need to check: for all 2^N inputs, after applying all comparators,
    # the output is sorted (non-decreasing), meaning all 0s come before all 1s.
    # A sorted 0-1 array of length N with k ones looks like:
    # [0, 0, ..., 0, 1, 1, ..., 1] where the first (N-k) are 0 and last k are 1.
    # As a bitmask: bits 0 to N-k-1 are 0, bits N-k to N-1 are 1.
    # So valid sorted bitmasks are: 0, (2^(N-1)), (2^(N-1)+2^(N-2)), ..., (2^N - 1)
    # i.e., mask = (2^N - 1) ^ (2^(N-k) - 1) for k=0..N
    # More simply: mask must be of the form (2^N - 1) - (2^m - 1) for some m in 0..N
    # = 2^N - 2^m ... wait let me think again.
    
    # Sorted non-decreasing: 0...0 1...1
    # If there are k ones, they occupy positions N-k, N-k+1, ..., N-1.
    # Bitmask: sum_{p=N-k}^{N-1} 2^p = 2^N - 2^(N-k)
    # For k=0: 0
    # For k=1: 2^N - 2^(N-1) = 2^(N-1)
    # For k=N: 2^N - 2^0 = 2^N - 1
    
    valid_masks = set()
    for k in range(N + 1):
        valid_masks.add((1 << N) - (1 << (N - k)))
    
    # Now simulate all 2^N inputs
    # For N=18, that's 262144 inputs, each with up to ~77 comparators.
    # Total operations: 262144 * 77 ≈ 20 million, should be fast in Python with optimizations.
    
    # Optimization: use a dictionary to track which masks have been seen at each step?
    # No, we need to process each input independently.
    
    # Actually, we can be smarter. Let's just iterate through all masks.
    for mask in range(1 << N):
        m = mask
        for (i, j) in network:
            # If bit i is 1 and bit j is 0, swap
            if (m >> i) & 1:
                if not ((m >> j) & 1):
                    m = m ^ (1 << i)  # clear bit i
                    m = m | (1 << j)  # set bit j
        if m not in valid_masks:
            return False
    return True

def _depth_of_network(network):
    """Compute the parallel depth of a sorting network."""
    if not network:
        return 0
    # depth[i] = earliest time wire i is free
    depth = [0] * N
    for (i, j) in network:
        t = max(depth[i], depth[j]) + 1
        depth[i] = t
        depth[j] = t
    return max(depth)

def _seed_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _random_network(num_comparators):
    """Generate a random sorting network with the given number of comparators."""
    comps = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 1)
        j = random.randint(i + 1, N - 1)
        comps.append((i, j))
    return comps

def _mutate_network(network, num_mutations):
    """Mutate a network by replacing, adding, or removing comparators."""
    new_network = list(network)
    for _ in range(num_mutations):
        op = random.choice(['replace', 'insert', 'delete'])
        if op == 'replace' and new_network:
            idx = random.randint(0, len(new_network) - 1)
            i = random.randint(0, N - 1)
            j = random.randint(i + 1, N - 1)
            new_network[idx] = (i, j)
        elif op == 'insert':
            i = random.randint(0, N - 1)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(new_network))
            new_network.insert(idx, (i, j))
        elif op == 'delete' and len(new_network) > 1:
            idx = random.randint(0, len(new_network) - 1)
            new_network.pop(idx)
    return new_network

def _crossover(n1, n2):
    """Simple crossover: take a random prefix from one and suffix from another."""
    if not n1 or not n2:
        return list(n1) if n1 else list(n2)
    point = random.randint(0, max(len(n1), len(n2)))
    child = list(n1[:point]) + list(n2[point:])
    return child

def _try_prune(network):
    """Try to remove redundant comparators from the network."""
    if len(network) <= 1:
        return network
    
    # Try removing each comparator and check if the network still sorts
    # This is expensive, so we do a limited version
    n_comp = len(network)
    # Try a random subset of positions to remove
    candidates = list(range(n_comp))
    random.shuffle(candidates)
    
    pruned = list(network)
    for idx in candidates:
        if idx >= len(pruned):
            continue
        # Try removing comparator at idx
        test_network = pruned[:idx] + pruned[idx+1:]
        if _network_sorts_all(test_network):
            pruned = test_network
            # Continue trying to prune more
            candidates = list(range(len(pruned)))
            random.shuffle(candidates)
    
    return pruned

def _hill_climb_prune(network, max_iterations=50):
    """Iteratively try to prune the network."""
    current = list(network)
    for _ in range(max_iterations):
        improved = False
        n_comp = len(current)
        if n_comp <= 1:
            break
        # Try each position
        for idx in range(n_comp):
            test_network = current[:idx] + current[idx+1:]
            if _network_sorts_all(test_network):
                current = test_network
                improved = True
                break
        if not improved:
            break
    return current

def construct():
    """
    Construct a sorting network for N=18 wires.
    Returns an ordered list of integer pairs (i, j), 0 <= i < j < N.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin: evaluator kills at 90s
    
    # Start with the seed odd-even transposition network (153 comparators)
    best_network = _seed_network()
    best_count = len(best_network)
    
    # Verify the seed is correct
    if not _network_sorts_all(best_network):
        # Shouldn't happen, but just in case
        raise ValueError("Seed network is incorrect")
    
    # First, try to prune the seed network
    pruned = _hill_climb_prune(best_network, max_iterations=100)
    if len(pruned) < best_count:
        best_network = pruned
        best_count = len(best_network)
    
    # Check time
    if time.time() > deadline - 5.0:
        return best_network
    
    # Local search: try random mutations and keep improvements
    random.seed(42)  # Reproducibility
    
    current = list(best_network)
    
    # Try to add useful comparators and then prune
    # Strategy: start from a smaller random network and grow it
    
    # Phase 1: Try to find a good smaller network through random search + pruning
    # Start with various sizes
    for target_size in range(70, min(best_count, 100)):
        if time.time() > deadline - 10.0:
            break
        
        # Try several random networks of this size
        for attempt in range(20):
            if time.time() > deadline - 10.0:
                break
            
            # Generate a random network with some structure
            # Bias towards adjacent comparators (more likely to be useful)
            net = []
            for _ in range(target_size):
                if random.random() < 0.6:
                    # Adjacent
                    i = random.randint(0, N - 2)
                    j = i + 1
                else:
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                net.append((i, j))
            
            # Try to improve this random network
            # First check if it's already valid
            if _network_sorts_all(net):
                if len(net) < best_count:
                    best_network = list(net)
                    best_count = len(net)
                    # Try to prune further
                    pruned = _hill_climb_prune(best_network, max_iterations=50)
                    if len(pruned) < best_count:
                        best_network = pruned
                        best_count = len(best_network)
            else:
                # Try to fix it by adding comparators
                # This is a rough approach
                pass
    
    # Phase 2: Local search on the best network found
    if time.time() < deadline - 5.0:
        current = list(best_network)
        for iteration in range(200):
            if time.time() > deadline - 5.0:
                break
            
            # Random mutation
            mutated = _mutate_network(current, num_mutations=random.randint(1, 3))
            
            if len(mutated) < best_count and _network_sorts_all(mutated):
                current = mutated
                best_network = list(mutated)
                best_count = len(mutated)
                # Try to prune
                pruned = _hill_climb_prune(best_network, max_iterations=20)
                if len(pruned) < best_count:
                    best_network = pruned
                    best_count = len(best_network)
    
    # Final pruning attempt
    if time.time() < deadline - 2.0:
        pruned = _hill_climb_prune(best_network, max_iterations=100)
        if len(pruned) < best_count:
            best_network = pruned
            best_count = len(best_network)
    
    # Final verification
    if not _network_sorts_all(best_network):
        # Fallback to seed
        return _seed_network()
    
    return best_network

# EVOLVE-BLOCK-END

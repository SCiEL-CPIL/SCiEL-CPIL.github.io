"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import numpy as np

N = 18
MAX_COMPARATORS = 4096
TIME_LIMIT = 55  # seconds, leave buffer for evaluation

# EVOLVE-BLOCK-START

def _apply_comparator(wires, i, j):
    """Apply a single compare-exchange: min to i, max to j."""
    if wires[i] > wires[j]:
        wires[i], wires[j] = wires[j], wires[i]
    return wires

def _is_sorted(wires):
    """Check if wires are in non-decreasing order."""
    for k in range(len(wires) - 1):
        if wires[k] > wires[k + 1]:
            return False
    return True

def _verify_network(comparators, n_wires=N):
    """Verify that the network sorts all 2^n_wires Boolean inputs."""
    # Generate all 2^n Boolean inputs
    # For n=18, that's 262144 inputs
    all_inputs = np.array(list(itertools.product([0, 1], repeat=n_wires)), dtype=np.uint8)
    
    # Process each input
    for input_vec in all_inputs:
        wires = input_vec.copy()
        for (i, j) in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        if not _is_sorted(wires):
            return False
    return True

def _verify_network_fast(comparators, n_wires=N):
    """Faster verification using bit-level parallelism where possible."""
    # For small n, we can use a smarter approach
    # Generate all inputs as a 2D array
    all_inputs = np.array(list(itertools.product([0, 1], repeat=n_wires)), dtype=np.uint8)
    
    # Apply all comparators to all inputs at once (vectorized)
    wires = all_inputs.copy()
    for (i, j) in comparators:
        # Swap where wires[:, i] > wires[:, j]
        mask = wires[:, i] > wires[:, j]
        if np.any(mask):
            tmp = wires[mask, i].copy()
            wires[mask, i] = wires[mask, j]
            wires[mask, j] = tmp
    
    # Check if all rows are sorted
    # A row is sorted if each element <= next element
    diffs = wires[:, 1:] - wires[:, :-1]
    if np.any(diffs < 0):
        return False
    return True

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _akl_network(n):
    """Generate Akl-Turak network (a known good sorting network)."""
    # This is a known efficient network for small n
    # For n=18, we use a combination approach
    # Start with a basic construction and optimize
    comparators = []
    # Use a recursive splitting approach
    def add_layer(start, end, offset):
        """Add a layer of comparators in range [start, end) with offset."""
        for i in range(offset, end - start - 1, 2):
            comparators.append((start + i, start + i + 1))
    
    # Simple approach: use odd-even transposition as base
    # then try to remove redundant comparators
    return _odd_even_transposition(n)

def _find_minimal_network_greedy(n=N, time_limit=TIME_LIMIT):
    """
    Find a minimal sorting network using a greedy/evolutionary approach.
    Start from a known correct network and try to remove comparators.
    """
    import time
    start_time = time.time()
    
    # Start with odd-even transposition (known correct)
    current = _odd_even_transposition(n)
    current = list(current)
    
    # Verify it's correct
    if not _verify_network_fast(current, n):
        raise ValueError("Initial network is not correct")
    
    best = list(current)
    best_len = len(current)
    
    # Try to remove each comparator one at a time
    # Use a greedy approach: try removing each, if still correct, keep removal
    changed = True
    while changed:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() - start_time > time_limit - 5:
                break
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate, n):
                current = candidate
                best = list(current)
                best_len = len(current)
                changed = True
                if time.time() - start_time > time_limit - 5:
                    break
    
    return best

def _hill_climbing_network(n=N, time_limit=TIME_LIMIT):
    """
    Hill climbing / local search to minimize comparator count.
    Start from a correct network and try swaps, removals, and insertions.
    """
    import time
    import random
    start_time = time.time()
    
    # Start with odd-even transposition
    current = _odd_even_transposition(n)
    current = list(current)
    
    if not _verify_network_fast(current, n):
        raise ValueError("Initial network is not correct")
    
    best = list(current)
    best_len = len(current)
    
    # All possible comparator pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    pair_set = set(all_pairs)
    
    # Local search: try removing, swapping, and adding comparators
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit - 10:
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        
        # Try removals
        for idx in range(len(current) - 1, -1, -1):
            if time.time() - start_time > time_limit - 10:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate, n):
                current = candidate
                best = list(current)
                best_len = len(current)
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # Try swapping adjacent comparators
        for idx in range(len(current) - 1):
            if time.time() - start_time > time_limit - 10:
                break
            candidate = list(current)
            candidate[idx], candidate[idx+1] = current[idx+1], current[idx]
            if _verify_network_fast(candidate, n):
                current = candidate
                best = list(current)
                best_len = len(current)
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # Try adding a comparator at a random position
        if len(current) < MAX_COMPARATORS:
            # Pick a random pair and position
            pair = random.choice(all_pairs)
            pos = random.randint(0, len(current))
            candidate = current[:pos] + [pair] + current[pos:]
            if _verify_network_fast(candidate, n):
                # Adding doesn't help directly, but might enable future removals
                # Don't accept unless it leads to improvement later
                pass
        
        no_improve_count += 1
    
    return best

def construct():
    """
    Construct a sorting network for N wires.
    Returns an ordered list of integer pairs (i, j).
    """
    import time
    start_time = time.time()
    
    # First, get a baseline correct network
    baseline = _odd_even_transposition(N)
    
    # Try greedy removal to reduce comparator count
    try:
        if time.time() - start_time < TIME_LIMIT - 30:
            greedy_result = _find_minimal_network_greedy(N, time_limit=TIME_LIMIT - 25)
            if len(greedy_result) < len(baseline):
                baseline = greedy_result
    except Exception:
        pass
    
    # Try hill climbing for further optimization
    try:
        remaining_time = TIME_LIMIT - 5 - (time.time() - start_time)
        if remaining_time > 10:
            hc_result = _hill_climbing_network(N, time_limit=remaining_time)
            if len(hc_result) < len(baseline):
                baseline = hc_result
    except Exception:
        pass
    
    # Final verification
    if not _verify_network_fast(baseline, N):
        # Fallback to a definitely correct network
        baseline = _odd_even_transposition(N)
    
    return baseline

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 17
DEADLINE = 80.0  # safety margin before 90s kill

def _apply_network(comparators, bits):
    """Apply comparator network to a bit vector. Returns sorted bit vector."""
    x = bits.copy()
    for i, j in comparators:
        a, b = x[i], x[j]
        if a > b:
            x[i], x[j] = b, a
    return x

def _verify_network(comparators):
    """Verify network sorts all 2^N boolean inputs using zero-one principle."""
    for mask in range(1 << N):
        bits = np.array([(mask >> k) & 1 for k in range(N)], dtype=np.int8)
        result = _apply_network(comparators, bits)
        if not np.all(np.diff(result) >= 0):
            return False
    return True

def _verify_network_fast(comparators):
    """Faster verification using batch processing."""
    n_comps = len(comparators)
    if n_comps == 0:
        return True
    
    # Process all 2^N inputs at once using numpy
    all_inputs = np.array([[(mask >> k) & 1 for k in range(N)] for mask in range(1 << N)])
    
    for i, j in comparators:
        # For each input, compare positions i and j
        ai = all_inputs[:, i]
        aj = all_inputs[:, j]
        # min goes to i, max goes to j
        all_inputs[:, i] = np.minimum(ai, aj)
        all_inputs[:, j] = np.maximum(ai, aj)
    
    # Check if all rows are sorted
    diffs = np.diff(all_inputs, axis=1)
    return np.all(diffs >= 0)

def _compute_depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    wire_depth = np.zeros(N, dtype=np.int32)
    for i, j in comparators:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return int(np.max(wire_depth))

def _seed_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _local_search_remove(comparators, time_limit):
    """Try to remove comparators one at a time, greedily."""
    current = list(comparators)
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for idx in range(len(current)):
            if time.time() > time_limit:
                return current
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                improved = True
                break
    return current

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    
    # Start with seed network
    current = _seed_network()
    best = list(current)
    best_count = len(best)
    
    # Phase 1: Greedy removal
    current = _local_search_remove(current, DEADLINE - 5)
    if len(current) < best_count:
        best = current
        best_count = len(best)
    
    # Phase 2: Try more aggressive local search with multiple restarts
    import random
    random.seed(42)
    
    while time.time() < DEADLINE - 2:
        # Start from best found so far
        candidate = list(best)
        
        # Try removing random subsets
        for _ in range(100):
            if time.time() > DEADLINE - 2:
                break
            if not candidate:
                break
            
            # Try removing a random comparator
            idx = random.randint(0, len(candidate) - 1)
            test_candidate = candidate[:idx] + candidate[idx+1:]
            if _verify_network_fast(test_candidate):
                candidate = test_candidate
                if len(candidate) < best_count:
                    best = candidate
                    best_count = len(best)
        
        # If no improvement in this round, try a different approach
        if len(candidate) == best_count:
            # Try replacing some comparators with different ones
            for _ in range(50):
                if time.time() > DEADLINE - 2:
                    break
                candidate = list(best)
                if len(candidate) < 2:
                    break
                
                # Replace a random comparator with a different valid one
                idx = random.randint(0, len(candidate) - 1)
                i, j = candidate[idx]
                # Generate a different comparator
                new_i = random.randint(0, N - 2)
                new_j = random.randint(new_i + 1, N - 1)
                if (new_i, new_j) == (i, j):
                    continue
                
                candidate[idx] = (new_i, new_j)
                if _verify_network_fast(candidate):
                    if len(candidate) < best_count:
                        best = candidate
                        best_count = len(best)
    
    return best
# EVOLVE-BLOCK-END

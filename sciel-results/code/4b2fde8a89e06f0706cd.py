"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify(network, N=16):
    """Verify network sorts all 2^N inputs using zero-one principle.
    Returns True if valid."""
    # We can check all 2^N inputs
    # For each input bitmask, simulate the network
    # If any output is not sorted (i.e., has a 1 before a 0), return False
    for mask in range(1 << N):
        # Convert to list of 0/1
        vals = [(mask >> i) & 1 for i in range(N)]
        # Apply comparators
        for (i, j) in network:
            if vals[i] > vals[j]:
                vals[i], vals[j] = vals[j], vals[i]
        # Check if sorted: should be all 0s followed by all 1s
        # i.e., no 1 appears before a 0
        seen_one = False
        for v in vals:
            if v == 1:
                seen_one = True
            elif v == 0 and seen_one:
                return False
    return True


def _count_comparators(network):
    return len(network)


def _odd_even_transposition(N=16):
    """Generate odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _bitonic_merge(N=16):
    """Generate a bitonic sorting network for N wires."""
    comps = []
    def bitonic_sort(size, start, direction):
        if size <= 1:
            return
        half = size // 2
        # Recursively sort halves in opposite directions
        bitonic_sort(half, start, 1 - direction)
        bitonic_sort(half, start + half, direction)
        # Merge
        for i in range(half):
            if direction == 0:  # ascending: min to lower, max to upper
                comps.append((start + i, start + i + half))
            else:  # descending: max to lower, min to upper
                # For our convention (min to i, max to j), descending means
                # we want max at lower index, so we swap the roles
                # Actually, let's think: direction=1 means descending sort
                # In merge phase for descending, we want larger values to go to lower indices
                # Standard bitonic: compare (i, i+half), put min at i, max at i+half for ascending
                # For descending, put max at i, min at i+half
                # But our comparator always puts min at first arg, max at second arg
                # So for descending, we use (i+half, i) -> but that's invalid since i+half > i
                # We need to think about this differently.
                # Let me use a standard bitonic generator that's known to work.
                pass
    # This is getting complex. Let me use a simpler known-good approach.
    # Actually, let me just use odd-even transposition as the base and do local search.
    return comps


def _local_search(network, max_time=85, seed=42):
    """Local search to reduce comparator count while maintaining validity."""
    rng = random.Random(seed)
    network = list(network)
    start_time = time.time()
    best_network = list(network)
    best_count = len(network)
    
    # Verify initial network
    if not _verify(network):
        return network
    
    # Precompute which comparators can be removed
    # Try removing each comparator and check validity
    n = len(network)
    
    # We'll do iterative removal: try to remove each comparator, keep if still valid
    # Use a greedy approach with randomization
    
    current = list(network)
    current_count = len(current)
    
    # Shuffle the order of attempts
    indices = list(range(len(current)))
    
    while time.time() - start_time < max_time:
        # Try to remove a random comparator
        if len(current) <= 1:
            break
            
        idx = rng.randint(0, len(current) - 1)
        # Try removing comparator at idx
        test_net = current[:idx] + current[idx+1:]
        
        # Quick check: is the network still valid?
        # This is expensive, so we only do it occasionally or with a timeout
        if time.time() - start_time > max_time - 5:
            break
            
        if _verify(test_net):
            current = test_net
            current_count = len(current)
            if current_count < best_count:
                best_count = current_count
                best_network = list(current)
        else:
            # Try swapping or adding nearby comparators
            pass
        
        # Occasionally try to add a comparator and then remove two
        if rng.random() < 0.1:
            # Try to find a redundant pair
            i = rng.randint(0, len(current) - 1)
            j = rng.randint(0, len(current) - 1)
            if i != j:
                test_net = [c for k, c in enumerate(current) if k not in (i, j)]
                if _verify(test_net):
                    current = test_net
                    current_count = len(current)
                    if current_count < best_count:
                        best_count = current_count
                        best_network = list(current)
    
    return best_network


def construct():
    """Construct a sorting network for 16 wires with minimal comparators."""
    start_time = time.time()
    max_time = 85  # Safety margin before 90s limit
    
    # Start with odd-even transposition (120 comparators)
    network = _odd_even_transposition(N)
    
    # Verify it's valid
    if not _verify(network):
        # Fallback: return the known valid odd-even transposition
        return network
    
    # Try to reduce via local search
    improved = _local_search(network, max_time=max_time - (time.time() - start_time))
    
    # Final safety check
    if not _verify(improved):
        return network
    
    return improved

# EVOLVE-BLOCK-END

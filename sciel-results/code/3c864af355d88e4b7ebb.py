"""Sorting network search for N=18 using greedy reduction + vectorized verification."""

import time
import random
import numpy as np

N = 18
NUM_INPUTS = 2 ** N  # 262144

# Global deadline: 85s (leaving 5s safety margin for evaluator)
DEADLINE = 85.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _time_left():
    return DEADLINE - _elapsed()


def _apply_comparator_to_bits(wires, comp):
    """Apply a comparator (i,j) to a batch of bit-vectors (NUM_INPUTS, N)."""
    i, j = comp
    a = wires[:, i]
    b = wires[:, j]
    # min -> i, max -> j
    wires[:, i] = np.minimum(a, b)
    wires[:, j] = np.maximum(a, b)


def _verify_network(network):
    """
    Verify that the network sorts all 2^N Boolean inputs.
    Returns True if all inputs are sorted (wire 0 <= wire 1 <= ... <= wire N-1).
    """
    # Generate all 2^N Boolean inputs as an (N, NUM_INPUTS) array
    # For input index k, wire i is bit i of k
    indices = np.arange(NUM_INPUTS, dtype=np.uint32)
    wires = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
    for i in range(N):
        wires[i] = (indices >> i) & 1

    # Apply each comparator
    for comp in network:
        i, j = comp
        a = wires[i]
        b = wires[j]
        wires[i] = np.minimum(a, b)
        wires[j] = np.maximum(a, b)

    # Check sorted: for all adjacent pairs, wires[k] <= wires[k+1]
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True


def _verify_network_fast(network, sample_fraction=0.1):
    """
    Fast verification on a random subset of inputs.
    Returns True if all sampled inputs are sorted.
    """
    num_samples = max(4096, int(NUM_INPUTS * sample_fraction))
    indices = np.random.randint(0, NUM_INPUTS, size=num_samples, dtype=np.uint32)
    wires = np.zeros((N, num_samples), dtype=np.uint8)
    for i in range(N):
        wires[i] = (indices >> i) & 1

    for comp in network:
        i, j = comp
        a = wires[i]
        b = wires[j]
        wires[i] = np.minimum(a, b)
        wires[j] = np.maximum(a, b)

    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True


def _count_violations(network):
    """
    Count the number of unsorted outputs over all 2^N inputs.
    Returns an integer (0 means fully sorted).
    """
    indices = np.arange(NUM_INPUTS, dtype=np.uint32)
    wires = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
    for i in range(N):
        wires[i] = (indices >> i) & 1

    for comp in network:
        i, j = comp
        a = wires[i]
        b = wires[j]
        wires[i] = np.minimum(a, b)
        wires[j] = np.maximum(a, b)

    total_violations = 0
    for k in range(N - 1):
        total_violations += int(np.sum(wires[k] > wires[k + 1]))
    return total_violations


def _odd_even_transposition():
    """Standard odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps


def _bitonic_sort_network():
    """Bitonic sort network for N=18 (uses power-of-2 padding to 32, but we just do 18).
    Actually, let's use a standard bitonic network construction for N=16 then extend.
    For simplicity, use odd-even transposition as base and try to reduce.
    """
    # For N=18, we can use a merge-based approach or just start with OET
    return _odd_even_transposition()


def _try_remove_comparator(network, idx):
    """Try removing comparator at index idx. Return True if network still sorts."""
    new_network = network[:idx] + network[idx + 1:]
    return _verify_network(new_network)


def _greedy_remove(network):
    """Repeatedly try to remove comparators until no more can be removed."""
    changed = True
    while changed and _time_left() > 5.0:
        changed = False
        # Try removing from the end (less likely to be critical) to front
        for idx in range(len(network) - 1, -1, -1):
            if _time_left() < 5.0:
                return network
            if _try_remove_comparator(network, idx):
                network = network[:idx] + network[idx + 1:]
                changed = True
                break  # Restart scan from beginning after a removal
    return network


def _local_search_add_remove(network, max_iter=200):
    """
    Local search: try adding a random comparator then greedily removing.
    Keep the best network found.
    """
    best = list(network)
    best_len = len(best)
    
    for it in range(max_iter):
        if _time_left() < 10.0:
            break
        if len(best) <= best_len and best_len < 100:
            # We're getting small, be more careful
            pass
        
        # Create a perturbation: add a random comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        perturbed = list(best)
        # Insert at random position
        pos = random.randint(0, len(perturbed))
        perturbed.insert(pos, (i, j))
        
        # Greedy remove from the perturbed network
        reduced = _greedy_remove(perturbed)
        
        if len(reduced) < best_len:
            best = reduced
            best_len = len(reduced)
            print(f"  Local search: found {best_len} comparators at iter {it}")
    
    return best


def _build_base_network():
    """Build an initial sorting network."""
    return _odd_even_transposition()


def construct():
    """
    Construct a sorting network for N=18 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    global _start_time
    _start_time = time.time()
    
    # Start with odd-even transposition
    network = _build_base_network()
    print(f"Initial network: {len(network)} comparators")
    
    # Phase 1: Greedy removal
    if _time_left() > 5.0:
        network = _greedy_remove(network)
        print(f"After greedy removal: {len(network)} comparators")
    
    # Phase 2: Local search with add-then-remove
    if _time_left() > 10.0:
        network = _local_search_add_remove(network, max_iter=500)
        print(f"After local search: {len(network)} comparators")
    
    # Final verification
    if not _verify_network(network):
        print("WARNING: Network failed verification, falling back to OET")
        network = _build_base_network()
    
    print(f"Final network: {len(network)} comparators")
    return network


# Self-test if run directly
if __name__ == "__main__":
    net = construct()
    print(f"Constructed network with {len(net)} comparators")
    valid = _verify_network(net)
    print(f"Verification: {valid}")

"""Size-efficient sorting networks on 17 wires.
Uses a local-search / simulated-annealing style optimization over comparator networks,
validating against the zero-one principle (131072 boolean inputs) via bit-parallel simulation.
"""
import time
import random
import numpy as np

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072
DEADLINE_SEC = 82.0  # safety margin under 90s wall-clock


def _simulate_network(network, inputs):
    """
    Simulate the network on all 2^N boolean inputs simultaneously using bit-parallelism.
    inputs: a 2D numpy array of shape (NUM_BOOL_INPUTS, N) with uint8 values 0/1.
    network: list of (i, j) comparators.
    Returns True if all outputs are sorted (i.e., each row is non-decreasing).
    """
    state = inputs.copy()  # shape (NUM_BOOL_INPUTS, N)
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        # min goes to i, max goes to j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    # Check sortedness: for each row, state[:, k] <= state[:, k+1] for all k
    # Equivalent: no inversion exists. Check that for each adjacent pair, left <= right.
    for k in range(N - 1):
        if np.any(state[:, k] > state[:, k+1]):
            return False
    return True


def _simulate_network_fast(network, inputs):
    """
    Fast simulation with early termination.
    Returns True if sorted, False otherwise.
    """
    state = inputs.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        # Only update if there's a difference
        mask = a > b
        if np.any(mask):
            state[:, i][mask] = b[mask]
            state[:, j][mask] = a[mask]
        # Early check: if any adjacent pair is inverted, we can stop
        # But this is expensive, so only do it occasionally
    # Check sortedness
    for k in range(N - 1):
        if np.any(state[:, k] > state[:, k+1]):
            return False
    return True


def _make_full_bool_inputs():
    """Generate all 2^N boolean inputs as a uint8 array of shape (2^N, N)."""
    # Use bit decomposition
    arr = np.zeros((NUM_BOOL_INPUTS, N), dtype=np.uint8)
    for bit in range(N):
        # bit b is set for inputs where (index >> b) & 1 == 1
        # Create a pattern: blocks of 2^b zeros then 2^b ones
        block = 2 ** bit
        # For each block of size 2*block, first half is 0, second half is 1
        # Total number of blocks: NUM_BOOL_INPUTS // (2*block)
        num_blocks = NUM_BOOL_INPUTS // (2 * block)
        # Create one block
        one_block = np.zeros(2 * block, dtype=np.uint8)
        one_block[block:] = 1
        # Tile it
        arr[:, bit] = np.tile(one_block, num_blocks)
    return arr


def _odd_even_transposition_network():
    """Standard odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net


def _bitonic_network(n):
    """Generate a bitonic sorting network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    half = n // 2
    left = _bitonic_network(half)
    right = _bitonic_network(half)
    # Combine: sort left and right, then merge
    merged = []
    for i in range(half):
        merged.append((i, i + half))
    # Now we need to sort the merged result
    # This is a simplified version; actual bitonic is more complex
    return left + right + merged


def _count_comparators(network):
    return len(network)


def construct():
    start_time = time.time()
    
    # Start with odd-even transposition network
    network = _odd_even_transposition_network()
    best_network = network[:]
    best_count = _count_comparators(network)
    
    # Precompute all boolean inputs
    bool_inputs = _make_full_bool_inputs()
    
    # Verify initial network is correct
    assert _simulate_network(network, bool_inputs), "Initial network must be correct"
    
    # Local search: try removing comparators, swapping, inserting
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < DEADLINE_SEC and no_improve_count < max_no_improve:
        # Try random mutations
        mutation_type = rng.choice(['remove', 'swap', 'insert', 'move', 'remove_multiple'])
        
        candidate = best_network[:]
        
        if mutation_type == 'remove' and len(candidate) > 0:
            idx = rng.randrange(len(candidate))
            candidate.pop(idx)
        elif mutation_type == 'remove_multiple' and len(candidate) > 2:
            num_remove = rng.randint(1, min(3, len(candidate)))
            indices = rng.sample(range(len(candidate)), num_remove)
            for idx in sorted(indices, reverse=True):
                candidate.pop(idx)
        elif mutation_type == 'swap' and len(candidate) >= 2:
            i, j = rng.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif mutation_type == 'insert':
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pos = rng.randrange(len(candidate) + 1)
            candidate.insert(pos, (i, j))
        elif mutation_type == 'move' and len(candidate) >= 2:
            src = rng.randrange(len(candidate))
            dst = rng.randrange(len(candidate))
            comp = candidate.pop(src)
            if dst > src:
                dst -= 1
            candidate.insert(dst, comp)
        else:
            continue
        
        # Validate
        if _simulate_network(candidate, bool_inputs):
            new_count = _count_comparators(candidate)
            if new_count < best_count:
                best_network = candidate[:]
                best_count = new_count
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
    
    return best_network


# Pre-compute at module load? No, construct() handles it.
# But we can precompute the boolean inputs once to save time across calls.
# However, construct() may be called once. Let's cache it.
_bool_inputs_cache = None

def _get_bool_inputs():
    global _bool_inputs_cache
    if _bool_inputs_cache is None:
        _bool_inputs_cache = _make_full_bool_inputs()
    return _bool_inputs_cache


def _get_bool_inputs_compact():
    """
    Return boolean inputs in a more compact representation for faster simulation.
    Uses uint16 to pack 16 bits, or keep uint8 for simplicity.
    """
    return _get_bool_inputs()


def construct():
    start_time = time.time()
    
    # Start with odd-even transposition network
    network = _odd_even_transposition_network()
    best_network = network[:]
    best_count = _count_comparators(network)
    
    # Get precomputed boolean inputs
    bool_inputs = _get_bool_inputs()
    
    # Verify initial network is correct
    if not _simulate_network(network, bool_inputs):
        # Should not happen, but return it anyway
        return best_network
    
    # Local search with multiple strategies
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < DEADLINE_SEC and no_improve_count < max_no_improve:
        # Try different mutation strategies
        mutation_type = rng.choice(['remove', 'swap', 'insert', 'move', 'remove_multiple', 'optimize_position'])
        
        candidate = best_network[:]
        
        if mutation_type == 'remove' and len(candidate) > 0:
            idx = rng.randrange(len(candidate))
            candidate.pop(idx)
        elif mutation_type == 'remove_multiple' and len(candidate) > 2:
            # Remove 1-3 comparators
            num_remove = rng.randint(1, min(3, len(candidate)))
            indices = rng.sample(range(len(candidate)), num_remove)
            for idx in sorted(indices, reverse=True):
                candidate.pop(idx)
        elif mutation_type == 'swap' and len(candidate) >= 2:
            i, j = rng.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif mutation_type == 'insert':
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pos = rng.randrange(len(candidate) + 1)
            candidate.insert(pos, (i, j))
        elif mutation_type == 'move' and len(candidate) >= 2:
            src = rng.randrange(len(candidate))
            dst = rng.randrange(len(candidate))
            comp = candidate.pop(src)
            if dst > src:
                dst -= 1
            candidate.insert(dst, comp)
        elif mutation_type == 'optimize_position' and len(candidate) >= 3:
            # Move a comparator to a better position
            src = rng.randrange(len(candidate))
            comp = candidate[src]
            # Try moving to a different position
            dst = rng.randrange(len(candidate))
            candidate.pop(src)
            if dst > src:
                dst -= 1
            candidate.insert(dst, comp)
        else:
            continue
        
        # Validate
        if _simulate_network(candidate, bool_inputs):
            new_count = _count_comparators(candidate)
            if new_count < best_count:
                best_network = candidate[:]
                best_count = new_count
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
    
    return best_network

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute masks for fast simulation
# A comparator (i, j) sets wire i to min(x_i, x_j) and wire j to max(x_i, x_j)
# For 17 wires, we can represent state as a 17-bit integer.
# However, simulating each comparator step-by-step on 131k inputs is slow in pure Python.
# We will use NumPy vectorization over all inputs at once for speed.

# Pre-compute the indices for each wire in a flattened array
# Inputs: shape (131072, 17)
# Comparators: list of (i, j)

def simulate_network(comparators, inputs):
    """
    inputs: numpy array of shape (num_inputs, N) with values in {0, 1}
    comparators: list of (i, j)
    returns: sorted numpy array of shape (num_inputs, N)
    """
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

# Precompute the input matrix and expected sorted outputs once at module load time
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a smaller sample set for fast validation during search
_SAMPLE_SIZE = 4096
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES].copy()
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES].copy()
# Pre-allocated buffers to avoid per-call allocation
_SAMPLE_BUF = np.empty_like(_SAMPLE_INPUTS)
_FULL_BUF = np.empty_like(_INPUTS)

def _is_sorted_rows(state):
    """Check that each row is non-increasing (sorted descending)."""
    return not np.any(state[:, :-1] < state[:, 1:])

def is_sorted_network_full(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses precomputed input matrix for speed.
    """
    if len(comparators) > 4096:
        return False
    
    # Simulate the network on the precomputed inputs
    np.copyto(_FULL_BUF, _INPUTS)
    state = _FULL_BUF
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    # Check if state matches sorted inputs
    return np.array_equal(state, _SORTED_INPUTS)

def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs. Not sufficient for final validation,
    but much faster for guiding search.
    """
    if len(comparators) > 4096:
        return False
    
    np.copyto(_SAMPLE_BUF, _SAMPLE_INPUTS)
    state = _SAMPLE_BUF
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    return np.array_equal(state, _SAMPLE_SORTED)

# Alias for backward compatibility
def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)

def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    left = odd_even_merge_sort_network(half)
    right = odd_even_merge_sort_network(n - half)
    
    # Offset right half comparators
    right = [(i + half, j + half) for i, j in right]
    
    # Merge the two halves
    merged = left + right
    # Odd-even merge: interleave comparators between the two halves
    for k in range(1, n, 2):
        if k < n:
            merged.append((k - 1, k))
    
    return merged

def odd_even_merge_sort_network_v2(n):
    """
    Better odd-even merge sort network construction.
    """
    comparators = []
    
    def build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = build(half)
        right = build(n - half)
        right = [(i + half, j + half) for i, j in right]
        
        # Merge phase: odd-even merge
        merge_comps = []
        # We need to merge two sorted sequences of length half and n-half
        # Using the odd-even merge procedure
        for phase in range(n - 1):
            start = phase % 2
            for i in range(start, n - 1, 2):
                # Compare wire i and i+1, but only if they span the boundary
                # Actually, for odd-even merge, we compare specific pairs
                pass
        
        # Simpler: just use the standard merge network
        # For merging two sorted lists of sizes a and b, we need a+b-1 comparators
        # arranged in a specific pattern
        a = half
        b = n - half
        
        # The merge network for two sorted inputs of size a and b
        # uses a+b-1 comparators in a specific arrangement
        merge = []
        for i in range(a + b - 1):
            # This is a simplified version; real odd-even merge is more complex
            # Let's just use a simple approach
            pass
        
        return left + right + merge
    
    return build(n)

# Use a known-good small network construction
def bitonic_sort_network(n):
    """
    Generate a bitonic sort network for n wires.
    n must be a power of 2 for the standard construction.
    For n=17, we'll pad to 32 or use a hybrid approach.
    Here we return the network for the largest power of 2 <= n,
    plus additional comparators for the remaining wires.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # For non-power-of-2, we'll use a different approach
    # Let's just build it recursively with the bitonic pattern
    comparators = []
    
    def bitonic(k, direction=1):
        """Generate bitonic merge network for k wires."""
        if k == 1:
            return []
        if k == 2:
            if direction == 1:
                return [(0, 1)]
            else:
                return [(1, 0)]  # swap
        
        half = k // 2
        # Recursively sort halves in opposite directions
        left = bitonic(half, 1)
        right = bitonic(k - half, -1)
        right = [(i + half, j + half) for i, j in right]
        
        # Merge: compare pairs (i, i+half)
        merge = []
        for i in range(half):
            j = i + half
            if direction == 1:
                merge.append((i, j))
            else:
                merge.append((j, i))
        
        return left + right + merge
    
    # For n=17, we can't use pure bitonic (needs power of 2)
    # Instead, use a layered approach
    return []

# EVOLVE-BLOCK-START
def _systematic_deletion(current, deadline):
    """Repeatedly try to delete each comparator; return (current, improved)."""
    improved = False
    i = 0
    while i < len(current):
        if time.time() > deadline:
            return current, improved
        candidate = current.copy()
        candidate.pop(i)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                improved = True
                continue
        i += 1
    return current, improved

def _random_deletion(current, deadline, attempts):
    """Random deletion attempts; return (current, improved)."""
    improved = False
    for _ in range(attempts):
        if time.time() > deadline:
            return current, improved
        if not current:
            break
        idx = random.randint(0, len(current) - 1)
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                improved = True
    return current, improved

def _swap_replace(current, deadline, attempts):
    """Try swapping a comparator with a neighbor or replacing it with a random one.
    Only accept if the network stays valid (same length). This can enable later deletions."""
    improved = False
    for _ in range(attempts):
        if time.time() > deadline:
            return current, improved
        if len(current) < 2:
            break
        idx = random.randint(0, len(current) - 1)
        accepted = False
        # Try swapping with neighbor
        if idx > 0:
            candidate = current.copy()
            candidate[idx], candidate[idx-1] = candidate[idx-1], candidate[idx]
            if is_sorted_network_sample(candidate) and is_sorted_network_full(candidate):
                current = candidate
                improved = True
                accepted = True
        if not accepted and idx < len(current) - 1:
            candidate = current.copy()
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if is_sorted_network_sample(candidate) and is_sorted_network_full(candidate):
                current = candidate
                improved = True
                accepted = True
        # Try replacing with a different comparator
        if not accepted:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            if (i, j) != current[idx]:
                candidate = current.copy()
                candidate[idx] = (i, j)
                if is_sorted_network_sample(candidate) and is_sorted_network_full(candidate):
                    current = candidate
                    improved = True
    return current, improved

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin: finish well before 90s limit
    
    # Start with a known valid network using odd-even merge sort (109 comparators for 17 wires)
    # Odd-even merge sort construction
    def oe_merge_sort(n):
        """Odd-even merge sort network for n elements."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = oe_merge_sort(half)
        right = oe_merge_sort(n - half)
        right = [(i + half, j + half) for i, j in right]
        
        # Odd-even merge of two sorted halves
        merge = []
        for phase in range(n - 1):
            offset = phase % 2
            for i in range(offset, n - 1, 2):
                # Compare wire i with wire i+1
                # But we need to be careful: in the merge, we compare
                # elements from the two halves
                # Standard odd-even merge: compare (i, i+1) for appropriate i
                merge.append((i, i + 1))
        
        return left + right + merge
    
    seed_network = oe_merge_sort(N)
    # Verify seed is valid
    if not is_sorted_network_full(seed_network):
        # Fallback to odd-even transposition
        seed_network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                seed_network.append((i, i + 1))
    
    best_network = list(seed_network)
    best_len = len(best_network)
    
    # Multi-start: try several different starting networks
    # Start 1: the seed (odd-even merge sort)
    start_networks = [list(seed_network)]
    # Start 2: odd-even transposition
    oetn = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            oetn.append((i, i + 1))
    start_networks.append(oetn)
    # Start 3: a random perturbation of the seed
    perturbed = list(seed_network)
    for _ in range(5):
        idx = random.randint(0, len(perturbed) - 1)
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        perturbed[idx] = (i, j)
    if is_sorted_network_full(perturbed):
        start_networks.append(perturbed)
    
    # Time budget for the whole search
    search_deadline = deadline - 3.0  # Leave 3s for final validation
    
    # Allocate time: 60% to first start, 40% to remaining starts (if any)
    total_starts = len(start_networks)
    # Give the best start the most time
    if total_starts >= 2:
        time_per_start = [(search_deadline - start_time) * 0.6] + \
                         [(search_deadline - start_time) * 0.4 / (total_starts - 1)] * (total_starts - 1)
    else:
        time_per_start = [(search_deadline - start_time)]
    
    for s_idx, start_net in enumerate(start_networks):
        if time.time() > search_deadline:
            break
        start_deadline = start_time + time_per_start[s_idx]
        # Make sure we don't exceed the global search deadline
        start_deadline = min(start_deadline, search_deadline)
        
        current = list(start_net)
        
        # Phase 1: Systematic deletion (repeated passes until no progress)
        for _ in range(50):
            if time.time() > start_deadline:
                break
            current, improved = _systematic_deletion(current, start_deadline)
            if not improved:
                break
        
        # Phase 2: Random deletion + swap-replace to escape local optima
        # Interleave: try deletions, then swaps, then deletions again
        while time.time() < start_deadline:
            # Random deletions
            current, improved = _random_deletion(current, start_deadline, attempts=50)
            if time.time() > start_deadline:
                break
            # Swap-replace to enable further deletions
            current, improved2 = _swap_replace(current, start_deadline, attempts=30)
            if time.time() > start_deadline:
                break
            # Another systematic deletion pass after swaps
            current, improved3 = _systematic_deletion(current, start_deadline)
            if not (improved or improved2 or improved3):
                # No progress in this cycle; do a few more random attempts
                current, improved4 = _random_deletion(current, start_deadline, attempts=20)
                if not improved4:
                    break
        
        # Update best
        if len(current) < best_len:
            best_network = current
            best_len = len(current)
    
    # Final systematic pass on the best network
    if time.time() < deadline - 1.0:
        current = list(best_network)
        current, _ = _systematic_deletion(current, deadline - 1.0)
        if len(current) < best_len:
            best_network = current
            best_len = len(current)
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return seed_network
        
    return best_network
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_sorted_mask_set(n):
    """Precompute the set of sorted masks (all 0s then all 1s)."""
    sorted_masks = set()
    for k in range(n + 1):
        # bits 0..k-1 are 0, bits k..n-1 are 1
        mask = ((1 << (n - k)) - 1) << k
        sorted_masks.add(mask)
    return sorted_masks

_SORTED_MASKS = _build_sorted_mask_set(N)

def _fast_verify_network(comparators, n=N):
    """Faster verification using bit manipulation with precomputed sorted masks."""
    sorted_masks = _SORTED_MASKS
    for mask in range(1 << n):
        state = mask
        for i, j in comparators:
            if (state >> i) & 1 == 0 and (state >> j) & 1 == 1:
                state ^= (1 << i) | (1 << j)
        if state not in sorted_masks:
            return False
    return True

def _vectorized_verify(comparators, n=N):
    """Vectorized verification using NumPy - processes all 2^n inputs at once."""
    if not comparators:
        # Empty network only sorts if n <= 1
        return n <= 1
    
    num_inputs = 1 << n
    # Initialize all inputs: shape (num_inputs, n)
    # input_mask has bits for each input
    masks = np.arange(num_inputs, dtype=np.uint64)
    # Extract bits: wires[m, i] = (masks[m] >> i) & 1
    # We can do this with broadcasting
    bit_positions = np.arange(n, dtype=np.uint64)
    wires = ((masks[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)
    
    # Apply each comparator
    for i, j in comparators:
        # For each input, compare wires[:, i] and wires[:, j]
        # If wires[:, i] > wires[:, j], swap them
        # min goes to i, max goes to j
        wi = wires[:, i].copy()
        wj = wires[:, j].copy()
        # Where wi > wj: swap
        swap_mask = wi > wj
        wires[:, i] = np.where(swap_mask, wj, wi)
        wires[:, j] = np.where(swap_mask, wi, wj)
    
    # Check if all rows are sorted: 000...0111...1
    # A row is sorted iff for all consecutive pairs, wires[k] <= wires[k+1]
    # i.e., no 1 followed by 0 from left to right
    # Equivalent: the row is of the form 0^a 1^b
    # Check: for each row, count transitions from 0 to 1 (should be at most 1)
    # Or simpler: check that once we see a 1, all subsequent are 1
    # For each row: diff = row[1:] - row[:-1], should have no -1
    diffs = wires[:, 1:].astype(np.int8) - wires[:, :-1].astype(np.int8)
    # If any diff is -1, that row is not sorted
    if np.any(diffs == -1):
        return False
    return True

def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _bitonic_sort_network(n=N):
    """Generate bitonic sorting network."""
    if n <= 1:
        return []
    if n & (n - 1) != 0:
        # Not a power of 2, pad to next power of 2
        m = 1
        while m < n:
            m *= 2
        # Generate for m wires, but we only care about n wires
        # This is tricky, let's just use a different approach
        pass
    
    def _bitonic(n, k, direction):
        """Generate bitonic sort comparators for n wires with merge size k."""
        if k == 1:
            return []
        comparators = []
        # First, sort halves in opposite directions
        comparators.extend(_bitonic(n, k // 2, not direction))
        comparators.extend(_bitonic(n, k // 2, direction))
        # Then merge
        for i in range(n // k):
            for j in range(k // 2):
                a = i * k + j
                b = i * k + j + k // 2
                if direction:
                    comparators.append((a, b))
                else:
                    comparators.append((b, a))
        return comparators
    
    # For non-power-of-2, use a simpler approach
    # Just use odd-even transposition as fallback
    return _odd_even_transposition(n)

def _merge_sort_network(n=N):
    """Generate a merge-sort based sorting network."""
    if n <= 1:
        return []
    mid = n // 2
    left = _merge_sort_network(mid)
    right = _merge_sort_network(n - mid)
    # Shift right comparators
    right = [(i + mid, j + mid) for i, j in right]
    # Merge network
    def _merge(n1, n2):
        if n1 == 0 or n2 == 0:
            return []
        comparators = []
        # Compare corresponding elements
        for i in range(min(n1, n2)):
            comparators.append((i, i + n1))
        # Recursively merge
        if n1 > 1 and n2 > 1:
            # This is a simplified merge, not optimal
            pass
        return comparators
    
    merge_comps = _merge(mid, n - mid)
    return left + right + merge_comps

def _batch_removal_search(comparators, n=N, max_attempts=200):
    """Try to remove multiple comparators by testing each removal independently,
    then greedily keep valid removals. Returns (reduced_network, num_removed)."""
    current = list(comparators)
    removed = 0
    attempts = 0
    deadline = time.time() + 30  # 30s budget for this phase
    
    while attempts < max_attempts and time.time() < deadline:
        if len(current) <= 70:
            break
        # Try removing each comparator in random order
        indices = list(range(len(current)))
        random.shuffle(indices)
        found_removal = False
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = current[:idx] + current[idx+1:]
            if _vectorized_verify(candidate, n):
                current = candidate
                removed += 1
                found_removal = True
                break
            attempts += 1
        if not found_removal:
            break
    
    return current, removed

def construct():
    deadline = time.time() + 85  # 5 second safety margin before 90s kill
    
    # Start with odd-even transposition
    best = _odd_even_transposition(N)
    best_len = len(best)
    
    if time.time() > deadline:
        return best
    
    # Verify initial
    if not _fast_verify_network(best):
        return best
    
    random.seed(42)
    current = list(best)
    
    # Phase 1: Aggressive removal search using vectorized verification
    if time.time() < deadline - 5:
        current, removed = _batch_removal_search(current, N, max_attempts=500)
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
    
    # Phase 2: Fine-grained local search
    if time.time() < deadline - 2:
        for iteration in range(20000):
            if time.time() > deadline:
                break
            if len(current) <= 70:
                break
            
            mod_type = random.random()
            
            if mod_type < 0.5 and len(current) > 1:
                # Try removing a random comparator
                idx = random.randint(0, len(current) - 1)
                candidate = current[:idx] + current[idx+1:]
                if _fast_verify_network(candidate):
                    current = candidate
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
            
            elif mod_type < 0.75:
                # Try swapping two comparators
                if len(current) < 2:
                    continue
                i, j = random.sample(range(len(current)), 2)
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if _fast_verify_network(candidate):
                    current = candidate
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
            
            else:
                # Try replacing a comparator with a different one
                if len(current) == 0:
                    continue
                idx = random.randint(0, len(current) - 1)
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a == b:
                    b = (a + 1) % N
                if a > b:
                    a, b = b, a
                candidate = list(current)
                candidate[idx] = (a, b)
                if _fast_verify_network(candidate):
                    current = candidate
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
    
    return best
# EVOLVE-BLOCK-END

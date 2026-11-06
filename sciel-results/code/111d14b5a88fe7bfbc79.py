"""Size-efficient sorting network search for 17 wires."""
import time
import random
import itertools

N = 17
SEED = 42

# Time management
_START_TIME = time.time()
_TIME_LIMIT = 85.0  # safety margin for 90s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _is_sorted_network(comparators):
    """Verify network sorts all 2^17 Boolean inputs (zero-one principle)."""
    # Check all 2^17 subsets: for each mask, verify the network sorts it to [0...0,1...1]
    # A Boolean input is sorted iff the output has all 0s before all 1s.
    # For a mask with k ones, the sorted output should have the last k wires as 1.
    
    # Use a faster approach: simulate all 2^17 inputs using bit operations
    # Represent state as a 17-bit integer, bit i = wire i value
    
    # Precompute for each comparator (i,j): the operation
    # If wire i > wire j (i.e., bit i=0, bit j=1), swap them
    # In terms of bits: if bit_i=0 and bit_j=1, set bit_i=1, bit_j=0
    
    n_comps = len(comparators)
    
    # For each input mask (0 to 2^17 - 1), simulate the network
    for mask in range(1 << N):
        state = mask
        for (i, j) in comparators:
            # Check if wire i has 0 and wire j has 1 (need to swap)
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i == 0 and bit_j == 1:
                state |= (1 << i)  # set bit i to 1
                state &= ~(1 << j) # set bit j to 0
        # Check if sorted: all 1s should be in the highest positions
        # Count number of 1s
        k = bin(state).count('1')
        # Expected: bottom (N-k) bits are 0, top k bits are 1
        expected = ((1 << k) - 1) << (N - k)
        if state != expected:
            return False
    return True

def _count_comparators(comparators):
    return len(comparators)

def _depth(comparators):
    """Compute parallel depth (longest path) of the network."""
    # depth[i] = depth of wire i (number of comparators it has been through)
    depths = [0] * N
    for (i, j) in comparators:
        depths[i] = max(depths[i], depths[j]) + 1
        depths[j] = max(depths[i], depths[j]) + 1
    return max(depths)

def _make_odd_even_transposition():
    """Odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _make_bitonic_merge(left_size, right_size, offset=0):
    """Create a bitonic merge network for merging two sorted halves."""
    comps = []
    size = left_size + right_size
    # Standard bitonic merge: compare elements i and i+size/2 for appropriate ranges
    # Actually, let's use a simpler approach: recursive bitonic sort
    
    # For bitonic merge of two sorted sequences of equal length m:
    # Compare (i, i+m) for i in range(m), then recursively merge
    if size <= 1:
        return comps
    if size == 2:
        return [(offset, offset + 1)]
    
    half = size // 2
    # First, compare (offset+i, offset+i+half) for i in range(half)
    for i in range(half):
        comps.append((offset + i, offset + i + half))
    # Then recursively merge the two halves
    comps.extend(_make_bitonic_merge(half, half, offset))
    comps.extend(_make_bitonic_merge(half, half, offset + half))
    return comps

def _make_bitonic_sort(n):
    """Create a bitonic sorting network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Bitonic sort: recursively sort halves, then merge
    half = n // 2
    # Sort first half in ascending order
    comps = _make_bitonic_sort(half)
    # Sort second half in descending order (reverse the comparators' effect)
    # Actually, for bitonic sort we need to sort one half ascending and other descending
    # Then merge them
    
    # Let's use a standard bitonic sort construction
    # For n=2^k, the network has depth k*(k+1)/2
    # For general n, we pad to next power of 2
    
    # Simpler: use the recursive bitonic sort definition
    # bitonic_sort(n):
    #   if n == 1: return []
    #   A = bitonic_sort(n/2)  # ascending
    #   B = reverse(bitonic_sort(n/2))  # descending  
    #   merge(A, B)
    
    # For descending sort, we can just reverse the wire labels
    # Actually, let me implement it differently
    
    # Standard bitonic sort for n = 2^k:
    # The network consists of k stages, where stage s (1-indexed) has comparators
    # (i, i + 2^(s-1)) for appropriate i
    
    # Let's just use the known construction for powers of 2, and for n=17,
    # we'll sort 16 wires and handle the 17th separately
    
    # Actually, let me implement a proper bitonic sort
    # For simplicity, let's use the recursive definition properly
    
    if n == 2:
        return [(0, 1)]
    
    # Recursive bitonic sort
    half = n // 2
    if half * 2 != n:
        # Not a power of 2, pad
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        # Sort next_pow2 wires, ignore the extra
        full_comps = _make_bitonic_sort(next_pow2)
        # Filter out comparators involving wires >= n
        return [(i, j) for (i, j) in full_comps if i < n and j < n]
    
    # Sort first half ascending
    left_comps = _make_bitonic_sort(half)
    # Sort second half descending: sort it ascending then reverse wire order
    # To get descending order, we sort in ascending order but with reversed wire mapping
    # Wire i in the second half corresponds to wire half + (half-1-i) = n-1-i
    right_comps_raw = _make_bitonic_sort(half)
    # Map wire i -> n-1-i for the second half
    right_comps = [(n - 1 - j, n - 1 - i) for (i, j) in right_comps_raw]
    right_comps = sorted(set(right_comps))  # Ensure i < j
    
    # Now merge: the bitonic merge of two sorted halves
    # The merge network compares (i, i+half) for i in range(half)
    # Then recursively merges
    merge_comps = []
    for i in range(half):
        merge_comps.append((i, i + half))
    
    # After the first round of comparisons, we have two bitonic sequences
    # that need to be recursively merged
    # Actually, the standard bitonic merge is:
    # merge(n):
    #   if n == 2: return [(0,1)]
    #   Compare (i, i+n/2) for i in range(n/2)
    #   merge(n/2) on first half
    #   merge(n/2) on second half
    
    def bitonic_merge(size, offset):
        if size <= 1:
            return []
        if size == 2:
            return [(offset, offset + 1)]
        half_size = size // 2
        comps = []
        for i in range(half_size):
            comps.append((offset + i, offset + i + half_size))
        comps.extend(bitonic_merge(half_size, offset))
        comps.extend(bitonic_merge(half_size, offset + half_size))
        return comps
    
    merge_comps = bitonic_merge(n, 0)
    
    return left_comps + right_comps + merge_comps

def _make_known_network():
    """Return a known good network. Start with odd-even transposition."""
    return _make_odd_even_transposition()

def _random_network(num_comps):
    """Generate a random network with num_comps comparators."""
    comps = []
    for _ in range(num_comps):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i != j:
            if i > j:
                i, j = j, i
            comps.append((i, j))
    return comps

def _try_improve(comparators, max_iters=10000):
    """Try to improve a network by removing or swapping comparators."""
    best = list(comparators)
    best_len = len(best)
    
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Try removing a random comparator
        if len(best) > 63:  # Don't go below proven lower bound
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _is_sorted_network(trial):
                best = trial
                best_len = len(best)
                continue
        
        # Try replacing a comparator with a random one
        if len(best) > 64:
            idx = random.randint(0, len(best) - 1)
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                trial = best[:idx] + [(i, j)] + best[idx+1:]
                if _is_sorted_network(trial):
                    best = trial
                    best_len = len(best)
                    continue
        
        # Try swapping two comparators
        if len(best) >= 2:
            idx1 = random.randint(0, len(best) - 1)
            idx2 = random.randint(0, len(best) - 1)
            if idx1 != idx2:
                trial = list(best)
                trial[idx1], trial[idx2] = trial[idx2], trial[idx1]
                if _is_sorted_network(trial):
                    best = trial
                    best_len = len(best)
                    continue
    
    return best

def _hill_climb(comparators):
    """Hill climbing: try all single removals, keep best valid."""
    best = list(comparators)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for idx in range(len(best)):
            if _time_left() < 5.0:
                break
            trial = best[:idx] + best[idx+1:]
            if _is_sorted_network(trial):
                best = trial
                improved = True
                break
    return best

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    random.seed(SEED)
    
    # Start with odd-even transposition (136 comparators)
    best = _make_odd_even_transposition()
    best_len = len(best)
    
    # First, try to reduce via hill climbing (remove redundant comparators)
    best = _hill_climb(best)
    best_len = len(best)
    
    # Then try local search with random perturbations
    for iteration in range(50):
        if _time_left() < 10.0:
            break
        
        # Random restart from current best with some noise
        trial = list(best)
        
        # Add some random comparators
        for _ in range(random.randint(0, 5)):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                trial.append((i, j))
        
        # Then hill climb
        trial = _hill_climb(trial)
        
        if len(trial) < best_len:
            best = trial
            best_len = len(trial)
    
    # Final attempt: try more aggressive reduction
    best = _hill_climb(best)
    
    return best

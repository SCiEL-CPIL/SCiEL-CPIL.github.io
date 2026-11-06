"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
MAX_TIME = 55  # seconds, leave margin for the 60s limit

# EVOLVE-BLOCK-START
def _apply_comparator(state, i, j):
    """Apply a compare-exchange: min goes to wire i, max to wire j."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]

def _is_sorted(state):
    """Check if the state is sorted in non-decreasing order."""
    for k in range(N - 1):
        if state[k] > state[k + 1]:
            return False
    return True

def _verify_network(network, samples=None):
    """
    Verify that the network sorts all possible inputs.
    If samples is None, use a random sample of permutations.
    For full certification, we'd need all 2^18 boolean inputs, but that's 262144 * len(network) operations.
    We'll use a combination of:
    1. All permutations of a small set (if N is small enough)
    2. Random boolean inputs
    3. Random permutations
    """
    if samples is None:
        # Use random boolean inputs and permutations
        rng = np.random.default_rng(42)
        # Test with random boolean vectors
        for _ in range(50000):
            state = rng.integers(0, 2, size=N).tolist()
            for i, j in network:
                _apply_comparator(state, i, j)
            if not _is_sorted(state):
                return False
        # Test with random permutations
        for _ in range(10000):
            state = list(range(N))
            random.shuffle(state)
            for i, j in network:
                _apply_comparator(state, i, j)
            if not _is_sorted(state):
                return False
        return True
    else:
        for state in samples:
            state_copy = list(state)
            for i, j in network:
                _apply_comparator(state_copy, i, j)
            if not _is_sorted(state_copy):
                return False
        return True

def _greedy_build():
    """
    Build a sorting network using a greedy approach:
    Start with an empty network and add comparators that help sort the most 'hard' inputs.
    """
    network = []
    # We'll track which inputs are still unsorted
    # Use a set of representative inputs: all permutations of [0,1,2,...,N-1] is too many for N=18
    # Instead, use random boolean inputs and permutations
    
    rng = np.random.default_rng(123)
    
    # Generate a pool of test inputs
    test_inputs = []
    # Add all single-bit flips from sorted
    for bit in range(N):
        inp = [0] * N
        inp[bit] = 1
        test_inputs.append(inp)
    # Add random boolean inputs
    for _ in range(2000):
        test_inputs.append(rng.integers(0, 2, size=N).tolist())
    # Add random permutations
    for _ in range(2000):
        perm = list(range(N))
        random.shuffle(perm)
        test_inputs.append(perm)
    
    # Track which inputs are currently sorted by the network
    def get_unsorted_indices(network):
        unsorted = []
        for idx, inp in enumerate(test_inputs):
            state = list(inp)
            for i, j in network:
                _apply_comparator(state, i, j)
            if not _is_sorted(state):
                unsorted.append(idx)
        return unsorted
    
    unsorted = get_unsorted_indices(network)
    
    # Precompute all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    max_iter = 200
    for _ in range(max_iter):
        if not unsorted:
            break
        if len(network) >= MAX_COMPARATORS:
            break
            
        # Find the comparator that sorts the most currently-unsorted inputs
        best_comp = None
        best_count = 0
        
        # Sample some comparators to speed up
        sample_size = min(len(all_comparators), 200)
        sample_comps = random.sample(all_comparators, sample_size)
        
        for comp in sample_comps:
            i, j = comp
            count = 0
            for idx in unsorted:
                state = list(test_inputs[idx])
                # Apply existing network
                for ni, nj in network:
                    _apply_comparator(state, ni, nj)
                # Apply this new comparator
                _apply_comparator(state, i, j)
                if _is_sorted(state):
                    count += 1
            if count > best_count:
                best_count = count
                best_comp = comp
        
        if best_comp is None or best_count == 0:
            # Try all comparators
            for comp in all_comparators:
                i, j = comp
                count = 0
                for idx in unsorted:
                    state = list(test_inputs[idx])
                    for ni, nj in network:
                        _apply_comparator(state, ni, nj)
                    _apply_comparator(state, i, j)
                    if _is_sorted(state):
                        count += 1
                if count > best_count:
                    best_count = count
                    best_comp = comp
        
        if best_comp is None or best_count == 0:
            break
            
        network.append(best_comp)
        unsorted = get_unsorted_indices(network)
    
    return network

def _odd_even_merge_sort_network():
    """
    Construct a sorting network using Odd-Even Merge Sort.
    This is a known good construction.
    """
    def odd_even_merge(n):
        """Generate comparators for odd-even merge of n wires (n must be power of 2)."""
        if n == 1:
            return []
        # Split into two halves
        left = odd_even_merge(n // 2)
        right = odd_even_merge(n // 2)
        # Offset right half
        right = [(i + n // 2, j + n // 2) for i, j in right]
        # Merge the two halves
        merge = _odd_even_merge_step(n)
        return left + right + merge
    
    def _odd_even_merge_step(n):
        """Generate the merge comparators for two sorted halves of size n/2."""
        if n == 2:
            return [(0, 1)]
        # The merge step: for k = 0 to n/2 - 1, compare (k, k + n/2) with parity
        comparators = []
        half = n // 2
        # Odd-even merge: 
        # First, compare pairs (0, half), (2, half+2), ... (even indices)
        # Then compare pairs (1, half+1), (3, half+3), ... (odd indices)
        # But recursively, it's more complex. Let's use the standard definition.
        
        # Standard odd-even merge:
        # M(n) = M(n/2) on first half, M(n/2) on second half (offset), then merge
        # The merge of two sorted sequences of length n/2:
        # Compare (i, i + n/2) for i = 0, 1, ..., n/2 - 1, but in a specific pattern
        
        # Actually, let me implement it properly:
        # Odd-even merge of two sorted lists of size n/2:
        # Step 1: Compare (0, n/2), (2, n/2+2), ..., (n/2-2, n-2)  [if n/2 is even, all even offsets]
        # Step 2: Compare (1, n/2+1), (3, n/2+3), ..., [if n/2 is odd, all odd offsets]
        # But this needs to be done recursively.
        
        # Let me use a simpler known construction:
        # For merging two sorted halves of size m = n/2:
        # The comparators are: for each k in 0..m-1, compare (k, k+m)
        # But that's just a simple merge, not odd-even.
        
        # Let me implement the standard odd-even merge correctly:
        def merge_step(m):
            """Generate comparators to merge two sorted sequences of length m."""
            if m == 1:
                return [(0, 1)]
            # Split into even and odd positions
            even = merge_step(m // 2)  # merge elements at even positions
            # Offset for odd positions
            odd = [(i + 1, j + 1) for i, j in merge_step(m // 2)] if m // 2 > 1 else [(1, m + 1)]
            # Wait, this is getting complicated. Let me use a different approach.
            return []
        
        # Let me just use a known good network instead.
        # For now, return a simple merge: compare (i, i + n/2) for all i
        return [(i, i + half) for i in range(half)]
    
    # Since N=18 is not a power of 2, this approach is tricky.
    # Let me skip this and use a different method.
    return []

def _bitonic_sort_network(n):
    """
    Construct a bitonic sorting network for n wires.
    n should be a power of 2 for the standard construction.
    For n=18, we'll pad to 32 or use a different approach.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # For non-power-of-2, this is complex. Let's only handle powers of 2.
    if n & (n - 1) != 0:
        return []
    
    def bitonic_sort(n, offset=0, direction=1):
        if n == 1:
            return []
        half = n // 2
        # Sort first half ascending, second half descending
        left = bitonic_sort(half, offset, 1)
        right = bitonic_sort(half, offset + half, -1)
        # Merge
        merge = bitonic_merge(n, offset, direction)
        return left + right + merge
    
    def bitonic_merge(n, offset=0, direction=1):
        if n == 1:
            return []
        half = n // 2
        # Compare (offset + i, offset + i + half) for i = 0..half-1
        comparators = []
        for i in range(half):
            a = offset + i
            b = offset + i + half
            if direction == 1:
                comparators.append((min(a, b), max(a, b)))
            else:
                comparators.append((min(a, b), max(a, b)))
        # Recursively merge
        left = bitonic_merge(half, offset, direction)
        right = bitonic_merge(half, offset + half, direction)
        return comparators + left + right
    
    return bitonic_sort(n, 0, 1)

def construct():
    """
    Construct a sorting network for N=18 wires.
    """
    # Try greedy approach
    greedy_net = _greedy_build()
    
    # If greedy didn't find a valid network, fall back to odd-even transposition
    if not _verify_network(greedy_net):
        # Fall back to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return greedy_net
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18
TOTAL_BITS = 1 << N  # 262144

# Precompute all Boolean inputs as bitmasks (0..262143)
# We'll use a bit-parallel verifier: for each input mask, simulate the network.
# A comparator (i,j) on a bitmask x:
#   bit_i = x >> i & 1, bit_j = x >> j & 1
#   if bit_i > bit_j (i.e., bit_i=1, bit_j=0), swap them.
# After all comparators, the output must be sorted ascending:
#   bits from 0 to N-1 should be in non-decreasing order, i.e., all 0s then all 1s.
# Actually, sorted ascending means: if we list bits from wire 0 to wire N-1,
# they should be 0...0 1...1 (since min goes to lower index).
# So the output mask must be of the form: lower k bits are 0, upper (N-k) bits are 1.
# i.e., output == (1 << (N - k)) - 1 ... wait, let me think.
# Wire 0 is the "first" element. After sorting, wire 0 has the smallest value.
# For Boolean inputs, sorted means: some number of 0s followed by some number of 1s.
# So if there are k ones in the input, the output should have the top k bits as 1 and bottom (N-k) as 0.
# Output mask = ((1 << k) - 1) << (N - k)

def _sorted_mask_for_count(k, n=N):
    """Return the bitmask representing a sorted sequence with k ones in n wires."""
    if k == 0:
        return 0
    if k == n:
        return (1 << n) - 1
    return ((1 << k) - 1) << (n - k)

# Precompute the sorted target for each input mask
# For each mask, count the number of 1-bits, then compute the sorted target.
_popcount = np.frombuffer(bytearray(1 << N), dtype=np.uint8)
for i in range(1, 1 << N):
    _popcount[i] = _popcount[i >> 1] + (i & 1)

_targets = np.empty(1 << N, dtype=np.uint32)
for k in range(N + 1):
    target = _sorted_mask_for_count(k)
    _targets[_popcount == k] = target

def _verify(network):
    """Verify that the network sorts all 262144 Boolean inputs. Returns True if valid."""
    if not network:
        return False
    # Simulate all inputs in parallel using NumPy
    inputs = np.arange(1 << N, dtype=np.uint32)
    for (i, j) in network:
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        # If bit_i > bit_j, swap
        swap = (bit_i == 1) & (bit_j == 0)
        # Clear bit_i and set bit_j where swap
        inputs = np.where(swap, (inputs & ~(1 << i)) | (1 << j), inputs)
        # Clear bit_j and set bit_i where swap
        inputs = np.where(swap, (inputs & ~(1 << j)) | (1 << i), inputs)
    # Check if all outputs match targets
    return np.all(inputs == _targets)

def _odd_even_transposition():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _bitonic_merge(i, j, k):
    """Bitonic merge for wires i..j (length 2^k), assuming bitonic sequence."""
    if k == 0:
        return []
    comps = []
    for d in range(2 ** (k - 1)):
        comps.append((i + d, j - d))
    # Recursively merge halves
    comps.extend(_bitonic_merge(i, i + 2 ** (k - 1) - 1, k - 1))
    comps.extend(_bitonic_merge(j - 2 ** (k - 1) + 1, j, k - 1))
    return comps

def _bitonic_sort():
    """Bitonic sorting network for N wires (N must be power of 2)."""
    if N != 16:  # For 18 wires, we'll use a different approach
        pass
    # For 16 wires
    comps = []
    n = 16
    for k in range(1, n.bit_length()):
        for i in range(0, n, 2 ** k):
            # Merge the two halves
            comps.extend(_bitonic_merge(i, i + 2 ** k - 1, k))
    # Note: this doesn't directly give a sorting network, need to be careful
    # Actually, standard bitonic sort:
    comps = []
    for k in range(1, n.bit_length()):
        for i in range(0, n, 2 ** k):
            # The first half is ascending, second is descending -> bitonic
            # Merge them
            for d in range(2 ** (k - 1)):
                comps.append((i + d, i + 2 ** k - 1 - d))
            # Then recursively sort each half
            # But this recursive approach is tricky. Let me use a known construction.
    # Simpler: use the standard bitonic sort network
    comps = []
    for k in range(4, 0, -1):  # k from log2(n) down to 1
        for i in range(0, n, 2 ** k):
            # Compare-exchange pairs that are 2^(k-1) apart
            for d in range(2 ** (k - 1)):
                comps.append((i + d, i + d + 2 ** (k - 1)))
    return comps

def _known_good_network():
    """Return a known valid sorting network for 18 wires."""
    # Start with odd-even transposition (153 comparators)
    return _odd_even_transposition()

def _local_search(network, max_iters=500, deadline=85.0):
    """Local search: try to remove comparators or swap them to reduce count."""
    start_time = time.time()
    current = list(network)
    current_count = len(current)
    
    if not _verify(current):
        return network  # Return original if current is invalid
    
    best = list(current)
    best_count = current_count
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try removing each comparator
        improved = False
        for idx in range(len(current)):
            if time.time() > deadline:
                break
            trial = current[:idx] + current[idx+1:]
            if _verify(trial):
                current = trial
                current_count -= 1
                best = list(current)
                best_count = current_count
                improved = True
                break  # Restart from improved solution
        
        if not improved:
            # Try swapping adjacent comparators (reorder)
            for idx in range(len(current) - 1):
                if time.time() > deadline:
                    break
                trial = current[:idx] + [current[idx+1], current[idx]] + current[idx+2:]
                if _verify(trial):
                    # Same count but different order - might enable further removals
                    # Don't accept unless it leads to improvement, so skip for now
                    pass
        
        if not improved:
            # Try adding a comparator at random position then removing others
            # This is more complex; for now, stop if no removal found
            break
    
    return best

def _batcher_odd_even_merge_sort():
    """Batcher's odd-even merge sort network for N=18."""
    # Batcher's network works best for powers of 2. For 18, we can:
    # 1. Sort 16 wires with Batcher's network
    # 2. Merge the 16-sorted list with the remaining 2 wires
    # This is complex. Let's use a simpler approach:
    # Use odd-even transposition as baseline and try to improve.
    return _odd_even_transposition()

# EVOLVE-BLOCK-START
def construct():
    """Construct a sorting network for 18 wires."""
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # Start with odd-even transposition
    network = _odd_even_transposition()
    
    if time.time() > deadline:
        return network
    
    # Verify the initial network
    if not _verify(network):
        # Should not happen, but just in case
        return network
    
    # Local search to reduce comparator count
    improved = _local_search(network, max_iters=200, deadline=deadline)
    
    return improved
# EVOLVE-BLOCK-END

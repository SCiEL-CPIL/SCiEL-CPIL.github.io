"""Sorting network for 18 wires: local search with bitwise verification."""
import time
import numpy as np

N = 18
NUM_INPUTS = 1 << N  # 262144

# Precompute input matrix: shape (NUM_INPUTS, N), each row is one boolean input
def _build_input_matrix():
    """Build (2^N, N) boolean matrix where row i has bits of i."""
    idx = np.arange(NUM_INPUTS, dtype=np.uint32)
    bits = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for b in range(N):
        bits[:, b] = (idx >> b) & 1
    return bits

INPUTS = _build_input_matrix()  # (262144, 18) uint8

# Precompute sorted reference: for each input, the sorted order is 0s then 1s
# A network is correct iff for every input, the output is sorted (non-decreasing along wires)
# i.e., output[:, i] <= output[:, i+1] for all i, for all inputs.

def _verify(network):
    """Check if network sorts all boolean inputs. Returns True if correct."""
    state = INPUTS.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        # min goes to i, max goes to j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    # Check sorted: state[:, k] <= state[:, k+1] for all k
    for k in range(N - 1):
        if np.any(state[:, k] > state[:, k + 1]):
            return False
    return True

def _bitonic_sort_network(n):
    """Generate bitonic sort network for n wires (n must be power of 2 or handle odd)."""
    # For non-power-of-2, we can still use bitonic merge patterns
    # Standard bitonic sort for n = 2^k
    # For n=18, use a hybrid: sort into groups and merge
    # Simpler: use the known bitonic construction for the next power of 2, then restrict
    # Actually, let's just generate a valid bitonic network for n wires directly.
    
    comparators = []
    
    def bitonic_merge(lo, hi, direction):
        """Bitonic merge on wires [lo, hi). direction: 1 for ascending, 0 for descending."""
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        # Compare elements that are 'half' apart
        for i in range(lo, hi - half):
            a = i
            b = i + half
            if direction == 1:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        # Recursively merge each half
        bitonic_merge(lo, lo + half, direction)
        bitonic_merge(lo + half, hi, direction)
    
    def bitonic_sort(lo, hi, direction):
        """Bitonic sort on wires [lo, hi)."""
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        bitonic_sort(lo, lo + half, 1)   # ascending
        bitonic_sort(lo + half, hi, 0)    # descending
        bitonic_merge(lo, hi, direction)
    
    bitonic_sort(0, N, 1)
    return comparators

def construct():
    deadline = time.time() + 80  # 80s limit, 10s safety margin
    
    # Start with bitonic sort network
    network = _bitonic_sort_network(N)
    
    # Verify it's correct (should be)
    if not _verify(network):
        # Fallback: use a known-correct simple network
        # Odd-even transposition
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(network)
    best_len = len(best_network)
    
    # Local search: try removing each comparator
    # Also try reordering and inserting
    
    # Phase 1: Greedy removal - repeatedly try to remove each comparator
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for idx in range(len(network)):
            if time.time() > deadline - 5:
                break
            candidate = network[:idx] + network[idx+1:]
            if _verify(candidate):
                network = candidate
                improved = True
                break  # Restart scan after removal
    
    best_network = list(network)
    best_len = len(best_network)
    
    # Phase 2: Try swapping adjacent comparators (reordering)
    # This can help when order matters for subsequent removals
    if time.time() < deadline - 15:
        for _ in range(20):  # Multiple passes
            if time.time() > deadline - 5:
                break
            found_swap = False
            for idx in range(len(network) - 1):
                if time.time() > deadline - 5:
                    break
                candidate = list(network)
                candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
                if _verify(candidate):
                    network = candidate
                    found_swap = True
                    break
            if not found_swap:
                break
    
    best_network = list(network)
    best_len = len(best_network)
    
    # Phase 3: Try inserting a comparator and then removing others
    # This explores a larger space
    if time.time() < deadline - 20:
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            # Try inserting a random comparator
            i = np.random.randint(0, N)
            j = np.random.randint(i+1, N)
            candidate = list(network)
            candidate.append((i, j))
            if _verify(candidate):
                network = candidate
                # Now try to remove comparators
                for idx in range(len(network) - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    test = network[:idx] + network[idx+1:]
                    if _verify(test):
                        network = test
                        break
    
    best_network = list(network)
    best_len = len(best_network)
    
    # Phase 4: More aggressive - try removing non-adjacent pairs
    if time.time() < deadline - 15:
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            i = np.random.randint(0, len(network))
            j = np.random.randint(0, len(network))
            if i == j:
                continue
            if i > j:
                i, j = j, i
            candidate = network[:i] + network[i+1:j] + network[j+1:]
            if len(candidate) < len(network) and _verify(candidate):
                network = candidate
    
    best_network = list(network)
    best_len = len(best_network)
    
    # Final verification
    if not _verify(best_network):
        # Emergency fallback
        best_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    return best_network

# Run construct to verify it works and measure time
if __name__ == "__main__":
    t0 = time.time()
    net = construct()
    t1 = time.time()
    print(f"Network size: {len(net)}, time: {t1-t0:.2f}s")
    print(f"Verified: {_verify(net)}")

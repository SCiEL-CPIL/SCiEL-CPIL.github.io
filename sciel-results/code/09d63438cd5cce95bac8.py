"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _batcher_odd_even_merge(n):
    """Generate Batcher odd-even merge sort network for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Recursively sort two halves
    left = _batcher_odd_even_merge(half)
    right = [(i + half, j + half) for (i, j) in left]
    # Odd-even merge: interleave sorted halves
    merge = _odd_even_merge(n)
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge network for n wires (power of 2, n >= 2)."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Recursively merge odd and even subsequences
    odd = _odd_even_merge(half)
    even = _odd_even_merge(half)
    # Shift odd subsequence by 1
    odd_shifted = [(i + 1, j + 1) for (i, j) in odd]
    # Compare-exchange between odd and even
    cross = [(i, i + 1) for i in range(0, n - 1, 2)]
    return odd + even + odd_shifted + cross

def _batcher_16():
    """Full Batcher OES network for 16 wires."""
    return _batcher_odd_even_merge(16)

def _verify_network_fast(comparators):
    """Verify sorting network using zero-one principle with numpy.
    Returns True if the network sorts all 2^16 Boolean inputs."""
    # Use all 2^16 = 65536 possible Boolean inputs
    all_inputs = np.arange(65536, dtype=np.uint8)
    # Create a matrix of shape (65536, 16) where each row is a bit pattern
    bits = ((all_inputs[:, None] >> np.arange(16)[None, :]) & 1).astype(np.int8)
    
    # Apply each comparator
    for (i, j) in comparators:
        # min goes to i, max goes to j
        mn = np.minimum(bits[:, i], bits[:, j])
        mx = np.maximum(bits[:, i], bits[:, j])
        bits[:, i] = mn
        bits[:, j] = mx
    
    # Check if all rows are sorted (non-decreasing)
    # A row is sorted if bits[:, k] <= bits[:, k+1] for all k
    diffs = np.diff(bits, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _verify_network(comparators):
    """Slower but simpler verification for debugging."""
    for mask in range(65536):
        bits = [(mask >> i) & 1 for i in range(N)]
        for (i, j) in comparators:
            if bits[i] > bits[j]:
                bits[i], bits[j] = bits[j], bits[i]
        if any(bits[k] > bits[k+1] for k in range(N-1)):
            return False
    return True

def _local_search(start_network, time_budget=60.0):
    """Local search to minimize comparator count while maintaining correctness."""
    deadline = time.time() + time_budget
    current = list(start_network)
    current_count = len(current)
    
    # Verify starting point
    if not _verify_network_fast(current):
        return current
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5.0:  # safety margin
        # Try different mutation strategies
        mutation_type = rng.choice(['remove', 'swap', 'add'])
        
        candidate = list(current)
        
        if mutation_type == 'remove' and len(candidate) > 50:
            # Remove a random comparator
            idx = rng.randrange(len(candidate))
            candidate.pop(idx)
            # Try to remove more in a batch
            extra_removals = rng.randint(1, 3)
            for _ in range(extra_removals):
                if len(candidate) > 50:
                    idx = rng.randrange(len(candidate))
                    candidate.pop(idx)
                    
        elif mutation_type == 'swap' and len(candidate) >= 2:
            # Swap two comparators
            i, j = rng.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
        elif mutation_type == 'add':
            # Add a random comparator
            i, j = sorted(rng.sample(range(N), 2))
            candidate.append((i, j))
            
        # Verify candidate
        if _verify_network_fast(candidate):
            current = candidate
            current_count = len(current)
        else:
            # If we removed too much, try a smaller removal
            if mutation_type == 'remove' and len(current) > 55:
                # Try removing just one
                candidate2 = list(current)
                idx = rng.randrange(len(candidate2))
                candidate2.pop(idx)
                if _verify_network_fast(candidate2):
                    current = candidate2
                    current_count = len(current)
    
    return current

def construct():
    # Start with Batcher OES network
    base = _batcher_16()
    
    # Verify base network
    if not _verify_network_fast(base):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Apply local search with time budget
    result = _local_search(base, time_budget=50.0)
    
    # Final verification
    if not _verify_network_fast(result):
        return base
    
    return result

# EVOLVE-BLOCK-END

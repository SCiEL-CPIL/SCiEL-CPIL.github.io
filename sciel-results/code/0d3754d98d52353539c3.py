"""Sorting network for 16 wires - optimized for minimal comparator count."""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 85  # seconds, with safety margin before 90s kill

def _batcher_oes(n):
    """Generate Batcher's odd-even merge sort network for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    # Recursively sort first half and second half
    left = _batcher_oes(k)
    right = _batcher_oes(k)
    right_shifted = [(i + k, j + k) for i, j in right]
    
    # Merge the two sorted halves
    merge = _batcher_merge(n)
    
    return left + right_shifted + merge

def _batcher_merge(n):
    """Generate Batcher's odd-even merge for n wires (two sorted n/2 sequences)."""
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    result = []
    
    # Odd-even merge: interleave and compare
    # For merging two sorted sequences of length k:
    # We need to do a butterfly-like pattern
    
    # Standard Batcher OES merge
    # For n elements, merge two sorted k-element sequences
    depth = int(np.log2(n))
    for d in range(1, depth + 1):
        step = 2 ** (depth - d)
        for i in range(0, n, 2 * step):
            for j in range(step):
                a = i + j
                b = i + j + step
                if a < b:
                    result.append((a, b))
    
    return result

def _is_sorted_vec(vec):
    """Check if a vector is non-decreasing."""
    for i in range(len(vec) - 1):
        if vec[i] > vec[i + 1]:
            return False
    return True

def _verify_network(network, sample_size=None):
    """Verify the network sorts all Boolean inputs (or a sample)."""
    if sample_size is None:
        # Full verification
        for mask in range(1 << N):
            vec = [(mask >> i) & 1 for i in range(N)]
            for i, j in network:
                if vec[i] > vec[j]:
                    vec[i], vec[j] = vec[j], vec[i]
            if not _is_sorted_vec(vec):
                return False
        return True
    else:
        # Sample verification
        for _ in range(sample_size):
            mask = random.getrandbits(N)
            vec = [(mask >> i) & 1 for i in range(N)]
            for i, j in network:
                if vec[i] > vec[j]:
                    vec[i], vec[j] = vec[j], vec[i]
            if not _is_sorted_vec(vec):
                return False
        return True

def construct():
    """
    Return a fixed compare-exchange network that sorts all 16-input sequences.
    Each comparator is a tuple (i, j) with 0 <= i < j < N.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Generate Batcher's OES network
    network = _batcher_oes(N)
    
    # Verify it's correct
    if not _verify_network(network):
        # Fallback to odd-even transposition
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    best_network = network[:]
    best_count = len(network)
    
    # Try to remove redundant comparators
    # Use a greedy approach: try removing each comparator and check if still valid
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing comparators from end to start
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            temp_network = best_network[:idx] + best_network[idx+1:]
            
            # Quick sample check first
            if not _verify_network(temp_network, sample_size=2000):
                continue
            
            # Full verification
            if _verify_network(temp_network):
                best_network = temp_network
                best_count = len(best_network)
                improved = True
                break  # Restart the removal process
        
        if not improved:
            break
    
    # Final verification
    if not _verify_network(best_network):
        # Should not happen, but fallback
        best_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    return best_network

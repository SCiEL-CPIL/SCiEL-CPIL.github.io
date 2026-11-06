"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 60  # seconds, with safety margin

# Precompute a known good network (odd-even transposition) as fallback
def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# Cache for the best network found
_best_network = None
_network_computed = False

def _is_sorted_network(network):
    """Verify the network sorts all 2^16 Boolean inputs using zero-one principle."""
    if not network:
        return False
    # For each of the 2^16 Boolean inputs, check if output is sorted
    for mask in range(1 << N):
        # Create input vector
        vec = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        # Apply comparators
        for i, j in network:
            if vec[i] > vec[j]:
                vec[i], vec[j] = vec[j], vec[i]
        # Check if sorted (non-decreasing)
        for i in range(N - 1):
            if vec[i] > vec[i + 1]:
                return False
    return True

def _apply_network(vec, network):
    """Apply the network to a vector and return the result."""
    vec = list(vec)
    for i, j in network:
        if vec[i] > vec[j]:
            vec[i], vec[j] = vec[j], vec[i]
    return vec

def _is_sorted_vec(vec):
    """Check if a vector is non-decreasing."""
    for i in range(len(vec) - 1):
        if vec[i] > vec[i + 1]:
            return False
    return True

def _construct_improved():
    """
    Attempt to find a more efficient sorting network using a greedy/evolutionary approach.
    Uses the zero-one principle for verification.
    """
    global _best_network, _network_computed
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # 5 second safety margin
    
    # Start with odd-even transposition
    current_network = _odd_even_transposition()
    current_count = len(current_network)
    _best_network = current_network[:]
    
    # Try to remove redundant comparators using a greedy approach
    # For each comparator, check if removing it still yields a valid sorting network
    # This is expensive, so we'll use a smarter approach
    
    # Better approach: Use a known construction or try local search
    # For 16 wires, the optimal is known to be around 60 comparators
    
    # Let's try a systematic reduction using the zero-one principle
    # We'll use a randomized local search to try to remove comparators
    
    import random
    
    best_network = current_network[:]
    best_count = current_count
    
    # Try removing comparators one at a time in random order
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        network = best_network[:]
        removed = False
        
        # Try to remove each comparator
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Temporarily remove this comparator
            temp_network = network[:idx] + network[idx+1:]
            
            # Quick check: does this still sort a random sample of inputs?
            # Use a smaller sample for speed
            valid = True
            for _ in range(1000):
                mask = random.getrandbits(N)
                vec = [(mask >> i) & 1 for i in range(N)]
                for i, j in temp_network:
                    if vec[i] > vec[j]:
                        vec[i], vec[j] = vec[j], vec[i]
                if not _is_sorted_vec(vec):
                    valid = False
                    break
            
            if valid:
                # Verify with full zero-one principle
                if _is_sorted_network(temp_network):
                    network = temp_network
                    best_network = network[:]
                    best_count = len(network)
                    removed = True
        
        if not removed:
            # Try adding and reordering (skip for now, focus on removal)
            # If no more can be removed, try a different approach
            break
    
    # Final verification
    if _is_sorted_network(best_network):
        _best_network = best_network[:]
    
    _network_computed = True
    return _best_network

def construct():
    """
    Return a fixed compare-exchange network that sorts all 16-input sequences.
    Each comparator is a tuple (i, j) with 0 <= i < j < N.
    """
    global _best_network, _network_computed
    
    if _network_computed:
        return _best_network
    
    # Try to find an improved network
    try:
        network = _construct_improved()
    except Exception:
        network = _odd_even_transposition()
    
    _best_network = network
    _network_computed = True
    return network
# EVOLVE-BLOCK-END

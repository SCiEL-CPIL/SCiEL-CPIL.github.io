"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
N = 18

# EVOLVE-BLOCK-START
def _odd_even_transposition_network():
    """Construct odd-even transposition sorting network."""
    comparators = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comparators.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comparators.append((i, i + 1))
    return comparators

def _bitonic_merge_network(lo, hi):
    """Generate comparators for bitonic merge of [lo, hi)."""
    if hi - lo <= 1:
        return []
    mid = (lo + hi) // 2
    result = []
    # Compare lo+i with mid+i for i in range(hi-lo)
    for i in range(hi - lo):
        result.append((lo + i, mid + i))
    # Recursively merge halves
    result += _bitonic_merge_network(lo, mid)
    result += _bitonic_merge_network(mid, hi)
    return result

def _bitonic_sort_network(lo, hi, direction='up'):
    """Generate comparators for bitonic sort of [lo, hi)."""
    if hi - lo <= 1:
        return []
    mid = (lo + hi) // 2
    result = []
    # Sort first half up, second half down
    result += _bitonic_sort_network(lo, mid, 'up')
    result += _bitonic_sort_network(mid, hi, 'down')
    # Merge
    if direction == 'up':
        result += _bitonic_merge_network(lo, hi)
    else:
        # For down direction, reverse the merge
        merge = _bitonic_merge_network(lo, hi)
        result += [(j, i) for (i, j) in merge]
    return result

def _construct_bitonic_18():
    """Construct a bitonic sorting network specifically for 18 wires."""
    # Pad to 32 wires, build bitonic network, then extract relevant comparators
    n_pad = 32
    comparators = []
    
    # Standard bitonic sort construction
    for size in range(2, n_pad + 1, 2):
        half = size // 2
        for start in range(0, n_pad, size):
            # Sort first half ascending
            for s in range(2, half + 1, 2):
                sh = s // 2
                for st in range(start, start + half, s):
                    for i in range(sh):
                        comparators.append((st + i, st + i + sh))
            # Sort second half descending
            for s in range(2, half + 1, 2):
                sh = s // 2
                for st in range(start + half, start + size, s):
                    for i in range(sh):
                        comparators.append((st + i + sh, st + i))
            # Merge
            for i in range(half):
                comparators.append((start + i, start + half + i))
    
    # Filter to only wires < N
    comparators = [(i, j) for (i, j) in comparators if i < N and j < N]
    return comparators

def _verify_network_bitparallel(comparators):
    """Verify network using bit-parallel simulation over all 2^N Boolean inputs."""
    n = N
    num_inputs = 1 << n
    
    # Represent all inputs as a matrix: num_inputs x n bits
    # Use numpy for efficiency
    inputs = np.zeros((num_inputs, n), dtype=np.uint8)
    for i in range(num_inputs):
        for j in range(n):
            inputs[i, j] = (i >> j) & 1
    
    # Apply each comparator
    for i, j in comparators:
        # For each input, if state[i] > state[j], swap
        # state[i] > state[j] means state[i]=1 and state[j]=0
        mask = (inputs[:, i] == 1) & (inputs[:, j] == 0)
        # Swap where mask is true
        inputs[mask, i] = 0
        inputs[mask, j] = 1
    
    # Check if all outputs are sorted
    # For each input, check that no adjacent pair is inverted
    for k in range(n - 1):
        inverted = (inputs[:, k] == 1) & (inputs[:, k + 1] == 0)
        if np.any(inverted):
            return False
    
    return True

def _verify_network_fast(comparators, max_tests=50000):
    """Fast verification using random Boolean inputs."""
    for _ in range(max_tests):
        input_seq = [random.randint(0, 1) for _ in range(N)]
        state = input_seq[:]
        for i, j in comparators:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        for k in range(N - 1):
            if state[k] > state[k + 1]:
                return False
    return True

def _prune_comparators(comparators, deadline):
    """Prune redundant comparators using iterative removal."""
    if len(comparators) == 0:
        return comparators
    
    best = comparators[:]
    n = len(comparators)
    changed = True
    
    while changed and time.time() < deadline:
        changed = False
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing comparator at idx
            candidate = best[:idx] + best[idx + 1:]
            
            # Quick verification
            if _verify_network_fast(candidate, max_tests=3000):
                best = candidate
                n -= 1
                changed = True
    
    return best

def _local_search(comparators, deadline):
    """Try local search to improve the network."""
    best = comparators[:]
    best_size = len(best)
    
    # Try swapping adjacent comparators
    while time.time() < deadline:
        improved = False
        for idx in range(len(best) - 1):
            if time.time() > deadline:
                break
            
            # Try swapping comparator idx and idx+1
            candidate = best[:]
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            
            if _verify_network_fast(candidate, max_tests=2000):
                best = candidate
                improved = True
        
        if not improved:
            break
    
    return best

def construct():
    """Construct a sorting network for N=18 wires."""
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Start with odd-even transposition as baseline
    comparators = _odd_even_transposition_network()
    baseline_size = len(comparators)
    
    # Try bitonic network
    bitonic = _construct_bitonic_18()
    if len(bitonic) < len(comparators):
        if _verify_network_fast(bitonic, max_tests=10000):
            comparators = bitonic
    
    # Verify current best
    if not _verify_network_fast(comparators, max_tests=20000):
        # Fallback
        return _odd_even_transposition_network()
    
    # Prune redundant comparators
    if time.time() < deadline - 5:
        comparators = _prune_comparators(comparators, deadline - 5)
    
    # Local search
    if time.time() < deadline - 2:
        comparators = _local_search(comparators, deadline - 2)
    
    # Final verification
    if not _verify_network_fast(comparators, max_tests=50000):
        # Fallback
        return _odd_even_transposition_network()
    
    return comparators
# EVOLVE-BLOCK-END


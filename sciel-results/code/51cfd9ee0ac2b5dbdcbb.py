"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 17
_DEADLINE = 80  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
# Pre-compute expected sorted masks for each possible popcount
_EXPECTED_MASKS = {}
def _init_expected_masks():
    """Pre-compute expected sorted masks for all possible popcounts."""
    for k in range(N + 1):
        # k ones at the rightmost positions
        if k == 0:
            _EXPECTED_MASKS[k] = 0
        else:
            _EXPECTED_MASKS[k] = (1 << k) - 1
_init_expected_masks()

def _popcount(mask):
    """Count set bits in mask."""
    return bin(mask).count('1')

def _verify_fast(network):
    """Fast verification using zero-one principle with pre-computed expected masks."""
    for mask in range(1 << N):
        for i, j in network:
            vi = (mask >> i) & 1
            vj = (mask >> j) & 1
            if vi > vj:
                mask |= (1 << i)
                mask &= ~(1 << j)
            elif vi < vj:
                mask &= ~(1 << i)
                mask |= (1 << j)
        # Check if mask is in sorted form: all 1s at the right
        k = _popcount(mask)
        if mask != _EXPECTED_MASKS[k]:
            return False
    return True

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _find_redundant_indices(network):
    """Find indices of comparators that appear more than once (redundant candidates)."""
    from collections import Counter
    counter = Counter(network)
    redundant = []
    for idx, comp in enumerate(network):
        if counter[comp] > 1:
            redundant.append(idx)
    return redundant

def construct():
    start_time = time.time()
    network = _seed_network()
    best = list(network)
    best_len = len(best)
    
    # Verify seed
    if not _verify_fast(network):
        return best
    
    # Multiple passes of greedy removal with different strategies
    passes = 0
    max_passes = 20  # safety cap
    
    while passes < max_passes:
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE:
            break
        
        # Strategy 1: Try removing redundant comparators first
        redundant = _find_redundant_indices(network)
        for idx in sorted(redundant, reverse=True):
            if time.time() - start_time > _DEADLINE:
                break
            candidate = list(network)
            candidate.pop(idx)
            if _verify_fast(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
        
        # Strategy 2: Reverse-order removal
        for attempt in range(len(network) - 1, -1, -1):
            if time.time() - start_time > _DEADLINE:
                break
            candidate = list(network)
            candidate.pop(attempt)
            if _verify_fast(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
        
        # Strategy 3: Forward-order removal
        i = 0
        while i < len(network):
            if time.time() - start_time > _DEADLINE:
                break
            candidate = list(network)
            candidate.pop(i)
            if _verify_fast(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
            else:
                i += 1
        
        # Check if we made any progress this pass
        if len(network) == best_len:
            # No progress, try one more pass with shuffled order
            if time.time() - start_time > _DEADLINE - 5:
                break
            indices = list(range(len(network)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > _DEADLINE:
                    break
                candidate = list(network)
                candidate.pop(idx)
                if _verify_fast(candidate):
                    network = candidate
                    best = list(candidate)
                    best_len = len(best)
        
        passes += 1
    
    return best
# EVOLVE-BLOCK-END

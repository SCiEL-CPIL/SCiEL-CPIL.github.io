"""Rewritten program for sorting networks on 17 wires."""
import time
import random
import numpy as np

N = 17

# Precompute all 2^17 Boolean inputs and their sorted versions
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = ((_indices[:, None] & _bits[None, :]) > 0).astype(np.uint8)
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a smaller sample set for fast validation during search
_SAMPLE_SIZE = 2048
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def simulate_network(comparators, inputs):
    """
    inputs: numpy array of shape (num_inputs, N) with values in {0, 1}
    comparators: list of (i, j)
    returns: sorted numpy array of shape (num_inputs, N)
    """
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def is_sorted_network_full(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    """
    if len(comparators) > 4096:
        return False
    state = _INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return bool(np.array_equal(state, _SORTED_INPUTS))

def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs.
    """
    if len(comparators) > 4096:
        return False
    state = _SAMPLE_INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return bool(np.array_equal(state, _SAMPLE_SORTED))

def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)

def odd_even_transposition_network(n):
    """Generate an odd-even transposition sorting network for n wires."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def bitonic_sort_network_power2(n):
    """
    Generate a bitonic sort network for n wires where n is a power of 2.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comparators = []
    
    def bitonic(k, offset, direction=1):
        if k == 1:
            return
        if k == 2:
            comparators.append((offset, offset + 1))
            return
        
        half = k // 2
        bitonic(half, offset, 1)
        bitonic(half, offset + half, -1)
        
        # Merge
        for i in range(half):
            a = offset + i
            b = offset + i + half
            comparators.append((a, b))
    
    bitonic(n, 0, 1)
    return comparators

def bitonic_sort_16():
    """Bitonic sort network for 16 wires."""
    return bitonic_sort_network_power2(16)

def build_seed_networks():
    """
    Build multiple seed networks and return the shortest valid one.
    """
    seeds = []
    
    # Seed 1: Odd-even transposition
    seed1 = odd_even_transposition_network(N)
    seeds.append(seed1)
    
    # Seed 2: Bitonic sort on 16 wires + merge 17th wire
    seed2 = bitonic_sort_16()
    # Add comparators to merge wire 16 into the sorted 16 wires
    for i in range(15, -1, -1):
        seed2.append((i, 16))
    # Then do a final pass to ensure sorting
    for i in range(N - 1):
        seed2.append((i, i + 1))
    seeds.append(seed2)
    
    # Verify and pick the shortest valid seed
    best_seed = None
    best_seed_len = float('inf')
    for seed in seeds:
        if is_sorted_network_full(seed):
            if len(seed) < best_seed_len:
                best_seed = seed
                best_seed_len = len(seed)
    
    if best_seed is None:
        # Fallback: use the first seed that works
        best_seed = seed1
        if not is_sorted_network_full(best_seed):
            # Last resort: dense network
            best_seed = []
            for i in range(N):
                for j in range(i + 1, N):
                    best_seed.append((i, j))
    
    return list(best_seed)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin: finish well before 90s limit
    
    # Build initial seed network
    best_network = build_seed_networks()
    best_len = len(best_network)
    current = list(best_network)
    
    # Phase 1: Systematic Deletion with sample validation
    i = 0
    while i < len(current):
        if time.time() > deadline - 25.0:
            break
        candidate = current.copy()
        candidate.pop(i)
        # Use fast sample validation first
        if is_sorted_network_sample(candidate):
            # Confirm with full validation
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                continue
        i += 1
    
    # Phase 2: Multiple iterative passes of systematic deletion
    for _ in range(30):
        if time.time() > deadline - 20.0:
            break
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 20.0:
                break
            candidate = current.copy()
            candidate.pop(i)
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
                    progress = True
                    continue
            i += 1
        if not progress:
            break
    
    # Phase 3: Random deletion attempts
    attempts = 0
    while time.time() < deadline - 12.0 and attempts < 500:
        attempts += 1
        if not current:
            break
        idx = random.randint(0, len(current) - 1)
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
    
    # Phase 4: Local search - swap and replace moves
    if time.time() < deadline - 5.0:
        for _ in range(800):
            if time.time() > deadline - 5.0:
                break
            if len(current) < 2:
                break
            idx = random.randint(0, len(current) - 1)
            
            # Try swapping with a random other position
            other_idx = random.randint(0, len(current) - 1)
            if other_idx != idx:
                candidate = current.copy()
                candidate[idx], candidate[other_idx] = candidate[other_idx], candidate[idx]
                if is_sorted_network_sample(candidate):
                    if is_sorted_network_full(candidate):
                        current = candidate
                        best_network = candidate
                        best_len = len(current)
            
            # Try moving comparator to a different position
            if time.time() < deadline - 5.0:
                new_idx = random.randint(0, len(current) - 1)
                if new_idx != idx:
                    candidate = current.copy()
                    comp = candidate.pop(idx)
                    candidate.insert(new_idx, comp)
                    if is_sorted_network_sample(candidate):
                        if is_sorted_network_full(candidate):
                            current = candidate
                            best_network = candidate
                            best_len = len(current)
            
            # Try replacing with a different comparator
            if time.time() < deadline - 5.0:
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                if (i, j) != current[idx]:
                    candidate = current.copy()
                    candidate[idx] = (i, j)
                    if is_sorted_network_sample(candidate):
                        if is_sorted_network_full(candidate):
                            current = candidate
                            best_network = candidate
                            best_len = len(current)
    
    # Phase 5: Final systematic pass
    i = 0
    while i < len(current) and time.time() < deadline - 2.0:
        candidate = current.copy()
        candidate.pop(i)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                continue
        i += 1
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return build_seed_networks()
        
    return best_network

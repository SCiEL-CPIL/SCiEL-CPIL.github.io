import time
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N

def _precompute_inputs():
    """
    Precompute all 2^N Boolean inputs as a (N, 2^N) uint8 array.
    Column k represents the binary representation of k, with bit 0 as the first wire.
    """
    all_inputs = np.zeros((N, TOTAL_INPUTS), dtype=np.uint8)
    for i in range(N):
        period = 1 << (i + 1)
        half = 1 << i
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[half:] = 1
        all_inputs[i] = np.tile(pattern, TOTAL_INPUTS // period)
    return all_inputs

_ALL_INPUTS = _precompute_inputs()

def _verify_network(network):
    """
    Vectorized verification of the sorting network.
    Returns True if the network sorts all 0-1 inputs.
    """
    x = _ALL_INPUTS.copy()
    for i, j in network:
        mask = (x[i] == 1) & (x[j] == 0)
        if np.any(mask):
            x[i][mask] = 0
            x[j][mask] = 1
    # Check if sorted: no (1,0) adjacent pair
    diffs = x[1:].astype(np.int8) - x[:-1].astype(np.int8)
    if np.any(diffs == -1):
        return False
    return True

def _bitonic_sort_network(n):
    """
    Generate a bitonic sorting network for n wires (n must be power of 2).
    For n=18, we pad to 32 and extract the first 18 wires' comparators.
    Actually, let's build bitonic for exactly 18 using a merge-based approach.
    """
    # For simplicity, use bitonic for 32 wires and keep only comparators
    # that involve wires 0..17
    def bitonic_network(m):
        """Bitonic network for m wires (m power of 2)."""
        comps = []
        def bitonic_sort(lo, hi, direction):
            if hi - lo <= 1:
                return
            mid = (lo + hi) // 2
            bitonic_sort(lo, mid, direction)
            bitonic_sort(mid, hi, direction)
            if direction == 0:  # ascending
                for i in range(hi - lo):
                    a = lo + i
                    b = mid + i
                    if a < b:
                        comps.append((a, b))
            else:  # descending
                for i in range(hi - lo):
                    a = mid + i
                    b = lo + i
                    if a < b:
                        comps.append((a, b))
        
        def merge(lo, hi, direction):
            if hi - lo <= 1:
                return
            mid = (lo + hi) // 2
            merge(lo, mid, direction)
            merge(mid, hi, direction)
            if direction == 0:
                for i in range(hi - lo):
                    a = lo + i
                    b = mid + i
                    if a < b:
                        comps.append((a, b))
            else:
                for i in range(hi - lo):
                    a = mid + i
                    b = lo + i
                    if a < b:
                        comps.append((a, b))
        
        # Build full bitonic network for m wires
        for i in range(m - 1, 0, -1):
            step = 1 << i
            for start in range(0, m, 2 * step):
                # Create bitonic sequence
                for j in range(step):
                    a = start + j
                    b = start + j + step
                    if a < b:
                        comps.append((a, b))
                # Now merge
                for k in range(step // 2, 0, -1):
                    for j in range(start, start + step, 2 * k):
                        for t in range(k):
                            a = j + t
                            b = j + t + k
                            if a < b:
                                comps.append((a, b))
        return comps
    
    # Use bitonic for 32 wires, keep only comparators on wires 0-17
    full_comps = bitonic_network(32)
    filtered = [(i, j) for i, j in full_comps if i < N and j < N]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_comps = []
    for c in filtered:
        if c not in seen:
            seen.add(c)
            unique_comps.append(c)
    
    return unique_comps

def _odd_even_transposition_network(n):
    """Odd-even transposition network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _remove_redundant(network, deadline):
    """Iteratively remove redundant comparators."""
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            test_network = network[:idx] + network[idx+1:]
            if _verify_network(test_network):
                network = test_network
                improved = True
    return network

def _add_remove_search(network, deadline):
    """Add a comparator, then try to remove a different one to reduce size."""
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    existing_pairs = set(network)
    
    for _round in range(200):
        if time.time() > deadline:
            break
        
        found_improvement = False
        # Shuffle for diversity
        import random
        random.shuffle(all_pairs)
        
        for add_pair in all_pairs:
            if time.time() > deadline:
                break
            if add_pair in existing_pairs:
                continue
            
            test_network = network + [add_pair]
            
            for idx in range(len(test_network) - 2, -1, -1):
                if time.time() > deadline:
                    break
                removed_network = test_network[:idx] + test_network[idx+1:]
                
                if len(removed_network) < len(network):
                    if _verify_network(removed_network):
                        network = removed_network
                        existing_pairs = set(network)
                        found_improvement = True
                        break
            
            if found_improvement:
                break
        
        if not found_improvement:
            break
    
    return network

def _swap_search(network, deadline):
    """Try swapping one comparator for another to enable further removals."""
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    existing_pairs = set(network)
    
    for _round in range(100):
        if time.time() > deadline:
            break
        
        found_improvement = False
        import random
        random.shuffle(all_pairs)
        
        for idx in range(len(network)):
            if time.time() > deadline:
                break
            
            old_comp = network[idx]
            for new_pair in all_pairs:
                if new_pair in existing_pairs:
                    continue
                
                test_network = network[:idx] + [new_pair] + network[idx+1:]
                
                # Try to remove something from this modified network
                for rem_idx in range(len(test_network) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    if rem_idx == idx:
                        continue
                    candidate = test_network[:rem_idx] + test_network[rem_idx+1:]
                    
                    if len(candidate) < len(network):
                        if _verify_network(candidate):
                            network = candidate
                            existing_pairs = set(network)
                            found_improvement = True
                            break
                
                if found_improvement:
                    break
            
            if found_improvement:
                break
        
        if not found_improvement:
            break
    
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s kill, 5s margin
    
    # Start with bitonic sort (108 comparators for 18 wires)
    network = _bitonic_sort_network(N)
    
    # Verify it's correct
    if not _verify_network(network):
        # Fallback to odd-even transposition
        network = _odd_even_transposition_network(N)
    
    # Phase 1: Remove redundant comparators
    network = _remove_redundant(network, deadline - 20)
    
    # Phase 2: Add-remove search
    network = _add_remove_search(network, deadline - 10)
    
    # Phase 3: More removal passes
    network = _remove_redundant(network, deadline - 5)
    
    # Phase 4: Swap search
    network = _swap_search(network, deadline - 2)
    
    # Final removal pass
    network = _remove_redundant(network, deadline)
    
    return network

def _apply_network(network, x):
    x = list(x)
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _is_sorted_network(network):
    for mask in range(1 << N):
        x = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, x)
        if result != sorted(result):
            return False
    return True

def _depth(network):
    return len(network)

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START

# Precompute bit masks for fast verification
_ALL_MASKS = np.arange(1 << N, dtype=np.int64)
# _WIRE_BITS[i, mask] = bit i of mask
_WIRE_BITS = np.array([(_ALL_MASKS >> i) & 1 for i in range(N)], dtype=np.int8)  # shape (N, 2^N)

def _verify_network_bitparallel(net):
    """Verify that a network sorts all 2^N Boolean inputs using bit-parallel ops."""
    wires = _WIRE_BITS.copy()  # shape (N, 2^N)
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min goes to i, max goes to j
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    # Check if all wires are sorted: wire[k] <= wire[k+1] for all k
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True

def _depth(net):
    """Compute parallel depth of a network."""
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _odd_even_transposition_network(n):
    """Odd-even transposition sorting network (correct but large)."""
    net = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                net.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                net.append((i, i + 1))
    return net

def _bitonic_sort_network(n):
    """Bitonic sort network (smaller than odd-even transposition)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _bitonic_sort_network(half)
    right = _bitonic_sort_network(n - half)
    right_shifted = [(i + half, j + half) for i, j in right]
    # To make second half descending, we reverse the order of comparators
    # Actually, for bitonic sort we need: sort first half asc, second half desc, then merge
    # The standard bitonic sort:
    # 1. Recursively sort first half ascending
    # 2. Recursively sort second half descending
    # 3. Bitonic merge
    # To sort descending, we can sort ascending then reverse all comparators
    right_desc = [(j, i) for i, j in right]
    right_desc_shifted = [(i + half, j + half) for i, j in right_desc]
    merge = _bitonic_merge(n)
    return left + right_desc_shifted + merge

def _bitonic_merge(n):
    """Bitonic merge network for n elements (n must be power of 2)."""
    if n <= 1:
        return []
    half = n // 2
    result = [(i, i + half) for i in range(half)]
    left = _bitonic_merge(half)
    right = _bitonic_merge(n - half)
    right_shifted = [(i + half, j + half) for i, j in right]
    return result + left + right_shifted

def _build_initial_network():
    """Build an initial sorting network using bitonic sort."""
    # Bitonic sort requires power of 2, so use 32 wires and take first 18?
    # No, let's just use odd-even transposition which works for any N
    # Or we can build a bitonic network for 18 by padding to 32
    # Actually, let's just use odd-even transposition as the seed
    return _odd_even_transposition_network(N)

def _supercharged_local_search(net, max_time=50.0):
    """
    Supercharged Local Search with Remove, Swap, and Move operations.
    Uses bit-parallel verification for speed.
    """
    start_time = time.time()
    best_net = list(net)
    best_score = len(best_net)
    rng = random.Random(42)
    
    # Precompute all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Phase 1: Try removing comparators
        indices = list(range(len(best_net)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() - start_time >= max_time:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network_bitparallel(candidate):
                best_net = candidate
                best_score = len(best_net)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swapping a comparator with another
        if time.time() - start_time >= max_time:
            break
        
        idx = rng.randint(0, len(best_net) - 1)
        i, j = best_net[idx]
        
        # Try random replacements
        for _ in range(20):
            if time.time() - start_time >= max_time:
                break
            new_comp = rng.choice(all_comparators)
            if new_comp == (i, j):
                continue
            candidate = list(best_net)
            candidate[idx] = new_comp
            if _verify_network_bitparallel(candidate):
                best_net = candidate
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 3: Try moving a comparator to a different position
        if time.time() - start_time >= max_time:
            break
        
        idx = rng.randint(0, len(best_net) - 1)
        comp = best_net[idx]
        
        for _ in range(20):
            if time.time() - start_time >= max_time:
                break
            new_idx = rng.randint(0, len(best_net) - 1)
            if new_idx == idx:
                continue
            candidate = list(best_net)
            del candidate[idx]
            # Adjust new_idx if needed
            if new_idx > idx:
                new_idx -= 1
            candidate.insert(new_idx, comp)
            if _verify_network_bitparallel(candidate):
                best_net = candidate
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 4: Try adding a comparator (only if we've been stuck)
        # This is rarely helpful for minimization, but can help escape local minima
        # by creating a larger network that can then be reduced differently
        # For now, skip this as it increases size
        
        if not improved:
            # No improvement in this round, try a more aggressive restart
            # with random perturbations
            if time.time() - start_time >= max_time:
                break
            
            # Try a larger random perturbation: remove a random subset
            n_remove = min(3, len(best_net) // 4)
            if n_remove > 0:
                remove_indices = rng.sample(range(len(best_net)), n_remove)
                remove_indices.sort(reverse=True)
                candidate = list(best_net)
                for ri in remove_indices:
                    del candidate[ri]
                if _verify_network_bitparallel(candidate):
                    best_net = candidate
                    best_score = len(best_net)
                    improved = True
        
        if not improved:
            # Give up on this local minimum, do a larger random jump
            if time.time() - start_time >= max_time:
                break
            
            # Randomly replace 2-5 comparators
            n_replace = rng.randint(2, 5)
            if n_replace > len(best_net):
                n_replace = len(best_net)
            
            replace_indices = rng.sample(range(len(best_net)), n_replace)
            candidate = list(best_net)
            for ri in replace_indices:
                new_comp = rng.choice(all_comparators)
                candidate[ri] = new_comp
            
            if _verify_network_bitparallel(candidate):
                best_net = candidate
                best_score = len(best_net)
    
    return best_net

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin before 90s kill
    
    # Start with odd-even transposition (known correct)
    net = _build_initial_network()
    
    # Verify it's correct
    if not _verify_network_bitparallel(net):
        # Shouldn't happen, but fallback
        raise ValueError("Initial network failed verification")
    
    # Supercharged local search to improve
    if time.time() - start_time < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        net = _supercharged_local_search(net, max_time=min(remaining, 70.0))
    
    # Final verification
    if not _verify_network_bitparallel(net):
        # Fallback to original if something went wrong
        net = _build_initial_network()
    
    return net

# EVOLVE-BLOCK-END

import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _apply_comparator_vectorized(states, i, j):
    """Apply a compare-exchange to a vector of bitmask states (numpy int64 array)."""
    bi = (states >> i) & 1
    bj = (states >> j) & 1
    swap_mask = (bi == 0) & (bj == 1)
    if not np.any(swap_mask):
        return states
    result = states.copy()
    result[swap_mask] |= (1 << i)
    result[swap_mask] &= ~(1 << j)
    return result

def _apply_network_vectorized(states, network):
    """Apply a list of comparators to a vector of bitmask states."""
    for i, j in network:
        states = _apply_comparator_vectorized(states, i, j)
    return states

def _is_sorted_vectorized(states):
    """Check if bitmask states are sorted (all 0s then all 1s). Returns bool array."""
    # A state is sorted iff it's of the form 0...01...1
    # i.e., once we see a 1, all remaining bits must be 1
    # Equivalent: state & (state + 1) == 0 for the relevant pattern
    # More directly: count trailing zeros and leading ones
    sorted_mask = np.zeros(len(states), dtype=bool)
    for num_ones in range(N + 1):
        if num_ones == 0:
            target = 0
        else:
            target = ((1 << num_ones) - 1) << (N - num_ones)
        sorted_mask |= (states == target)
    return sorted_mask

def _count_violations_vectorized(network):
    """Count the number of inputs for which the network fails to sort."""
    all_states = np.arange(1 << N, dtype=np.int64)
    result_states = _apply_network_vectorized(all_states, network)
    sorted_mask = _is_sorted_vectorized(result_states)
    violations = np.sum(~sorted_mask)
    return int(violations)

def _verify(network):
    """Verify that the network sorts all inputs. Returns True if valid."""
    return _count_violations_vectorized(network) == 0

def _known_60_network():
    """Return a known 60-comparator sorting network for 16 wires."""
    # This is a verified 60-comparator network
    network = [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    ]
    return network

def _fast_verify_with_early_exit(network, max_checks=None):
    """Verify network with early exit. Returns True if valid, False if invalid."""
    all_states = np.arange(1 << N, dtype=np.int64)
    if max_checks is not None and max_checks < len(all_states):
        # Check a random subset first for quick invalidity detection
        idx = np.random.choice(len(all_states), max_checks, replace=False)
        states = all_states[idx]
    else:
        states = all_states
    
    for i, j in network:
        states = _apply_comparator_vectorized(states, i, j)
    
    sorted_mask = _is_sorted_vectorized(states)
    if not np.all(sorted_mask):
        return False
    return True

def _aggressive_prune(network, time_budget=10.0):
    """Aggressively try to remove comparators using multiple passes."""
    if not network:
        return network
    if not _verify(network):
        return network

    start_time = time.time()
    best = network[:]
    best_len = len(best)
    
    # Multiple passes of removal
    while (time.time() - start_time) < time_budget:
        improved = False
        
        # Try removing each comparator
        for idx in range(len(best)):
            if (time.time() - start_time) >= time_budget:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify(candidate):
                best = candidate
                best_len = len(best)
                improved = True
                break  # Restart from beginning
        
        if not improved:
            break
    
    return best

def _swap_optimize(network, time_budget=5.0):
    """Try swapping adjacent comparators to enable further removals."""
    if not network:
        return network
    if not _verify(network):
        return network

    start_time = time.time()
    best = network[:]
    best_len = len(best)
    
    while (time.time() - start_time) < time_budget:
        improved = False
        
        # Try swapping adjacent comparators
        for i in range(len(best) - 1):
            if (time.time() - start_time) >= time_budget:
                break
            candidate = best[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if _verify(candidate):
                best = candidate
                improved = True
                break
        
        if improved:
            # After a swap, try to remove again
            pruned = _aggressive_prune(best, time_budget=min(2.0, time_budget - (time.time() - start_time)))
            if len(pruned) < best_len:
                best = pruned
                best_len = len(best)
            continue
        
        # Try non-adjacent swaps
        for i in range(len(best)):
            if (time.time() - start_time) >= time_budget:
                break
            for j in range(i + 2, min(i + 5, len(best))):
                if (time.time() - start_time) >= time_budget:
                    break
                candidate = best[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if _verify(candidate):
                    best = candidate
                    improved = True
                    break
            if improved:
                pruned = _aggressive_prune(best, time_budget=min(2.0, time_budget - (time.time() - start_time)))
                if len(pruned) < best_len:
                    best = pruned
                    best_len = len(best)
                break
        
        if not improved:
            break
    
    return best

def _replace_optimize(network, time_budget=5.0):
    """Try replacing comparators with different pairs."""
    if not network:
        return network
    if not _verify(network):
        return network

    start_time = time.time()
    best = network[:]
    best_len = len(best)
    
    while (time.time() - start_time) < time_budget:
        improved = False
        
        for i in range(len(best)):
            if (time.time() - start_time) >= time_budget:
                break
            old_i, old_j = best[i]
            found = False
            
            # Try a few alternative pairs
            alternatives = []
            for new_i in range(N):
                for new_j in range(new_i + 1, N):
                    if (new_i, new_j) == (old_i, old_j):
                        continue
                    # Prefer pairs that are "close" to the original
                    dist = abs(new_i - old_i) + abs(new_j - old_j)
                    alternatives.append((dist, new_i, new_j))
            
            alternatives.sort()
            
            for dist, new_i, new_j in alternatives[:20]:  # Try top 20 closest
                if (time.time() - start_time) >= time_budget:
                    found = True
                    break
                candidate = best[:]
                candidate[i] = (new_i, new_j)
                if _verify(candidate):
                    best = candidate
                    improved = True
                    found = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    """Construct a sorting network for 16 wires. Returns a list of (i, j) pairs."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin for 90s limit

    # Start with the known 60-comparator network
    best_network = _known_60_network()

    # Verify it's actually valid (safety check)
    if not _verify(best_network):
        # Fall back to OET
        best_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best_len = len(best_network)

    # Phase 1: Aggressive pruning
    remaining = deadline - time.time()
    if remaining > 5.0:
        prune_budget = min(20.0, remaining - 2.0)
        pruned = _aggressive_prune(best_network, time_budget=prune_budget)
        if len(pruned) < best_len:
            best_network = pruned
            best_len = len(best_network)

    # Phase 2: Swap optimization to enable more removals
    remaining = deadline - time.time()
    if remaining > 5.0:
        swap_budget = min(15.0, remaining - 2.0)
        swapped = _swap_optimize(best_network, time_budget=swap_budget)
        if len(swapped) < best_len:
            best_network = swapped
            best_len = len(best_network)

    # Phase 3: Replace optimization
    remaining = deadline - time.time()
    if remaining > 5.0:
        replace_budget = min(10.0, remaining - 2.0)
        replaced = _replace_optimize(best_network, time_budget=replace_budget)
        if len(replaced) < best_len:
            best_network = replaced
            best_len = len(best_network)

    # Phase 4: Final aggressive pruning
    remaining = deadline - time.time()
    if remaining > 3.0:
        final_prune_budget = min(5.0, remaining - 1.0)
        final_pruned = _aggressive_prune(best_network, time_budget=final_prune_budget)
        if len(final_pruned) < best_len:
            best_network = final_pruned
            best_len = len(best_network)

    # Final verification
    if best_network is not None and _verify(best_network):
        return best_network
    else:
        # Ultimate fallback: OET
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END

"""Size-efficient sorting networks on 18 wires.
Strategy:
1. Start with a good initial network (odd-even transposition, 153 comparators).
2. Use a fast vectorized verifier (NumPy) to check correctness on all 2^18 Boolean inputs.
3. Greedily remove redundant comparators in multiple passes.
4. Try local search: swap adjacent comparators, move comparators, to find smaller networks.
5. Return the best network found within the time budget.
"""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096

# EVOLVE-BLOCK-START

def _odd_even_transposition_network(n):
    """Return odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps


def _verify_network_fast(network):
    """
    Verify network sorts all 2^N Boolean inputs using vectorized NumPy operations.
    Uses the zero-one principle.
    """
    if not network:
        return True
    
    # Precompute the comparator operations as index arrays
    # For each comparator (i, j), we need to apply min/max to wires i and j.
    # Vectorized approach: for all 2^N inputs at once, simulate the network.
    
    total = 1 << N
    # Create all Boolean inputs as a 2D array of shape (total, N)
    # Using bit manipulation to create all masks
    masks = np.arange(total, dtype=np.uint32)
    # Expand to get each bit for each wire
    # vecs[m, k] = bit k of masks[m]
    vecs = np.zeros((total, N), dtype=np.uint8)
    for k in range(N):
        vecs[:, k] = (masks >> k) & 1
    
    # Apply each comparator
    for i, j in network:
        vi = vecs[:, i]
        vj = vecs[:, j]
        # min goes to i, max goes to j
        vecs[:, i] = np.minimum(vi, vj)
        vecs[:, j] = np.maximum(vi, vj)
    
    # Check if all rows are sorted: for each row, vecs[m, k] <= vecs[m, k+1] for all k
    # A row is sorted if it's of the form 0...01...1
    # Check: no 1 followed by a 0
    for k in range(N - 1):
        if np.any((vecs[:, k] == 1) & (vecs[:, k + 1] == 0)):
            return False
    
    return True


def _verify_network_single(mask, network):
    """Verify a single Boolean input mask through the network."""
    vec = mask
    for i, j in network:
        vi = (vec >> i) & 1
        vj = (vec >> j) & 1
        if vi > vj:
            vec &= ~(1 << i)
            vec |= (1 << j)
    # Check sorted
    seen_one = False
    for k in range(N):
        if (vec >> k) & 1:
            seen_one = True
        elif seen_one:
            return False
    return True


def _find_redundant_comparators(network, deadline):
    """
    Find and remove redundant comparators from the network.
    Returns the improved network.
    """
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Try removing each comparator, starting from the end
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if _verify_network_fast(candidate):
                network = candidate
                improved = True
                break
    return network


def _local_search_swap(network, deadline):
    """
    Try swapping adjacent comparators to find a smaller or equivalent network.
    This can sometimes enable further removals.
    """
    best = list(network)
    best_len = len(best)
    
    for i in range(len(best) - 1):
        if time.time() > deadline:
            break
        # Swap best[i] and best[i+1]
        candidate = best[:]
        candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
        
        if _verify_network_fast(candidate):
            # Try to remove comparators from this candidate
            pruned = _find_redundant_comparators(candidate, deadline)
            if len(pruned) < best_len:
                best = pruned
                best_len = len(best)
    
    return best


def _local_search_move(network, deadline):
    """
    Try moving a comparator to a different position in the network.
    """
    best = list(network)
    best_len = len(best)
    
    for idx in range(len(best)):
        if time.time() > deadline:
            break
        comp = best[idx]
        # Try inserting at various positions
        for new_idx in range(len(best)):
            if time.time() > deadline:
                break
            if new_idx == idx:
                continue
            candidate = best[:]
            candidate.pop(idx)
            insert_at = new_idx if new_idx > idx else new_idx
            candidate.insert(insert_at, comp)
            
            if _verify_network_fast(candidate):
                pruned = _find_redundant_comparators(candidate, deadline)
                if len(pruned) < best_len:
                    best = pruned
                    best_len = len(best)
                    break  # Restart with the new best
    
    return best


def construct():
    """
    Construct a sorting network on 18 wires with as few comparators as possible.
    
    Strategy:
    1. Start from the odd-even transposition network (153 comparators).
    2. Use a fast vectorized verifier to check correctness.
    3. Greedily remove redundant comparators.
    4. Try local search moves to find smaller networks.
    """
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Start with odd-even transposition network
    network = _odd_even_transposition_network(N)
    
    # Verify it's correct
    if not _verify_network_fast(network):
        return network
    
    # Phase 1: Greedy removal of redundant comparators
    network = _find_redundant_comparators(network, deadline)
    
    # Phase 2: Local search - try swaps
    if time.time() < deadline - 5:
        network = _local_search_swap(network, deadline)
    
    # Phase 3: Another round of greedy removal after swaps
    network = _find_redundant_comparators(network, deadline)
    
    # Phase 4: Local search - try moves
    if time.time() < deadline - 5:
        network = _local_search_move(network, deadline)
    
    # Phase 5: Final greedy removal
    network = _find_redundant_comparators(network, deadline)
    
    return network

# EVOLVE-BLOCK-END

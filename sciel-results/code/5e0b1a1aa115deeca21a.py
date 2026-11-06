"""
Size-efficient sorting networks on 17 wires.
Structural change: SAT-style iterative removal using bitmask validation.
1. Start from odd-even transposition network.
2. Iteratively attempt to remove comparators in a smart order (redundant first).
3. Use bitmask-based validation for speed.
4. When stuck, try adding/removing comparators to escape local minima.
5. Track best network found and return it before deadline.
"""
import time
import random
import math

N = 17
TOTAL_INPUTS = 1 << N  # 131072

# Precompute initial state bitmasks for verification.
# state[i] is a bitmask where bit k is 1 if input k has a 1 at wire i.
INITIAL_STATE = [0] * N
for i in range(N):
    mask = 0
    block = 1 << i
    val = 0
    for m in range(block, TOTAL_INPUTS, 2 * block):
        bits_to_set = (1 << block) - 1
        val |= bits_to_set << m
    INITIAL_STATE[i] = val


def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1
    
    st = list(INITIAL_STATE)
    n = N
    
    for i, j in network:
        si = st[i]
        sj = st[j]
        st[i] = si & sj
        st[j] = si | sj
        
    for i in range(n - 1):
        if st[i] & ~st[i + 1]:
            return False
            
    return True


def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _compute_depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    depth = [0] * N
    for i, j in network:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)


def _try_remove_comparator(network, idx, deadline):
    """Try removing comparator at idx. Return new network if valid, else None."""
    if len(network) <= 1:
        return None
    candidate = network[:idx] + network[idx+1:]
    if _is_valid(candidate):
        return candidate
    return None


def _find_redundant_comparators(network, deadline):
    """
    Find all redundant comparators (those that can be removed while keeping validity).
    Returns a list of indices that are redundant.
    """
    redundant = []
    n = len(network)
    # Try removing each comparator
    for idx in range(n):
        if time.time() > deadline:
            break
        candidate = network[:idx] + network[idx+1:]
        if _is_valid(candidate):
            redundant.append(idx)
    return redundant


def _greedy_remove_all_redundant(network, deadline):
    """
    Iteratively remove redundant comparators until none remain.
    Returns the reduced network.
    """
    current = list(network)
    
    while time.time() < deadline:
        redundant = _find_redundant_comparators(current, deadline)
        if not redundant:
            break
        
        # Remove the first redundant comparator found
        idx = redundant[0]
        current = current[:idx] + current[idx+1:]
        
        if len(current) <= 1:
            break
            
    return current


def _random_local_search(network, deadline):
    """
    Perform random local search to escape local minima.
    Tries to add, remove, or modify comparators.
    Returns the best network found.
    """
    current = list(network)
    best = list(current)
    best_len = len(current)
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to remove a random comparator
            if len(current) > 20:
                idx = random.randint(0, len(current) - 1)
                candidate = current[:idx] + current[idx+1:]
                if _is_valid(candidate):
                    current = candidate
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
                        
        elif move_type < 0.5:
            # Try to add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(current))
            candidate = current[:idx] + [(i, j)] + current[idx:]
            if _is_valid(candidate):
                current = candidate
                
        elif move_type < 0.7:
            # Try to modify a comparator's wires
            if len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate = list(current)
                candidate[idx] = (i, j)
                if _is_valid(candidate):
                    current = candidate
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
                        
        elif move_type < 0.85:
            # Try to swap two adjacent comparators
            if len(current) >= 2:
                idx = random.randint(0, len(current) - 2)
                candidate = list(current)
                candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
                if _is_valid(candidate):
                    current = candidate
                    
        else:
            # Try to move a comparator to a different position
            if len(current) >= 2:
                idx = random.randint(0, len(current) - 1)
                new_idx = random.randint(0, len(current) - 1)
                if idx != new_idx:
                    candidate = list(current)
                    comp = candidate.pop(idx)
                    candidate.insert(new_idx, comp)
                    if _is_valid(candidate):
                        current = candidate
        
        # Periodically try greedy removal
        if random.random() < 0.1:
            reduced = _greedy_remove_all_redundant(current, min(time.time() + 2, deadline))
            if len(reduced) < best_len:
                best = list(reduced)
                best_len = len(reduced)
                current = list(reduced)
                
    return best


def construct():
    """
    Construct a sorting network with as few comparators as possible.
    Uses a combination of greedy removal and random local search.
    """
    start_time = time.time()
    # 90s limit, keep 10s margin for safety
    deadline = start_time + 78.0
    
    # Start with Odd-Even Transposition
    current_network = _construct_odd_even_transposition()
    best_network = list(current_network)
    best_len = len(current_network)
    
    # Verify start
    if not _is_valid(current_network):
        return best_network
    
    # Phase 1: Aggressive greedy removal
    greedy_deadline = min(start_time + 30.0, deadline)
    current_network = _greedy_remove_all_redundant(current_network, greedy_deadline)
    
    if len(current_network) < best_len:
        best_network = list(current_network)
        best_len = len(current_network)
    
    # Phase 2: Random local search to escape local minima
    if time.time() < deadline:
        local_search_deadline = deadline
        improved_network = _random_local_search(current_network, local_search_deadline)
        
        if len(improved_network) < best_len:
            best_network = list(improved_network)
            best_len = len(improved_network)
    
    return best_network


# EVOLVE-BLOCK-END

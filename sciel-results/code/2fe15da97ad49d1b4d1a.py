# EVOLVE-BLOCK-START
import time
import random

N = 18
_INITIAL_BITSETS = None

def _init_bitsets():
    """
    Initialize 18 bitsets, each of length 2^N.
    bitset[k] has bit i set if the i-th input (0 to 2^N-1) has a 1 in wire k.
    """
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is not None:
        return _INITIAL_BITSETS
        
    total_bits = 1 << N
    bitsets = [0] * N
    
    for k in range(N):
        period = 1 << (k + 1)
        half_period = 1 << k
        # Mask for one period: lower half 0s, upper half 1s
        pattern = (1 << half_period) - 1
        pattern <<= half_period
        
        # Repeat pattern to fill total_bits
        current_val = pattern
        current_len = period
        
        while current_len < total_bits:
            current_val |= (current_val << current_len)
            current_len *= 2
            
        bitsets[k] = current_val

    _INITIAL_BITSETS = bitsets
    return bitsets

def _is_sorting_network(comparators):
    """
    Check if network sorts all 2^N Boolean inputs using bitset simulation.
    """
    if not comparators:
        return N <= 1
        
    # Get a copy of the initial bitsets
    bitsets = list(_init_bitsets())
    
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        bitsets[i] = bi & bj
        bitsets[j] = bi | bj
        
    # Check if sorted:
    # For all k < l, bitsets[k] must be a subset of bitsets[l].
    # It is sufficient to check adjacent wires: k and k+1.
    for k in range(N - 1):
        if bitsets[k] & ~bitsets[k + 1]:
            return False
            
    return True

def _greedy_reduce(comparators, deadline):
    """
    Greedy reduction: try removing each comparator, keep if valid.
    """
    current = list(comparators)
    improved = True
    # Limit passes to avoid infinite loops if time is tight
    max_passes = 5
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        i = 0
        while i < len(current):
            if time.time() > deadline:
                return current
            
            # Try removing comparator at i
            saved = current[i]
            current.pop(i)
            
            if _is_sorting_network(current):
                improved = True
            else:
                # Restore
                current.insert(i, saved)
                i += 1
        
        if not improved:
            break
            
    return current

def _local_search(comparators, deadline):
    """
    Local search: try removing, swapping, and moving comparators.
    """
    current = list(comparators)
    best = list(current)
    best_size = len(current)
    
    # Try to remove any redundant comparators first
    current = _greedy_reduce(current, deadline)
    best = list(current)
    best_size = len(current)
    
    # Simple hill climbing / local search
    # We will try to remove more, or swap to enable removals.
    # For time efficiency, we focus on removals and simple swaps.
    
    stall_count = 0
    max_stall = 50
    
    while time.time() < deadline:
        if stall_count > max_stall:
            break
            
        improved = False
        n_comps = len(current)
        
        # Try removing a random comparator
        if n_comps > 0 and random.random() < 0.5:
            idx = random.randint(0, n_comps - 1)
            saved = current[idx]
            current.pop(idx)
            if _is_sorting_network(current):
                improved = True
                best = list(current)
                best_size = len(current)
            else:
                current.insert(idx, saved)
        
        # Try swapping two comparators
        elif n_comps > 1:
            i = random.randint(0, n_comps - 2)
            j = random.randint(i + 1, n_comps - 1)
            current[i], current[j] = current[j], current[i]
            if _is_sorting_network(current):
                # Check if we can remove something now
                # Just accept the swap as neutral/improvement in structure
                # We don't update best unless size decreases, but we keep current
                # to explore.
                pass
            else:
                current[i], current[j] = current[j], current[i]
        
        if not improved:
            stall_count += 1
        else:
            stall_count = 0
            
    return best

def construct():
    # Start with odd-even transposition network
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    start_time = time.time()
    # Safety margin: 55s limit
    deadline = start_time + 55
    
    best = list(seed_network)
    
    # 1. Initial Greedy reduction
    best = _greedy_reduce(best, deadline)
    
    # 2. Local Search
    # Run local search for remaining time
    candidate = _local_search(best, deadline)
    if len(candidate) < len(best):
        best = candidate
        
    # 3. Final Greedy Polish
    if time.time() < deadline - 1:
        final_deadline = deadline - 1
        best = _greedy_reduce(best, final_deadline)
        
    return best
# EVOLVE-BLOCK-END

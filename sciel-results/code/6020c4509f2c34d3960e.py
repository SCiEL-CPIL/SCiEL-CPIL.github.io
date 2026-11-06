"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute all Boolean inputs and sorted targets once at module load
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = ((_indices[:, None] >> _bits[None, :]) & 1).astype(np.uint8)
_SORTED_TARGETS = np.sort(_INPUTS, axis=1)

def simulate_network(comparators, inputs):
    """Simulate compare-exchange network. Returns output array."""
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs."""
    if len(comparators) > 4096:
        return False
    if len(comparators) == 0:
        return False
    output = simulate_network(comparators, _INPUTS)
    # Check if output matches sorted targets
    return np.array_equal(output, _SORTED_TARGETS)

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Uses the recursive odd-even merge construction.
    Returns list of (i, j) comparator pairs.
    """
    if n == 1:
        return []
    
    def _build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        # Recursively sort first half and second half (offset)
        left = _build(half)
        right = [(i + half, j + half) for (i, j) in _build(n - half)]
        
        # Odd-even merge
        merged = _odd_even_merge(n)
        
        return left + right + merged
    
    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        comparators = []
        
        if n % 2 == 0:
            # Even case: compare adjacent pairs starting from 0
            for i in range(0, n, 2):
                comparators.append((i, i + 1))
            # Now recursively merge: even positions form one sub-problem, odd form another
            even_merged = _odd_even_merge(n // 2)
            for (a, b) in even_merged:
                comparators.append((2 * a, 2 * b))
            odd_merged = _odd_even_merge(n // 2)
            for (a, b) in odd_merged:
                comparators.append((2 * a + 1, 2 * b + 1))
        else:
            # Odd case: compare (1,2),(3,4),...
            for i in range(1, n - 1, 2):
                comparators.append((i, i + 1))
            # Element 0 stays, element n-1 stays
            # Recursively merge the middle: positions 1..n-2 (n-2 elements)
            if n - 2 > 0:
                mid_merged = _odd_even_merge(n - 2)
                for (a, b) in mid_merged:
                    comparators.append((a + 1, b + 1))
        
        return comparators
    
    return _build(n)

def improve_by_removal(comparators, deadline):
    """
    Iteratively try to remove each comparator if the network remains correct.
    """
    best = list(comparators)
    
    i = 0
    while i < len(best):
        if time.time() > deadline - 3:
            break
        candidate = best[:i] + best[i+1:]
        if is_sorted_network(candidate):
            best = candidate
            # Don't increment i, re-check same position
        else:
            i += 1
    
    return best

def try_insert_and_remove(best, deadline):
    """
    Try inserting a random comparator at a random position, then removing redundancies.
    """
    if time.time() > deadline - 5:
        return best
    
    best_len = len(best)
    attempts = 0
    max_attempts = 200
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Insert a random comparator at a random position
        pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        # Try to remove comparators from the candidate
        k = 0
        improved = True
        while k < len(candidate) and improved:
            improved = False
            if time.time() > deadline - 3:
                break
            test = candidate[:k] + candidate[k+1:]
            if is_sorted_network(test):
                candidate = test
                improved = True
                # Don't increment k
            else:
                k += 1
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Use odd-even merge sort as the base network (much smaller than transposition)
    base = odd_even_merge_sort_network(N)
    
    # Verify base is correct
    if not is_sorted_network(base):
        # Fallback to odd-even transposition
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = list(base)
    best_len = len(best)
    
    # Phase 1: Remove redundant comparators
    if time.time() < deadline - 15:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Phase 2: Simulated Annealing / Local Search to escape local minima
    # This is the structural change: using SA with move operations (swap, move, replace, add, remove)
    if time.time() < deadline - 10:
        current = best.copy()
        current_len = best_len
        
        T = 5.0 # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995
        
        while time.time() < deadline - 10:
            if T < T_min:
                break
                
            # Generate a neighbor
            move_type = random.randint(0, 4)
            neighbor = current.copy()
            
            if move_type == 0: # Remove
                if not neighbor:
                    continue
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                
            elif move_type == 1: # Add
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                idx = random.randint(0, len(neighbor))
                neighbor.insert(idx, (i, j))
                
            elif move_type == 2: # Swap
                if len(neighbor) < 2:
                    continue
                i1 = random.randint(0, len(neighbor) - 1)
                i2 = random.randint(0, len(neighbor) - 1)
                if i1 != i2:
                    neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
                    
            elif move_type == 3: # Move
                if len(neighbor) < 2:
                    continue
                i1 = random.randint(0, len(neighbor) - 1)
                i2 = random.randint(0, len(neighbor) - 1)
                if i1 != i2:
                    item = neighbor.pop(i1)
                    neighbor.insert(i2, item)
                    
            else: # Replace
                if not neighbor:
                    continue
                idx = random.randint(0, len(neighbor) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                neighbor[idx] = (i, j)

            # Check validity
            if is_sorted_network(neighbor):
                neighbor_len = len(neighbor)
                delta_e = neighbor_len - current_len
                
                if delta_e <= 0:
                    # Improvement or equal
                    current = neighbor
                    current_len = neighbor_len
                    if current_len < best_len:
                        best = current.copy()
                        best_len = current_len
                else:
                    # Worse, accept with probability
                    prob = np.exp(-delta_e / T)
                    if random.random() < prob:
                        current = neighbor
                        current_len = neighbor_len
            else:
                # Invalid, reject
                pass
                
            T *= cooling_rate
            
        # If we found something better via SA, update best
        if current_len < best_len:
            best = current.copy()
            best_len = current_len
    
    # Phase 3: Another pass of removal to clean up after SA
    if time.time() < deadline - 5:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Phase 4: Final removal pass
    if time.time() < deadline - 3:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        # Should not happen, but fallback
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
# EVOLVE-BLOCK-END

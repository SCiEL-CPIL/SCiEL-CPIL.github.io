import time
import random

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85.0  # Safety margin from 90s limit

# Pre-compute masks at module load time to avoid doing it in construct()
_INITIAL_MASKS = [0] * N
for _w in range(N):
    block = 1 << _w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = (1 << N) // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    _INITIAL_MASKS[_w] = m

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _batcher_odd_even_merge(n):
    """
    Generate an odd-even merge network for n wires (n must be a power of 2).
    Returns list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    result = []
    
    def _oem(k, offset):
        """Odd-even merge of k elements starting at offset."""
        if k <= 1:
            return
        half_k = k // 2
        # Recursively merge even and odd subsequences
        # Even positions: offset, offset+2, ...
        _oem(half_k, offset)
        # Odd positions: offset+1, offset+3, ...
        _oem(half_k, offset + 1)
        # Compare adjacent pairs
        for i in range(offset, offset + k - 1):
            result.append((i, i + 1))
            
    _oem(n, 0)
    return result

def _batcher_odd_even_mergesort(n):
    """
    Generate Batcher's odd-even mergesort network for n wires.
    n must be a power of 2.
    """
    if n <= 1:
        return []
    half = n // 2
    # Recursively sort two halves
    left = _batcher_odd_even_mergesort(half)
    right = _batcher_odd_even_mergesort(half)
    right = [(i + half, j + half) for i, j in right]
    # Odd-even merge the two sorted halves
    merge = _batcher_odd_even_merge(n)
    return left + right + merge

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _build_initial_network():
    """
    Build a good initial sorting network.
    Strategy: Use Batcher's network for the next power of 2 >= N, 
    then project it down to N wires by removing comparators involving 
    wires >= N.
    """
    p = 1
    while p < N:
        p *= 2
        
    # Generate full network for p wires
    full_network = _batcher_odd_even_mergesort(p)
    
    # Project to N wires: keep only comparators where both indices < N
    projected = [(i, j) for i, j in full_network if i < N and j < N]
    
    # If projection fails or yields empty (shouldn't happen), fallback
    if not projected:
        return _odd_even_transposition()
        
    return projected

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Build initial network (Batcher projected)
    current = _build_initial_network()
    best = current[:]
    best_len = len(best)
    
    # Initial verification
    if not _verify(best):
        # Fallback to odd-even transposition if Batcher projection failed (unlikely)
        current = _odd_even_transposition()
        best = current[:]
        best_len = len(best)
    
    # Phase 1: Greedy removal of redundant comparators
    # This quickly reduces the size of the initial Batcher network
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing current[i]
            neighbor = current[:i] + current[i+1:]
            if _verify(neighbor):
                current = neighbor
                changed = True
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
                break  # Restart from beginning after a change to allow cascading removals
    
    # Phase 2: Simulated Annealing
    # Key improvement: Use 'insert_delete' moves to escape local minima
    # where no single removal is valid, but swapping one comparator for another
    # (possibly at a different position) enables further reductions.
    
    T = 10.0
    T_min = 0.05
    cooling_rate = 0.995
    
    rng = random.Random(42)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Pre-compute indices for faster random access
    all_comp_idx = {comp: idx for idx, comp in enumerate(all_comparators)}
    
    while time.time() < deadline and T > T_min:
        # Weighted move selection
        # 'insert_delete' is crucial for structural changes
        # 'remove' for direct optimization
        # 'swap' and 'move' for reordering (less critical if we have insert_delete)
        move_type = rng.choices(
            ['remove', 'insert_delete', 'swap', 'move'],
            weights=[3, 4, 2, 1]
        )[0]
        
        neighbor = current[:]
        valid_move = True
        
        if move_type == 'remove' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            neighbor.pop(idx)
            
        elif move_type == 'insert_delete' and len(neighbor) > 0:
            # Remove one comparator
            idx_remove = rng.randrange(len(neighbor))
            removed_comp = neighbor.pop(idx_remove)
            
            # Add a different comparator at a random position
            # Avoid adding the same comparator back immediately if possible, 
            # but random choice is fine.
            new_comp = rng.choice(all_comparators)
            idx_add = rng.randrange(len(neighbor) + 1)
            neighbor.insert(idx_add, new_comp)
            
        elif move_type == 'swap' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else:
                valid_move = False # No change, skip verification
                
        elif move_type == 'move' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                comp = neighbor.pop(i)
                if i < j:
                    j -= 1
                neighbor.insert(j, comp)
            else:
                valid_move = False

        if valid_move:
            if _verify(neighbor):
                delta = len(neighbor) - len(current)
                # Accept if better or with probability exp(-delta/T)
                # delta > 0 means neighbor is larger (worse)
                if delta <= 0:
                    current = neighbor
                    if len(current) < best_len:
                        best = current[:]
                        best_len = len(current)
                else:
                    # Metropolis criterion
                    if rng.random() < 2.718281828459045 ** (-delta / T):
                        current = neighbor
                        # We don't update best here if it's worse, 
                        # but we might recover later.
                        # However, keeping 'best' as the global minimum found so far is safer.

        T *= cooling_rate
        
    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best

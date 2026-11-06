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

def _batcher_odd_even_merge(n):
    """
    Generate an odd-even merge network for n wires (n must be a power of 2).
    Returns list of (i, j) comparators.
    """
    if n <= 1:
        return []
    half = n // 2
    # Recursively merge two halves
    left = _batcher_odd_even_merge(half)
    right = _batcher_odd_even_merge(half)
    # Shift right half by half
    right = [(i + half, j + half) for i, j in right]
    # Interleave: odd-even merge
    merged = []
    # Phase 1: compare pairs (0,1), (2,3), ... within each half
    for i in range(0, n - 1, 2):
        merged.append((i, i + 1))
    # Phase 2: compare pairs (1,2), (3,4), ... 
    for i in range(1, n - 1, 2):
        merged.append((i, i + 1))
    # Actually, proper odd-even merge:
    # Step 1: recursively sort the two halves
    # Step 2: compare (0,1), (2,3), ..., (n-2, n-1)
    # Step 3: compare (1,2), (3,4), ..., (n-3, n-2)
    # But we need to be more careful. Let me use the standard formulation.
    
    # Standard odd-even merge for power of 2:
    # 1. Recursively odd-even merge the even-indexed elements
    # 2. Recursively odd-even merge the odd-indexed elements  
    # 3. Compare (i, i+1) for all i
    
    # For simplicity, use a known-correct construction
    result = []
    
    def _oem(k, offset):
        """Odd-even merge of k elements starting at offset."""
        if k <= 1:
            return
        half_k = k // 2
        # Recursively merge even and odd subsequences
        _oem(half_k, offset)  # even positions: offset, offset+2, ...
        _oem(half_k, offset + 1)  # odd positions: offset+1, offset+3, ...
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

def _build_initial_network():
    """
    Build a good initial sorting network.
    Try to use a network for the next power of 2 >= N, then remove wires.
    For N=18, use 32-wire network and remove wires 18-31.
    """
    # Find next power of 2
    p = 1
    while p < N:
        p *= 2
    
    if p == N:
        network = _batcher_odd_even_mergesort(N)
    else:
        # Build for p wires, then filter to only use wires < N
        network = _batcher_odd_even_mergesort(p)
        # Remove comparators that involve wires >= N
        network = [(i, j) for i, j in network if i < N and j < N]
        # This may not be valid for N wires, so fall back to odd-even transposition
        # Actually, this approach is flawed. Let's just use odd-even transposition
        # or build a proper network.
        network = _odd_even_transposition()
    
    # Remove duplicates
    seen = set()
    unique = []
    for comp in network:
        if comp not in seen:
            seen.add(comp)
            unique.append(comp)
    return unique

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

def _batch_verify(comparator_lists):
    """
    Verify multiple comparator lists at once.
    Returns a list of booleans, one per input list.
    """
    results = []
    for comps in comparator_lists:
        state = _INITIAL_MASKS[:]
        for i, j in comps:
            a = state[i]
            b = state[j]
            state[i] = a & b
            state[j] = a | b
        
        valid = True
        for i in range(N - 1):
            if (state[i] & ~state[i+1]) != 0:
                valid = False
                break
        results.append(valid)
    return results

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _build_initial_network():
    """
    Build a good initial sorting network.
    For N=18, try to construct a smaller network than odd-even transposition.
    """
    # Start with odd-even transposition as baseline
    network = _odd_even_transposition()
    
    # Try to improve by removing redundant comparators greedily
    # This is a quick pass to get a smaller starting point
    changed = True
    t0 = time.time()
    while changed and time.time() - t0 < 5.0:
        changed = False
        for i in range(len(network) - 1, -1, -1):
            neighbor = network[:i] + network[i+1:]
            if _verify(neighbor):
                network = neighbor
                changed = True
                break  # Restart from beginning after a change
    
    return network

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Build initial network (already partially optimized)
    current = _build_initial_network()
    best = current[:]
    best_len = len(best)
    
    # Initial verification
    if not _verify(best):
        current = _odd_even_transposition()
        best = current[:]
        best_len = len(best)
    
    # Phase 1: Aggressive greedy removal
    # Try to remove each comparator one by one. If the network is still valid, keep it removed.
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
                break  # Restart from beginning after a change
    
    # Phase 2: Simulated Annealing with insert-delete moves
    # Key insight: insert-delete moves allow escaping local minima by temporarily
    # increasing size to enable larger reductions
    
    T = 8.0
    T_min = 0.05
    cooling_rate = 0.995
    
    rng = random.Random(42)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Track consecutive failures to adjust temperature
    fail_count = 0
    success_count = 0
    
    while time.time() < deadline and T > T_min:
        # Weighted move selection
        # Insert-delete moves are key for escaping local minima
        move_type = rng.choices(
            ['remove', 'insert_delete', 'swap', 'move'],
            weights=[4, 3, 1, 2]
        )[0]
        neighbor = current[:]
        
        if move_type == 'remove' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            neighbor.pop(idx)
            
        elif move_type == 'insert_delete':
            # Remove a random comparator and insert a different one
            # This allows the network to explore different structures
            if len(neighbor) > 0:
                idx = rng.randrange(len(neighbor))
                removed_comp = neighbor.pop(idx)
                # Choose a new comparator that's not already in the network
                available = [c for c in all_comparators if c not in neighbor]
                if available:
                    new_comp = rng.choice(available)
                    pos = rng.randrange(len(neighbor) + 1)
                    neighbor.insert(pos, new_comp)
                else:
                    # No available comparators, put it back
                    neighbor.insert(idx, removed_comp)
            
        elif move_type == 'swap' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
        elif move_type == 'move' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                comp = neighbor.pop(i)
                if i < j:
                    j -= 1
                neighbor.insert(j, comp)

        if _verify(neighbor):
            delta = len(neighbor) - len(current)
            # Accept if better or with probability exp(-delta/T)
            if delta <= 0 or rng.random() < 2.718281828459045 ** (-delta / T):
                current = neighbor
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
                success_count += 1
                fail_count = 0
            else:
                fail_count += 1
        else:
            fail_count += 1
        
        # Adjust cooling rate based on success
        if success_count > 10:
            cooling_rate = 0.998  # Cool slower when making progress
            success_count = 0
        elif fail_count > 50:
            cooling_rate = 0.99  # Cool faster when stuck
            fail_count = 0
        
        T *= cooling_rate
        
        # Periodic restart: if stuck for too long, try a fresh greedy pass
        if time.time() < deadline and len(current) > best_len + 5:
            # Try to get back to best by greedy removal from best
            candidate = best[:]
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(candidate) - 1, -1, -1):
                    neighbor = candidate[:i] + candidate[i+1:]
                    if _verify(neighbor):
                        candidate = neighbor
                        changed = True
                        break
            if len(candidate) < len(current):
                current = candidate
        
    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best

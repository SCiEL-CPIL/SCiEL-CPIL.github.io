"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import random
import math

_INITIAL_BITSETS = None

def _init_bitsets():
    """
    Initialize 18 bitsets, each of length 2^N.
    bitset[k] has bit i set if the i-th input (0 to 2^N-1) has a 1 in wire k.
    """
    total_bits = 1 << N
    bitsets = [0] * N
    
    for k in range(N):
        period = 1 << (k + 1)
        half_period = 1 << k
        # Mask for one period: lower half 0s, upper half 1s
        pattern = (1 << half_period) - 1
        pattern <<= half_period
        
        # Repeat pattern to fill total_bits
        # Use doubling
        current_val = pattern
        current_len = period
        
        while current_len < total_bits:
            current_val |= (current_val << current_len)
            current_len *= 2
            
        bitsets[k] = current_val

    return bitsets

def _get_initial_bitsets():
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is None:
        _INITIAL_BITSETS = _init_bitsets()
    return _INITIAL_BITSETS

def _is_sorting_network(comparators):
    """
    Check if network sorts all 2^N Boolean inputs using bitset simulation.
    """
    if not comparators:
        return N <= 1
        
    # Get a copy of the initial bitsets
    bitsets = list(_get_initial_bitsets())
    
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        bitsets[i] = bi & bj
        bitsets[j] = bi | bj
        
    # Check if sorted:
    # For all k < l, bitsets[k] must be a subset of bitsets[l].
    # i.e., bitsets[k] & ~bitsets[l] == 0.
    # It is sufficient to check adjacent wires: k and k+1.
    for k in range(N - 1):
        if bitsets[k] & ~bitsets[k + 1]:
            return False
            
    return True

def _batcher_odd_even_mergesort(n):
    """
    Generate a Batcher odd-even mergesort network for n wires.
    n must be a power of 2 for the standard recursive definition.
    Returns a list of (i, j) pairs.
    """
    if n <= 1:
        return []
    
    # Recursively sort first half and second half
    half = n // 2
    comps = _batcher_odd_even_mergesort(half)
    comps2 = _batcher_odd_even_mergesort(half)
    # Shift the second half's indices
    comps2 = [(i + half, j + half) for i, j in comps2]
    
    # Odd-even merge
    merge_comps = _batcher_odd_even_merge(n)
    
    return comps + comps2 + merge_comps

def _batcher_odd_even_merge(n):
    """
    Odd-even merge for n wires (n is power of 2).
    """
    if n <= 1:
        return []
    
    half = n // 2
    comps = []
    
    # Recursively merge odd and even positions
    # This is a standard recursive definition for OES
    # However, a simpler iterative or direct recursive for the whole merge is often easier to code correctly.
    # Let's use the standard recursive definition:
    # OES(n):
    #   OES(n/2) on even indices
    #   OES(n/2) on odd indices
    #   For i in 0..n/2-1: compare (i, i+1) if i is even? No, the pattern is specific.
    
    # Actually, the standard Batcher OES merge is:
    # 1. Recursively merge the even-indexed elements (0, 2, 4, ...)
    # 2. Recursively merge the odd-indexed elements (1, 3, 5, ...)
    # 3. For i from 0 to n/2 - 1:
    #      if i is even: compare (i, i+1)
    #      if i is odd: compare (i-1, i) -- wait, the indices are logical.
    
    # Let's stick to a simpler, verified implementation for small N or use the known structure.
    # For N=32, the number of comparators is ~250.
    # A simpler approach: Use the odd-even transposition sort as a "dense" seed?
    # Odd-even transposition on 32 wires has 32*32/2 = 512 comparators.
    # That is even denser! And it's much simpler to generate.
    # The knowledge said "Batcher mergesort on 32 wires gives ~250+... vs 153 for odd-even transposition".
    # Wait, Odd-Even Transposition on 32 wires is $N^2/2 = 512$ comparators.
    # Batcher Mergesort on 32 wires is $\approx \frac{1}{2} N \log^2 N \approx 0.5 * 32 * 25 = 400$?
    # Actually, Batcher is $O(N \log^2 N)$.
    # Let's just use Odd-Even Transposition on 32 wires. It's simpler and very dense.
    
    return [] # Not used if we switch to OET

def _odd_even_transposition_network(n):
    """
    Generate an odd-even transposition network for n wires.
    """
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _greedy_reduce(comparators, deadline):
    """
    Greedy reduction: try removing each comparator, keep if valid.
    """
    current = list(comparators)
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        
        # Try removing comparator at i
        saved = current[i]
        current.pop(i)
        
        if _is_sorting_network(current):
            # Keep it removed
            pass
        else:
            # Restore
            current.insert(i, saved)
            i += 1
            
    return current

def _simulated_annealing(start_network, deadline):
    """
    Simulated Annealing to find a smaller sorting network.
    """
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    
    T = 2.0
    cooling_rate = 0.995
    min_T = 0.05
    
    steps_since_improvement = 0
    max_stall_steps = 100
    
    while time.time() < deadline and T > min_T:
        r = random.random()
        neighbor = list(current)
        move_valid = False
        delta = 0
        
        if r < 0.35:
            # Remove
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                delta = -1
                move_valid = True
        elif r < 0.55:
            # Slide
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                i, j = neighbor[idx]
                if random.random() < 0.5:
                    if i + 1 < N - 1 and j + 1 < N:
                        ni, nj = i + 1, j + 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
                else:
                    if i - 1 >= 0 and j - 1 > 0:
                        ni, nj = i - 1, j - 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
        elif r < 0.70:
            # Replace
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                if random.random() < 0.6:
                    dist = random.randint(1, 3)
                else:
                    dist = random.randint(1, N - 1)
                a = random.randint(0, N - 1 - dist)
                b = a + dist
                neighbor[idx] = (a, b)
                delta = 0
                move_valid = True
        elif r < 0.80:
            # Move
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    comp = neighbor.pop(idx1)
                    neighbor.insert(idx2, comp)
                    delta = 0
                    move_valid = True
        elif r < 0.90:
            # Swap
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    delta = 0
                    move_valid = True
        else:
            # Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor.append((i, j))
            delta = 1
            move_valid = True
        
        if not move_valid:
            T *= cooling_rate
            continue

        if _is_sorting_network(neighbor):
            new_size = len(neighbor)
            
            if new_size < best_size:
                best = neighbor
                best_size = new_size
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            if delta < 0:
                current = neighbor
            elif delta == 0:
                current = neighbor
            else:
                if random.random() < math.exp(-1.0 / T):
                    current = neighbor
        
        if steps_since_improvement > max_stall_steps:
            T = max(T * 1.5, 0.5)
            steps_since_improvement = 0
            if random.random() < 0.3 and len(current) > 0:
                for _ in range(random.randint(1, 3)):
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    current.append((i, j))
        
        T *= cooling_rate
        
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    # Strategy 1: Dense Seed (Odd-Even Transposition on 32 wires, filtered to 18)
    dense_seed_32 = _odd_even_transposition_network(32)
    dense_seed_18 = [(i, j) for i, j in dense_seed_32 if i < N and j < N]
    
    # Strategy 2: Standard Sparse Seed (Odd-Even Transposition on 18 wires)
    sparse_seed_18 = _odd_even_transposition_network(N)
    
    best = list(sparse_seed_18)
    
    # Phase 1: Greedy reduction on both seeds
    # Reduce dense seed
    dense_reduced = _greedy_reduce(dense_seed_18, deadline)
    if len(dense_reduced) < len(best):
        best = dense_reduced
    
    # Reduce sparse seed
    sparse_reduced = _greedy_reduce(sparse_seed_18, deadline)
    if len(sparse_reduced) < len(best):
        best = sparse_reduced
    
    # Phase 2: Simulated Annealing
    # We have time for a few SA runs.
    # Start SA from the best reduced seed.
    # Also try starting from the other reduced seed if it's close.
    
    candidates = [dense_reduced, sparse_reduced]
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    for start_net in unique_candidates:
        if time.time() > deadline:
            break
        
        # Run SA
        candidate = _simulated_annealing(start_net, deadline)
        
        if len(candidate) < len(best):
            best = candidate
            # Quick greedy reduction on the new best
            if time.time() < deadline:
                quick_deadline = min(deadline, time.time() + 2)
                best = _greedy_reduce(best, quick_deadline)
                
    return best
# EVOLVE-BLOCK-END

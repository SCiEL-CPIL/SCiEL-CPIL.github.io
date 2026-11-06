"""Optimized two-colouring of [1,256] minimizing monochromatic 4-APs."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
MAX_D = N // 3  # 85

def _thue_morse_fast(n):
    """Generate Thue-Morse sequence: t(i) = popcount(i) % 2."""
    c = np.zeros(n, dtype=np.int8)
    # Thue-Morse: t(2k) = t(k), t(2k+1) = 1 - t(k)
    # Or simply: t(i) = parity of number of 1s in binary representation of i
    # Fast generation:
    c[0] = 0
    for i in range(1, n):
        # t(i) = t(i >> 1) ^ (i & 1)
        c[i] = c[i >> 1] ^ (i & 1)
    return c

def _precompute_ap_indices():
    """Precompute all 4-term APs and neighbor lists for fast delta calculation."""
    aps = []
    # neighbors[pos] = list of tuples (other1, other2, other3) for each AP containing pos
    # We don't strictly need to store the AP itself, just the other 3 positions
    neighbors = [[] for _ in range(N)]
    
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            p0 = a - 1
            p1 = p0 + d
            p2 = p0 + 2*d
            p3 = p0 + 3*d
            
            # For each position in this AP, store the other 3
            neighbors[p0].append((p1, p2, p3))
            neighbors[p1].append((p0, p2, p3))
            neighbors[p2].append((p0, p1, p3))
            neighbors[p3].append((p0, p1, p2))
            
    return neighbors

# Precompute once at module level
_NEIGHBORS = _precompute_ap_indices()

def _count_mono4_vectorized(c):
    """Vectorized count of monochromatic 4-APs."""
    count = 0
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            continue
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += int(np.sum(mask))
    return count

def _delta_flip(c, pos):
    """Compute change in mono-4-AP count when flipping position pos.
    Returns the delta (new_count - old_count)."""
    old_val = c[pos]
    # new_val = 1 - old_val
    
    delta = 0
    # For each AP containing pos, the other 3 positions are in _NEIGHBORS[pos]
    for n1, n2, n3 in _NEIGHBORS[pos]:
        v1 = c[n1]
        v2 = c[n2]
        v3 = c[n3]
        
        # Check if currently monochromatic (all 4 equal)
        # Since we are flipping pos, 'before' is determined by old_val
        # 'after' is determined by new_val
        
        # Before: mono if old_val == v1 == v2 == v3
        mono_before = (old_val == v1) and (v1 == v2) and (v2 == v3)
        
        # After: mono if (1-old_val) == v1 == v2 == v3
        # Which is equivalent to v1 == v2 == v3 AND v1 != old_val
        # Or simply: (1-old_val) == v1 and v1 == v2 and v2 == v3
        mono_after = ((1 - old_val) == v1) and (v1 == v2) and (v2 == v3)
        
        if mono_before and not mono_after:
            delta -= 1
        elif not mono_before and mono_after:
            delta += 1
            
    return delta

def _local_search_incremental(c, time_limit):
    """Local search using incremental delta computation."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try positions in random order for diversity
        order = list(range(N))
        random.shuffle(order)
        for pos in order:
            if (time.time() - start) >= time_limit:
                break
            delta = _delta_flip(best_c, pos)
            if delta < 0:
                best_c[pos] = 1 - best_c[pos]
                best_count += delta
                improved = True
    
    return best_c, best_count

def _simulated_annealing(c, time_limit):
    """Simulated annealing to escape local minima."""
    start = time.time()
    cur_c = c.copy()
    cur_count = _count_mono4_vectorized(cur_c)
    best_c = cur_c.copy()
    best_count = cur_count
    
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling = 0.995
    
    while (time.time() - start) < time_limit and T > T_min:
        # Pick random position
        pos = random.randint(0, N - 1)
        delta = _delta_flip(cur_c, pos)
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / T):
            cur_c[pos] = 1 - cur_c[pos]
            cur_count += delta
            if cur_count < best_count:
                best_count = cur_count
                best_c = cur_c.copy()
        
        T *= cooling
    
    return best_c, best_count

def _two_flip_search(c, time_limit):
    """Try flipping pairs of positions simultaneously."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try a limited set of pairs
        # Generate random pairs to avoid O(N^2) generation cost
        max_pairs = 500
        for _ in range(max_pairs):
            if (time.time() - start) >= time_limit:
                break
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j: continue
            
            # Flip both
            best_c[i] = 1 - best_c[i]
            best_c[j] = 1 - best_c[j]
            new_count = _count_mono4_vectorized(best_c)
            if new_count < best_count:
                best_count = new_count
                improved = True
            else:
                best_c[i] = 1 - best_c[i]
                best_c[j] = 1 - best_c[j]
    
    return best_c, best_count

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with Thue-Morse sequence
    c = _thue_morse_fast(N)
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)
    
    # Phase 1: Local search with incremental deltas
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 30))
    
    # Phase 2: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 15:
        sa_c, sa_count = _simulated_annealing(best_c, min(remaining - 15, 20))
        if sa_count < best_count:
            best_c = sa_c
            best_count = sa_count
    
    # Phase 3: More local search
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 20))
    
    # Phase 4: Two-flip search
    remaining = deadline - time.time()
    if remaining > 15:
        best_c, best_count = _two_flip_search(best_c, min(remaining - 15, 15))
    
    # Phase 5: Final local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 5, 10))
    
    # Phase 6: Try different Thue-Morse offsets with local search
    # Since local search is faster, we can try more offsets
    remaining = deadline - time.time()
    if remaining > 15:
        for offset in range(0, 16):
            if time.time() >= deadline - 10:
                break
            c2 = _thue_morse_fast(N)
            c2 = np.roll(c2, offset)
            # Short local search for each restart
            c2, c2_count = _local_search_incremental(c2, min(5, deadline - time.time() - 5))
            if c2_count < best_count:
                best_c = c2
                best_count = c2_count
    
    # Final local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 5, 10))
    
    # Convert to list of ints
    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
# EVOLVE-BLOCK-END

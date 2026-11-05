"""Optimized two-colouring of [1,256] to minimize monochromatic 4-APs."""
import time
import random
import numpy as np

# Precompute all 4-term arithmetic progressions in [1, 256]
# A 4-AP is (a, a+d, a+2d, a+3d) with d >= 1 and a+3d <= 256
N = 256

def _precompute_aps():
    """Return list of 4-tuples (a, a+d, a+2d, a+3d) for all valid 4-APs."""
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _count_mono(coloring, aps):
    """Count monochromatic 4-APs. coloring is a list/array of 0/1."""
    count = 0
    for (a, b, c, d) in aps:
        if coloring[a] == coloring[b] == coloring[c] == coloring[d]:
            count += 1
    return count

def _thue_morse():
    """Generate Thue-Morse sequence of length 256."""
    tm = [0] * N
    for i in range(1, N):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return tm

def _random_coloring():
    """Generate a random 0/1 coloring of length 256."""
    return [random.randint(0, 1) for _ in range(N)]

def _local_search(coloring, aps, max_iters=5000, deadline=None):
    """
    Local search: flip single bits to reduce mono-AP count.
    Returns (best_coloring, best_count).
    """
    best_coloring = list(coloring)
    best_count = _count_mono(best_coloring, aps)
    
    # Precompute for each position, which APs it belongs to
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[d].append(idx)
    
    def fast_count(coloring):
        """Count mono APs using the AP list."""
        cnt = 0
        for (a, b, c, d) in aps:
            if coloring[a] == coloring[b] == coloring[c] == coloring[d]:
                cnt += 1
        return cnt
    
    current = list(best_coloring)
    current_count = best_count
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try flipping each position
        for pos in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            old_val = current[pos]
            new_val = 1 - old_val
            current[pos] = new_val
            
            new_count = fast_count(current)
            
            if new_count < current_count:
                current_count = new_count
                if current_count < best_count:
                    best_count = current_count
                    best_coloring = list(current)
                improved = True
            else:
                current[pos] = old_val
        
        if not improved:
            # Try random restarts within local search
            if deadline is not None and time.time() > deadline:
                break
            # Perturb: flip a few random bits
            for _ in range(3):
                p = random.randint(0, N - 1)
                current[p] = 1 - current[p]
            current_count = fast_count(current)
            if current_count < best_count:
                best_count = current_count
                best_coloring = list(current)
    
    return best_coloring, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    aps = _precompute_aps()
    
    # Start with Thue-Morse
    best_coloring = _thue_morse()
    best_count = _count_mono(best_coloring, aps)
    
    # Try local search from Thue-Morse
    if time.time() < deadline - 5:
        lc, lc_count = _local_search(best_coloring, aps, max_iters=2000, deadline=deadline - 5)
        if lc_count < best_count:
            best_count = lc_count
            best_coloring = lc
    
    # Try random restarts with local search
    rng = random.Random(42)
    num_restarts = 0
    while time.time() < deadline - 10:
        # Random coloring with slight bias
        coloring = [rng.randint(0, 1) for _ in range(N)]
        init_count = _count_mono(coloring, aps)
        
        # Only do local search if initial count is not too bad
        if init_count < best_count + 50:
            lc, lc_count = _local_search(coloring, aps, max_iters=1000, deadline=deadline - 10)
            if lc_count < best_count:
                best_count = lc_count
                best_coloring = lc
        
        num_restarts += 1
        # Also try structured random patterns
        if num_restarts % 5 == 0:
            # Block pattern with random block sizes
            block_size = rng.randint(2, 8)
            coloring2 = [(i // block_size) % 2 for i in range(N)]
            c2 = _count_mono(coloring2, aps)
            if c2 < best_count:
                best_count = c2
                best_coloring = coloring2
            else:
                lc2, lc2_count = _local_search(coloring2, aps, max_iters=500, deadline=deadline - 10)
                if lc2_count < best_count:
                    best_count = lc2_count
                    best_coloring = lc2
    
    # Final: return as list of ints
    return [int(c) for c in best_coloring]

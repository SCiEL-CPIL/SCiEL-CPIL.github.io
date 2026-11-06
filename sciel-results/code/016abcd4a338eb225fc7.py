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
    if n > 0:
        c[0] = 0
        for i in range(1, n):
            c[i] = c[i >> 1] ^ (i & 1)
    return c

def _precompute_ap_indices():
    """Precompute all 4-term APs and for each position, which APs contain it."""
    aps = []  # list of (a, d, positions)
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            positions = (a - 1, a - 1 + d, a - 1 + 2*d, a - 1 + 3*d)
            aps.append((a, d, positions))
    
    # For each position, list of (ap_idx, role)
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, d, positions) in enumerate(aps):
        for role, p in enumerate(positions):
            pos_to_aps[p].append((idx, role))
    
    return aps, pos_to_aps

_APS, _POS_TO_APS = _precompute_ap_indices()
_TOTAL_APS = len(_APS)

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
    """Compute change in mono-4-AP count when flipping position pos."""
    old_val = c[pos]
    new_val = 1 - old_val
    delta = 0
    for ap_idx, role in _POS_TO_APS[pos]:
        p0, p1, p2, p3 = _APS[ap_idx][2]
        # Check other 3 positions
        if role == 0:
            others = (p1, p2, p3)
        elif role == 1:
            others = (p0, p2, p3)
        elif role == 2:
            others = (p0, p1, p3)
        else:
            others = (p0, p1, p2)
        
        # Count how many of the others match old_val and new_val
        # was_mono: all 4 == old_val. Since c[pos]==old_val, we need others == old_val
        # now_mono: all 4 == new_val. Since c[pos]==new_val (after flip), we need others == new_val
        
        # Optimization:
        # If any other != old_val, it wasn't mono.
        # If any other != new_val, it won't be mono.
        
        # Quick check
        if c[others[0]] == old_val and c[others[1]] == old_val and c[others[2]] == old_val:
            delta -= 1 # Was mono, now broken
        if c[others[0]] == new_val and c[others[1]] == new_val and c[others[2]] == new_val:
            delta += 1 # Was not mono (implied by above logic? No, if it was mono, others==old_val, so others!=new_val. So these are mutually exclusive)
            # Actually, if others == old_val, then others != new_val.
            # If others == new_val, then others != old_val.
            # So we can use if/elif
    return delta

def _delta_two_flip(c, i, j):
    """Compute change in mono-4-AP count when flipping positions i and j simultaneously."""
    if i == j:
        return 0
    
    # Union of APs containing i or j
    # We need to check APs that contain i, APs that contain j, and handle intersections.
    # An AP is affected if it contains i or j.
    # If it contains both, we must evaluate the joint flip.
    # If it contains only i, the change is determined by i's flip (j is unchanged).
    # If it contains only j, the change is determined by j's flip (i is unchanged).
    
    # Let's iterate over APs containing i
    # And APs containing j
    # But we must avoid double counting APs containing both.
    
    # Strategy:
    # 1. Get set of APs containing i
    # 2. Get set of APs containing j
    # 3. Union of these sets.
    # 4. For each AP in union, determine if it's mono before and after flipping i and j.
    
    # To be efficient, we can just iterate over _POS_TO_APS[i] and _POS_TO_APS[j]
    # and use a set of ap_indices to avoid duplicates.
    
    affected_aps = set()
    for ap_idx, _ in _POS_TO_APS[i]:
        affected_aps.add(ap_idx)
    for ap_idx, _ in _POS_TO_APS[j]:
        affected_aps.add(ap_idx)
        
    delta = 0
    old_i = c[i]
    old_j = c[j]
    new_i = 1 - old_i
    new_j = 1 - old_j
    
    for ap_idx in affected_aps:
        p0, p1, p2, p3 = _APS[ap_idx][2]
        
        # Get current colors
        c0 = c[p0]
        c1 = c[p1]
        c2 = c[p2]
        c3 = c[p3]
        
        # Determine new colors for p0..p3
        n0 = c0
        n1 = c1
        n2 = c2
        n3 = c3
        
        if p0 == i:
            n0 = new_i
        elif p0 == j:
            n0 = new_j
            
        if p1 == i:
            n1 = new_i
        elif p1 == j:
            n1 = new_j
            
        if p2 == i:
            n2 = new_i
        elif p2 == j:
            n2 = new_j
            
        if p3 == i:
            n3 = new_i
        elif p3 == j:
            n3 = new_j
            
        # Check mono before
        was_mono = (c0 == c1 == c2 == c3)
        # Check mono after
        is_mono = (n0 == n1 == n2 == n3)
        
        if was_mono and not is_mono:
            delta -= 1
        elif not was_mono and is_mono:
            delta += 1
            
    return delta

def _local_search_incremental(c, time_limit, random_seed=42):
    """Local search using incremental delta computation."""
    rng = random.Random(random_seed)
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        order = list(range(N))
        rng.shuffle(order)
        for pos in order:
            if (time.time() - start) >= time_limit:
                break
            delta = _delta_flip(best_c, pos)
            if delta < 0:
                best_c[pos] = 1 - best_c[pos]
                best_count += delta
                improved = True
    
    return best_c, best_count

def _two_flip_local_search(c, time_limit, random_seed=42):
    """Local search using 2-flip moves with fast delta calculation."""
    rng = random.Random(random_seed)
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try random pairs
        # To be more effective, we can try all pairs if time permits, or a large random sample
        # N=256, pairs ~ 32k. Delta calc is ~50 ops. 32k * 50 = 1.6M ops. Fast enough for a few passes.
        for i in range(N):
            if (time.time() - start) >= time_limit:
                break
            for j in range(i + 1, N):
                delta = _delta_two_flip(best_c, i, j)
                if delta < 0:
                    best_c[i] = 1 - best_c[i]
                    best_c[j] = 1 - best_c[j]
                    best_count += delta
                    improved = True
    
    return best_c, best_count

def _simulated_annealing(c, time_limit, random_seed=42):
    """Simulated annealing to escape local minima."""
    rng = random.Random(random_seed)
    start = time.time()
    cur_c = c.copy()
    cur_count = _count_mono4_vectorized(cur_c)
    best_c = cur_c.copy()
    best_count = cur_count
    
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    while (time.time() - start) < time_limit and T > T_min:
        # Pick random move: 80% single flip, 20% double flip
        if rng.random() < 0.8:
            pos = rng.randint(0, N - 1)
            delta = _delta_flip(cur_c, pos)
            if delta < 0 or rng.random() < np.exp(-delta / T):
                cur_c[pos] = 1 - cur_c[pos]
                cur_count += delta
                if cur_count < best_count:
                    best_count = cur_count
                    best_c = cur_c.copy()
        else:
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            if i != j:
                delta = _delta_two_flip(cur_c, i, j)
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    cur_c[i] = 1 - cur_c[i]
                    cur_c[j] = 1 - cur_c[j]
                    cur_count += delta
                    if cur_count < best_count:
                        best_count = cur_count
                        best_c = cur_c.copy()
        
        T *= cooling
    
    return best_c, best_count

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Start with Thue-Morse sequence
    c = _thue_morse_fast(N)
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)
    
    # Phase 1: Single-flip local search
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 30))
    
    # Phase 2: Two-flip local search (exploits fast delta)
    remaining = deadline - time.time()
    if remaining > 15:
        best_c, best_count = _two_flip_local_search(best_c, min(remaining - 15, 40))
    
    # Phase 3: Simulated Annealing with mixed moves
    remaining = deadline - time.time()
    if remaining > 20:
        sa_c, sa_count = _simulated_annealing(best_c, min(remaining - 20, 60))
        if sa_count < best_count:
            best_c = sa_c
            best_count = sa_count
    
    # Phase 4: Final Polish - Single then Double flip
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 5, 10))
        remaining = deadline - time.time()
        if remaining > 5:
            best_c, best_count = _two_flip_local_search(best_c, min(remaining - 2, 5))
    
    # Final verification
    final_count = _count_mono4_vectorized(best_c)
    if final_count < best_count:
        best_count = final_count
        
    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
# EVOLVE-BLOCK-END

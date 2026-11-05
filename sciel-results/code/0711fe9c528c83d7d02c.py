import random
import time
N = 16
TOTAL = N * N
DEADLINE = time.time() + 280.0

# Precomputation
# iso_bits[p][a] is a bitmask of points b such that {p, a, b} is isosceles.
# We store as list of lists of Python ints (big integers for bitsets).
iso_bits = [[0] * TOTAL for _ in range(TOTAL)]

def _precompute():
    pts = [(i // N, i % N) for i in range(TOTAL)]
    for i in range(TOTAL):
        for j in range(TOTAL):
            if i == j:
                continue
            p = pts[i]
            a = pts[j]
            d_pa = (p[0]-a[0])**2 + (p[1]-a[1])**2
            mask = 0
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                b = pts[k]
                d_pb = (p[0]-b[0])**2 + (p[1]-b[1])**2
                d_ab = (a[0]-b[0])**2 + (a[1]-b[1])**2
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    mask |= (1 << k)
            iso_bits[i][j] = mask

_precompute()

def _point_idx(p):
    return p[0] * N + p[1]

def _idx_point(i):
    return (i // N, i % N)

def _creates_isosceles_idx(p_idx, S_bits):
    """
    Check if adding point p_idx to set represented by S_bits creates isosceles.
    S_bits is a bitmask of points in S.
    """
    # For each a in S, check if any b in S is in iso_bits[p_idx][a]
    # We iterate over set bits in S_bits
    s = S_bits
    while s:
        # Get lowest set bit
        lsb = s & -s
        a_idx = lsb.bit_length() - 1
        s ^= lsb
        
        # Check intersection with iso_bits[p_idx][a_idx]
        # We need to check if any b in S is in the conflict set.
        # Note: iso_bits[p_idx][a_idx] contains b's that form isosceles with p_idx and a_idx.
        # We must ensure b is in S.
        # The intersection S_bits & iso_bits[p_idx][a_idx] will be non-zero if conflict exists.
        if S_bits & iso_bits[p_idx][a_idx]:
            return True
    return False

def _is_maximal(S_bits):
    """
    Check if set S_bits is maximal.
    """
    for p_idx in range(TOTAL):
        if S_bits & (1 << p_idx):
            continue
        if not _creates_isosceles_idx(p_idx, S_bits):
            return False
    return True

def _greedy(order, S_bits=0):
    """
    Greedy construction. order is list of point indices.
    Returns S_bits.
    """
    for p_idx in order:
        if not (S_bits & (1 << p_idx)): # already in? greedy usually picks unique
            if not _creates_isosceles_idx(p_idx, S_bits):
                S_bits |= (1 << p_idx)
    return S_bits

def _local_search(S_bits, rng, max_iters=200):
    """
    Local search to reduce size.
    """
    best_bits = S_bits
    best_count = bin(best_bits).count('1')
    
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try removing each point
        s = best_bits
        while s:
            lsb = s & -s
            idx = lsb.bit_length() - 1
            s ^= lsb
            
            candidate_bits = best_bits & ~lsb
            
            # Check if still maximal
            if not _is_maximal(candidate_bits):
                # Try to complete greedily with random order of remaining
                # This is expensive, maybe only do if count is small?
                # Or just skip? The original code did this.
                # Let's try a simple greedy fill
                remaining_mask = ((1 << TOTAL) - 1) & ~candidate_bits
                # Get list of indices
                rem_list = []
                r = remaining_mask
                while r:
                    lb = r & -r
                    ri = lb.bit_length() - 1
                    rem_list.append(ri)
                    r ^= lb
                rng.shuffle(rem_list)
                
                completed_bits = candidate_bits
                for p_idx in rem_list:
                    if not _creates_isosceles_idx(p_idx, completed_bits):
                        completed_bits |= (1 << p_idx)
                
                if bin(completed_bits).count('1') < best_count and _is_maximal(completed_bits):
                    best_bits = completed_bits
                    best_count = bin(best_bits).count('1')
                    improved = True
                    break # Restart loop with new best
            else:
                # If removing a point keeps it maximal, we found a smaller maximal set!
                # Wait, if it's maximal and smaller, that's a direct improvement.
                # But if it's NOT maximal, we need to fill.
                # If it IS maximal, it's a valid smaller set.
                best_bits = candidate_bits
                best_count -= 1
                improved = True
                break
        
        if not improved:
            break
            
    return best_bits

def _bits_to_list(bits):
    res = []
    s = bits
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        res.append(_idx_point(idx))
        s ^= lsb
    return res

def _list_to_bits(lst):
    bits = 0
    for p in lst:
        bits |= (1 << _point_idx(p))
    return bits

def construct():
    rng = random.Random(42)
    best_bits = 0
    best_len = float('inf')
    
    all_indices = list(range(TOTAL))
    
    # Phase 1: Multi-start random greedy
    n_starts = 1000 # Increased due to speed
    for _ in range(n_starts):
        if time.time() > DEADLINE:
            break
        order = list(all_indices)
        rng.shuffle(order)
        res_bits = _greedy(order)
        if _is_maximal(res_bits):
            l = bin(res_bits).count('1')
            if l < best_len:
                best_len = l
                best_bits = res_bits
                
    # Phase 2: Constrained first greedy (Optimized)
    # The original constrained greedy was very slow. 
    # With bitsets, we can afford to run it more.
    # But the logic "blocks count" requires checking for each p, how many q are blocked.
    # _creates_isosceles(q, S + [p]) is fast now.
    
    n_starts_c = 200
    for _ in range(n_starts_c):
        if time.time() > DEADLINE:
            break
        # Random seed
        seed_indices = rng.sample(all_indices, k=min(3, TOTAL))
        chosen_bits = 0
        for idx in seed_indices:
            if not _creates_isosceles_idx(idx, chosen_bits):
                chosen_bits |= (1 << idx)
                
        available_indices = [i for i in all_indices if not (chosen_bits & (1 << i))]
        
        while available_indices:
            if time.time() > DEADLINE:
                break
            best_p = None
            best_score = -1
            # Sample for speed if large
            sample = available_indices if len(available_indices) <= 50 else rng.sample(available_indices, 50)
            
            for p_idx in sample:
                if _creates_isosceles_idx(p_idx, chosen_bits):
                    continue
                # Check blocks
                new_bits = chosen_bits | (1 << p_idx)
                blocks = 0
                # Check how many other available points are blocked
                # We can check all available or sample
                # Since N=16, available is small. Check all.
                for q_idx in available_indices:
                    if q_idx == p_idx:
                        continue
                    if _creates_isosceles_idx(q_idx, new_bits):
                        blocks += 1
                if blocks > best_score:
                    best_score = blocks
                    best_p = p_idx
            
            if best_p is None:
                break
            chosen_bits |= (1 << best_p)
            available_indices.remove(best_p)
            
        if _is_maximal(chosen_bits):
            l = bin(chosen_bits).count('1')
            if l < best_len:
                best_len = l
                best_bits = chosen_bits
                
    # Phase 3: Local Search
    if best_bits:
        improved_bits = _local_search(best_bits, rng, max_iters=500)
        if bin(improved_bits).count('1') < best_len:
            best_bits = improved_bits
            best_len = bin(best_bits).count('1')
            
    # Phase 4: More random starts
    if time.time() < DEADLINE - 10:
        for _ in range(2000):
            if time.time() > DEADLINE:
                break
            order = list(all_indices)
            rng.shuffle(order)
            res_bits = _greedy(order)
            if _is_maximal(res_bits):
                l = bin(res_bits).count('1')
                if l < best_len:
                    best_len = l
                    best_bits = res_bits
                    # Quick local search on improvement
                    if time.time() < DEADLINE - 5:
                        improved_bits = _local_search(best_bits, rng, max_iters=100)
                        if bin(improved_bits).count('1') < best_len:
                            best_bits = improved_bits
                            best_len = bin(best_bits).count('1')

    return _bits_to_list(best_bits)

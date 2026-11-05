"""Minimal maximal no-isosceles set on [0, N)^2."""
import time
import random
import itertools

N = 16
TOTAL = N * N
DEADLINE = time.time() + 280.0  # safety margin before 330s kill

# Precompute isosceles conflict bitmasks
# iso_mask[i][j] = bitmask of points k such that {i, j, k} forms an isosceles triangle
# We'll index points as 0..TOTAL-1, where point p = (p // N, p % N)

def _point_to_idx(p):
    return p[0] * N + p[1]

def _idx_to_point(idx):
    return (idx // N, idx % N)

# Precompute distance squared between all pairs
def _precompute():
    """Precompute isosceles conflict masks."""
    # dist_sq[i][j] = squared distance between point i and point j
    dist_sq = [[0] * TOTAL for _ in range(TOTAL)]
    for i in range(TOTAL):
        pi = _idx_to_point(i)
        for j in range(i + 1, TOTAL):
            pj = _idx_to_point(j)
            dx = pi[0] - pj[0]
            dy = pi[1] - pj[1]
            d = dx * dx + dy * dy
            dist_sq[i][j] = d
            dist_sq[j][i] = d
    
    # iso_mask[i] = bitmask of all points j such that there exists k with {i,j,k} isosceles
    # More precisely: iso_mask[i][j] tells us which k's would create isosceles with i and j
    # For checking if adding p to chosen creates isosceles:
    #   For each pair (a,b) in chosen, check if p is in iso_mask[a][b]
    #   OR if d(p,a)==d(p,b) or d(p,a)==d(a,b) or d(p,b)==d(a,b)
    
    # Better approach: precompute for each point p, the set of pairs (a,b) that are "conflicting"
    # i.e., adding p would create an isosceles triangle with a and b.
    
    # iso_conflict[p] = set of (a,b) pairs such that {p,a,b} is isosceles
    # But for efficiency, let's use: can_add(p, chosen_mask) checks if p conflicts with any pair in chosen
    
    # Alternative: for each point p, precompute which points q are "incompatible" with p
    # in the sense that there exists some r such that {p,q,r} is isosceles.
    # But that's not quite right either.
    
    # Let's use this approach:
    # For a given set S (as bitmask), to check if point p can be added:
    #   For each a in S: for each b in S, a < b: check if {p,a,b} is isosceles
    #   This is O(|S|^2) which is fine for |S| < 50
    
    # Precompute: is_iso[p][a][b] = True if {p,a,b} is isosceles
    # That's 256^3 = 16.7M booleans, too much.
    
    # Instead: for each pair (a,b), precompute the set of p such that {p,a,b} is isosceles
    # iso_with_pair[a][b] = bitmask of p such that {p,a,b} is isosceles
    # That's 256*256/2 * 256 bits = ~4MB, manageable
    
    iso_with_pair = [[0] * TOTAL for _ in range(TOTAL)]
    for a in range(TOTAL):
        for b in range(a + 1, TOTAL):
            pa = _idx_to_point(a)
            pb = _idx_to_point(b)
            d_ab = dist_sq[a][b]
            mask = 0
            for p in range(TOTAL):
                if p == a or p == b:
                    continue
                pp = _idx_to_point(p)
                d_pa = dist_sq[p][a]
                d_pb = dist_sq[p][b]
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    mask |= (1 << p)
            iso_with_pair[a][b] = mask
            iso_with_pair[b][a] = mask
    
    return dist_sq, iso_with_pair

_DIST_SQ, _ISO_WITH_PAIR = _precompute()


def _creates_isosceles(p_idx, chosen_mask):
    """Check if adding p_idx to the set represented by chosen_mask creates any isosceles triangle."""
    # For each pair (a,b) in chosen_mask, check if p_idx is in _ISO_WITH_PAIR[a][b]
    # Iterate over set bits in chosen_mask
    m = chosen_mask
    points_in_chosen = []
    while m:
        low = m & (-m)
        idx = low.bit_length() - 1
        points_in_chosen.append(idx)
        m ^= low
    
    n = len(points_in_chosen)
    for i in range(n):
        a = points_in_chosen[i]
        for j in range(i + 1, n):
            b = points_in_chosen[j]
            if _ISO_WITH_PAIR[a][b] & (1 << p_idx):
                return True
    return False


def _is_maximal(chosen_mask):
    """Check if the set is maximal: no point can be added."""
    for p in range(TOTAL):
        if chosen_mask & (1 << p):
            continue
        if not _creates_isosceles(p, chosen_mask):
            return False
    return True


def _mask_to_list(chosen_mask):
    """Convert bitmask to list of point indices."""
    result = []
    m = chosen_mask
    while m:
        low = m & (-m)
        idx = low.bit_length() - 1
        result.append(idx)
        m ^= low
    return result


def _list_to_mask(point_list):
    """Convert list of point indices to bitmask."""
    mask = 0
    for p in point_list:
        mask |= (1 << p)
    return mask


def _greedy(order, chosen_mask=0):
    """Greedy construction following a given point order."""
    for p in order:
        if not _creates_isosceles(p, chosen_mask):
            chosen_mask |= (1 << p)
    return chosen_mask


def _local_search(chosen_mask, rng, max_iters=200):
    """Try to reduce size by removing a point and re-filling minimally."""
    best_mask = chosen_mask
    best_len = _mask_to_list(best_mask).__len__()
    
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        improved = False
        # Try removing each point
        points_in_best = _mask_to_list(best_mask)
        for i in range(len(points_in_best)):
            p_remove = points_in_best[i]
            candidate_mask = best_mask & ~(1 << p_remove)
            # Check if still maximal; if not, try greedy completion
            if not _is_maximal(candidate_mask):
                # Try to complete greedily
                remaining = [p for p in range(TOTAL) if not (candidate_mask & (1 << p))]
                rng.shuffle(remaining)
                completed_mask = candidate_mask
                for p in remaining:
                    if not _creates_isosceles(p, completed_mask):
                        completed_mask |= (1 << p)
                # Check if completed is maximal and smaller
                completed_len = bin(completed_mask).count('1')
                if completed_len < best_len and _is_maximal(completed_mask):
                    best_mask = completed_mask
                    best_len = completed_len
                    improved = True
                    break
        if not improved:
            break
    return best_mask


def _multi_start_greedy(rng, n_starts=50):
    """Multiple random-order greedy constructions."""
    all_points = list(range(TOTAL))
    best_mask = 0
    best_len = TOTAL + 1
    for _ in range(n_starts):
        if time.time() > DEADLINE:
            break
        order = list(all_points)
        rng.shuffle(order)
        result_mask = _greedy(order)
        result_len = bin(result_mask).count('1')
        if _is_maximal(result_mask) and result_len < best_len:
            best_mask = result_mask
            best_len = result_len
    return best_mask


def _constrained_first_greedy(rng, n_starts=30):
    """Greedy where we pick the point that blocks the most others."""
    all_points = list(range(TOTAL))
    best_mask = 0
    best_len = TOTAL + 1
    for _ in range(n_starts):
        if time.time() > DEADLINE:
            break
        # Randomize initial seed
        seed_points = rng.sample(all_points, k=min(3, len(all_points)))
        chosen_mask = 0
        for p in seed_points:
            if not _creates_isosceles(p, chosen_mask):
                chosen_mask |= (1 << p)
        # Now greedily add points, preferring those that are "most constrained"
        available = [p for p in all_points if not (chosen_mask & (1 << p))]
        while available:
            if time.time() > DEADLINE:
                break
            # Score each available point by how many it would block
            best_p = None
            best_score = -1
            # Sample a subset for speed
            sample = available if len(available) <= 200 else rng.sample(available, 200)
            for p in sample:
                if _creates_isosceles(p, chosen_mask):
                    continue
                # Count how many other available points this would block
                test_mask = chosen_mask | (1 << p)
                blocks = 0
                for q in sample:
                    if q == p:
                        continue
                    if _creates_isosceles(q, test_mask):
                        blocks += 1
                if blocks > best_score:
                    best_score = blocks
                    best_p = p
            if best_p is None:
                break
            chosen_mask |= (1 << best_p)
            available.remove(best_p)
        result_len = bin(chosen_mask).count('1')
        if _is_maximal(chosen_mask) and result_len < best_len:
            best_mask = chosen_mask
            best_len = result_len
    return best_mask


def construct():
    rng = random.Random(42)
    best_mask = 0
    best_len = TOTAL + 1

    # Phase 1: Multi-start random greedy
    result_mask = _multi_start_greedy(rng, n_starts=100)
    if result_mask != 0:
        best_mask = result_mask
        best_len = bin(result_mask).count('1')

    # Phase 2: Constrained-first greedy
    result_mask = _constrained_first_greedy(rng, n_starts=40)
    if result_mask != 0 and bin(result_mask).count('1') < best_len:
        best_mask = result_mask
        best_len = bin(result_mask).count('1')

    # Phase 3: Local search on best
    if best_mask != 0:
        improved_mask = _local_search(best_mask, rng, max_iters=100)
        improved_len = bin(improved_mask).count('1')
        if improved_len < best_len:
            best_mask = improved_mask
            best_len = improved_len

    # Phase 4: More random starts if time permits
    if time.time() < DEADLINE - 30:
        result_mask = _multi_start_greedy(rng, n_starts=200)
        if result_mask != 0 and bin(result_mask).count('1') < best_len:
            best_mask = result_mask
            best_len = bin(result_mask).count('1')
        if best_mask != 0:
            improved_mask = _local_search(best_mask, rng, max_iters=100)
            improved_len = bin(improved_mask).count('1')
            if improved_len < best_len:
                best_mask = improved_mask
                best_len = improved_len

    # Phase 5: Final local search
    if best_mask != 0 and time.time() < DEADLINE - 10:
        improved_mask = _local_search(best_mask, rng, max_iters=200)
        if bin(improved_mask).count('1') < best_len:
            best_mask = improved_mask

    # Convert back to list of (x, y) tuples
    if best_mask == 0:
        return []
    
    result = []
    for idx in _mask_to_list(best_mask):
        result.append(_idx_to_point(idx))
    return result

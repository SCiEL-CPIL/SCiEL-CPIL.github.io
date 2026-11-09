"""Combined approach: bitset precomputation + branch-and-bound on 3-uniform hypergraph
with coverage heuristic for greedy, and aggressive pruning."""
import time
import random
import numpy as np

N = 16
TOTAL_POINTS = N * N
POINTS = [(x, y) for x in range(N) for y in range(N)]

# Precompute distance squared matrix
_dist2 = np.zeros((TOTAL_POINTS, TOTAL_POINTS), dtype=np.int32)
for i in range(TOTAL_POINTS):
    for j in range(TOTAL_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        _dist2[i, j] = dx * dx + dy * dy

# Precompute isosceles relations using bitsets
# For each point p, iso_mask[p][i] is a bitset of points j such that {p, i, j} is isosceles
_iso_mask = []
for p in range(TOTAL_POINTS):
    row = [0] * TOTAL_POINTS
    for i in range(TOTAL_POINTS):
        if i == p:
            continue
        mask = 0
        dpi = _dist2[p, i]
        for j in range(TOTAL_POINTS):
            if j == p or j == i:
                continue
            dpj = _dist2[p, j]
            dij = _dist2[i, j]
            if dpi == dpj or dpi == dij or dpj == dij:
                mask |= (1 << j)
        row[i] = mask
    _iso_mask.append(row)

ALL_MASK = (1 << TOTAL_POINTS) - 1


def _can_add(p, S_mask):
    """Check if point p can be added to set S_mask without creating isosceles."""
    if (S_mask >> p) & 1:
        return False
    im = _iso_mask[p]
    S_temp = S_mask
    while S_temp:
        lsb = S_temp & -S_temp
        i = lsb.bit_length() - 1
        S_temp ^= lsb
        if S_mask & im[i]:
            return False
    return True


def _is_maximal(S_mask):
    """Check if S_mask is maximal: no point outside can be added."""
    not_S = ALL_MASK ^ S_mask
    p_bit = not_S
    while p_bit:
        lsb = p_bit & -p_bit
        p = lsb.bit_length() - 1
        p_bit ^= lsb
        im = _iso_mask[p]
        S_temp = S_mask
        blocked = False
        while S_temp:
            i_lsb = S_temp & -S_temp
            i = i_lsb.bit_length() - 1
            S_temp ^= i_lsb
            if S_mask & im[i]:
                blocked = True
                break
        if not blocked:
            return False
    return True


def _greedy_construct(order):
    """Build a maximal no-isosceles set by greedily adding points in given order."""
    S_mask = 0
    for p in order:
        if _can_add(p, S_mask):
            S_mask |= (1 << p)
    return S_mask


def _prune(S_mask):
    """Try to remove points while maintaining maximality. Returns pruned mask."""
    improved = True
    while improved:
        improved = False
        S_temp = S_mask
        while S_temp:
            lsb = S_temp & -S_temp
            i = lsb.bit_length() - 1
            S_temp ^= lsb
            new_S = S_mask ^ lsb
            if _is_maximal(new_S):
                S_mask = new_S
                improved = True
                break
    return S_mask


def _mask_to_list(S_mask):
    points = []
    m = S_mask
    while m:
        lsb = m & -m
        p = lsb.bit_length() - 1
        points.append(POINTS[p])
        m ^= lsb
    return points


def _coverage_score(p, S_mask, candidates_mask):
    """Score how many candidates p would block if added to S_mask."""
    im = _iso_mask[p]
    score = 0
    cand_temp = candidates_mask
    while cand_temp:
        lsb = cand_temp & -cand_temp
        q = lsb.bit_length() - 1
        cand_temp ^= lsb
        S_temp = S_mask
        blocked = False
        while S_temp:
            i_lsb = S_temp & -S_temp
            i = i_lsb.bit_length() - 1
            S_temp ^= i_lsb
            if S_mask & im[i]:
                blocked = True
                break
        if not blocked:
            score += 1
    return score


def _greedy_with_coverage(candidates_mask):
    """Greedy construction using coverage heuristic on a subset of candidates."""
    S_mask = 0
    remaining = candidates_mask
    while remaining:
        # Find all addable points
        addable = []
        cand_temp = remaining
        while cand_temp:
            lsb = cand_temp & -cand_temp
            p = lsb.bit_length() - 1
            cand_temp ^= lsb
            if _can_add(p, S_mask):
                addable.append(p)
        
        if not addable:
            break
        
        # If few candidates, pick best by coverage; otherwise random
        if len(addable) <= 20:
            best_p = addable[0]
            best_score = -1
            for p in addable:
                sc = _coverage_score(p, S_mask, remaining)
                if sc > best_score:
                    best_score = sc
                    best_p = p
        else:
            best_p = random.choice(addable)
        
        S_mask |= (1 << best_p)
        remaining ^= (1 << best_p)
    
    return S_mask


def _branch_and_bound(S_mask, target_size, deadline):
    """
    Branch and bound to try to find a smaller maximal no-isosceles set.
    We try to remove points from S_mask while maintaining maximality.
    Returns a potentially smaller mask.
    """
    if bin(S_mask).count('1') <= target_size:
        return S_mask
    
    # Get list of points in S_mask
    pts = []
    m = S_mask
    while m:
        lsb = m & -m
        p = lsb.bit_length() - 1
        pts.append(p)
        m ^= lsb
    
    # Try to find a smaller maximal set by removing points
    # Use a heuristic: try removing points that are "least critical"
    # A point is less critical if removing it still leaves the set maximal
    
    # First, try simple pruning
    pruned = _prune(S_mask)
    if bin(pruned).count('1') <= target_size:
        return pruned
    
    # Try local search: swap points
    current = pruned
    current_size = bin(current).count('1')
    
    for _ in range(20):
        if time.time() > deadline:
            break
        if current_size <= target_size:
            break
        
        # Try removing a point
        found = False
        pts_in_current = []
        m = current
        while m:
            lsb = m & -m
            p = lsb.bit_length() - 1
            pts_in_current.append(p)
            m ^= lsb
        random.shuffle(pts_in_current)
        
        for p in pts_in_current:
            new_S = current ^ (1 << p)
            if _is_maximal(new_S):
                current = new_S
                current_size -= 1
                found = True
                break
        
        if found:
            continue
        
        # Try swap
        swap_found = False
        for p in pts_in_current:
            if time.time() > deadline:
                break
            S_without = current ^ (1 << p)
            not_S = ALL_MASK ^ S_without
            cand_bit = not_S
            candidates = []
            while cand_bit:
                lsb = cand_bit & -cand_bit
                q = lsb.bit_length() - 1
                cand_bit ^= lsb
                if _can_add(q, S_without):
                    candidates.append(q)
            
            if not candidates:
                continue
            
            for q in random.sample(candidates, min(len(candidates), 10)):
                new_S = S_without | (1 << q)
                if _is_maximal(new_S):
                    current = new_S
                    swap_found = True
                    break
            if swap_found:
                break
        
        if not swap_found:
            break
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 285.0

    best_mask = 0
    best_size = TOTAL_POINTS + 1

    # Phase 1: Multiple greedy constructions with different orderings
    num_trials = 150

    for trial in range(num_trials):
        if time.time() > deadline - 40:
            break

        if trial == 0:
            order = list(range(TOTAL_POINTS))
        elif trial % 3 == 1:
            order = list(range(TOTAL_POINTS))
            random.shuffle(order)
        elif trial % 3 == 2:
            # Heuristic: order by centrality (more isosceles relations = more blocking)
            scores = []
            for p in range(TOTAL_POINTS):
                cnt = 0
                im = _iso_mask[p]
                for i in range(TOTAL_POINTS):
                    if i != p:
                        cnt += bin(im[i]).count('1')
                scores.append((random.random(), cnt, p))
            scores.sort(key=lambda x: (-x[1], x[0]))
            order = [s[2] for s in scores]
        else:
            # Use coverage-based greedy
            candidates_mask = ALL_MASK
            S_mask = _greedy_with_coverage(candidates_mask)
            size = bin(S_mask).count('1')
            if size < best_size:
                best_size = size
                best_mask = S_mask
            pruned = _prune(S_mask)
            pruned_size = bin(pruned).count('1')
            if pruned_size < best_size:
                best_size = pruned_size
                best_mask = pruned
            continue

        S_mask = _greedy_construct(order)
        size = bin(S_mask).count('1')

        if size < best_size:
            best_size = size
            best_mask = S_mask

        # Try to prune
        pruned = _prune(S_mask)
        pruned_size = bin(pruned).count('1')
        if pruned_size < best_size:
            best_size = pruned_size
            best_mask = pruned

    # Phase 2: Branch and bound / local search on best
    if time.time() < deadline - 30 and best_mask:
        current = _branch_and_bound(best_mask, best_size, deadline - 10)
        current_size = bin(current).count('1')
        if current_size < best_size:
            best_size = current_size
            best_mask = current

    # Phase 3: More random restarts with pruning
    if time.time() < deadline - 20:
        for _ in range(80):
            if time.time() > deadline - 10:
                break
            order = list(range(TOTAL_POINTS))
            random.shuffle(order)
            S_mask = _greedy_construct(order)
            pruned = _prune(S_mask)
            sz = bin(pruned).count('1')
            if sz < best_size:
                best_size = sz
                best_mask = pruned

    # Final local search
    if time.time() < deadline - 5 and best_mask:
        final = _branch_and_bound(best_mask, best_size, deadline - 2)
        final_size = bin(final).count('1')
        if final_size < best_size:
            best_size = final_size
            best_mask = final

    return _mask_to_list(best_mask)

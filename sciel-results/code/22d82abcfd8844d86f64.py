import random
import time
import itertools
import sys

N = 16
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N
DEADLINE = time.time() + 290  # Safety margin

# Precompute squared distances
# dist[i][j]
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute isosceles masks
# iso_mask[i][j] is a bitmask of points k such that triangle (i, j, k) is isosceles
# Note: k is distinct from i and j.
iso_mask = [[0] * NUM_PTS for _ in range(NUM_PTS)]

for i in range(NUM_PTS):
    for j in range(NUM_PTS):
        if i == j:
            continue
        mask = 0
        dij = dist[i][j]
        for k in range(NUM_PTS):
            if k == i or k == j:
                continue
            dik = dist[i][k]
            djk = dist[j][k]
            # Check isosceles conditions
            if dik == djk or dik == dij or djk == dij:
                mask |= (1 << k)
        iso_mask[i][j] = mask

# Precompute for each point q, the list of (i,j) pairs such that (q,i,j) is isosceles
# This helps with fast maximality checking
# block_pairs[q] = list of (i,j) where i<j and triangle(q,i,j) is isosceles
block_pairs = [[] for _ in range(NUM_PTS)]
for q in range(NUM_PTS):
    for i in range(NUM_PTS):
        if i == q:
            continue
        for j in range(i + 1, NUM_PTS):
            if j == q:
                continue
            # Check if (q, i, j) is isosceles
            dq_i = dist[q][i]
            dq_j = dist[q][j]
            di_j = dist[i][j]
            if dq_i == dq_j or dq_i == di_j or dq_j == di_j:
                block_pairs[q].append((i, j))

def can_add(q, mask_S):
    """
    Check if point q can be added to the set represented by mask_S.
    mask_S is an integer bitmask of indices.
    """
    # Iterate over all points i in mask_S
    m = mask_S
    while m:
        # Get the lowest set bit index
        lsb = m & -m
        i = lsb.bit_length() - 1
        m ^= lsb
        
        # Check if there is any j in mask_S such that (q, i, j) is isosceles
        # iso_mask[q][i] contains all such j's
        if mask_S & iso_mask[q][i]:
            return False
            
    return True

def is_maximal_fast(mask_S):
    """
    Fast check if the set represented by mask_S is maximal.
    Uses precomputed block_pairs.
    """
    # For every point q NOT in mask_S, check if it can be added
    # If any can be added, it's not maximal.
    for q in range(NUM_PTS):
        if mask_S & (1 << q):
            continue
        # Check if q is blocked by any pair in mask_S
        blocked = False
        for (i, j) in block_pairs[q]:
            if (mask_S & (1 << i)) and (mask_S & (1 << j)):
                blocked = True
                break
        if not blocked:
            return False
    return True

def greedy_maximal(start_mask=0, rng=None, heuristic_order=False):
    """
    Greedily add points to start_mask until maximal.
    Returns the final mask.
    If heuristic_order is True, use a heuristic that prioritizes points with high blocking power.
    """
    if rng is None:
        rng = random.Random()
        
    mask_S = start_mask
    remaining = [i for i in range(NUM_PTS) if not (mask_S & (1 << i))]
    
    if heuristic_order:
        # Score each remaining point by how many other points it blocks
        # A point that blocks many others is more valuable to include
        def blocking_score(q):
            score = 0
            for i in remaining:
                if i == q:
                    continue
                if mask_S | (1 << q) | (1 << i) and not can_add(i, mask_S | (1 << q)):
                    score += 1
            return score
        
        # Sort by descending blocking score, with random tie-breaking
        scored = [(blocking_score(q), q) for q in remaining]
        rng.shuffle(scored)  # Randomize before sort for tie-breaking
        scored.sort(key=lambda x: -x[0])
        remaining = [q for _, q in scored]
    else:
        rng.shuffle(remaining)
    
    for q in remaining:
        if can_add(q, mask_S):
            mask_S |= (1 << q)
            
    return mask_S

def mask_to_list(mask_S):
    """Convert bitmask to list of point indices."""
    pts = []
    m = mask_S
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        pts.append(i)
        m ^= lsb
    return pts

def local_search_swap(best_mask, best_len, rng, deadline):
    """
    Try swap moves: remove one point, add another.
    Returns (new_mask, new_len, improved)
    """
    pts_in_set = mask_to_list(best_mask)
    pts_not_in_set = [i for i in range(NUM_PTS) if not (best_mask & (1 << i))]
    
    rng.shuffle(pts_in_set)
    rng.shuffle(pts_not_in_set)
    
    for p_remove in pts_in_set:
        if time.time() > deadline:
            return best_mask, best_len, False
        
        for p_add in pts_not_in_set:
            if time.time() > deadline:
                return best_mask, best_len, False
            
            # Try swapping p_remove with p_add
            temp_mask = best_mask & ~(1 << p_remove)
            temp_mask |= (1 << p_add)
            
            # Check if valid (no isosceles) and maximal
            # First check validity
            valid = True
            m = temp_mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                m ^= lsb
                if temp_mask & iso_mask[p_add][i]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check maximality
            if is_maximal_fast(temp_mask):
                new_len = bin(temp_mask).count('1')
                if new_len < best_len:
                    return temp_mask, new_len, True
    
    return best_mask, best_len, False

def construct():
    """Construct a minimal maximal no-isosceles set."""
    best_mask = 0
    best_len = float('inf')
    rng = random.Random()
    
    # Phase 1: Random Greedy Restarts to find a good starting point
    while time.time() < DEADLINE - 30:
        # Alternate between random order and heuristic order
        use_heuristic = rng.random() < 0.5
        mask = greedy_maximal(rng=rng, heuristic_order=use_heuristic)
        curr_len = bin(mask).count('1')
        if curr_len < best_len:
            best_mask = mask
            best_len = curr_len
            
    # Phase 2: Local Search (Removal, Refill, and Swap)
    no_improve_count = 0
    while time.time() < DEADLINE - 5:
        improved = False
        
        # Try removal + refill
        pts_in_set = mask_to_list(best_mask)
        rng.shuffle(pts_in_set)
        
        for p in pts_in_set:
            if time.time() > DEADLINE - 5:
                return [POINTS[i] for i in mask_to_list(best_mask)]
            
            temp_mask = best_mask & ~(1 << p)
            
            # Check 1: Is it still maximal without p?
            if is_maximal_fast(temp_mask):
                best_mask = temp_mask
                best_len -= 1
                improved = True
                no_improve_count = 0
                break  # Restart search with new best
            
            # Check 2: Try to refill temp_mask to maximal and see if size decreases
            # Use heuristic order for refill
            new_mask = greedy_maximal(start_mask=temp_mask, rng=rng, heuristic_order=True)
            new_len = bin(new_mask).count('1')
            
            if new_len < best_len:
                best_mask = new_mask
                best_len = new_len
                improved = True
                no_improve_count = 0
                break  # Restart search with new best
        
        if improved:
            continue
        
        # Try swap moves
        new_mask, new_len, swap_improved = local_search_swap(best_mask, best_len, rng, DEADLINE - 5)
        if swap_improved:
            best_mask = new_mask
            best_len = new_len
            no_improve_count = 0
            continue
        
        # If no improvement, do a random restart to escape local minimum
        no_improve_count += 1
        if no_improve_count >= 3:
            no_improve_count = 0
            mask = greedy_maximal(rng=rng, heuristic_order=rng.random() < 0.5)
            curr_len = bin(mask).count('1')
            if curr_len < best_len:
                best_mask = mask
                best_len = curr_len

    return [POINTS[i] for i in mask_to_list(best_mask)]

import numpy as np
import time
import random

N = 16
NUM_PTS = N * N
ALL_MASK = (1 << NUM_PTS) - 1

# Precompute points
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
# dist[i, j] is squared distance between point i and point j
dist = np.zeros((NUM_PTS, NUM_PTS), dtype=np.int32)
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i, j] = d
        dist[j, i] = d

# Precompute conflict masks
# conflict[i][j] is a bitmask of points k such that (i, j, k) is isosceles
# We store as Python integers for fast bitwise ops
conflict_masks = [[0] * NUM_PTS for _ in range(NUM_PTS)]

for i in range(NUM_PTS):
    for j in range(i + 1, NUM_PTS):
        d_ij = dist[i, j]
        mask = 0
        # Vectorized check for k
        # d_ik = dist[i, :]
        # d_jk = dist[j, :]
        # is_iso = (d_ij == d_ik) | (d_ij == d_jk) | (d_ik == d_jk)
        d_ik = dist[i]
        d_jk = dist[j]
        is_iso = (d_ij == d_ik) | (d_ij == d_jk) | (d_ik == d_jk)
        # Exclude i and j
        is_iso[i] = False
        is_iso[j] = False
        # Convert boolean array to bitmask
        # Get indices of True
        idxs = np.where(is_iso)[0]
        for k in idxs:
            mask |= (1 << k)
        conflict_masks[i][j] = mask
        conflict_masks[j][i] = mask

def popcount(x):
    return bin(x).count('1')

def get_available(selected_mask, blocked_mask):
    return ALL_MASK & ~selected_mask & ~blocked_mask

def greedy_construct(seed):
    rng = random.Random(seed)
    selected_mask = 0
    blocked_mask = 0
    available_mask = ALL_MASK
    
    while available_mask:
        # Pick a point from available
        # Heuristic: Pick point that blocks most other available points
        best_p = -1
        best_cover_count = -1
        # Sample a few to speed up? Or iterate all?
        # Iterating all 256 is fast.
        candidates = []
        temp_mask = available_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            p = lsb.bit_length() - 1
            candidates.append(p)
            temp_mask ^= lsb
            
        for p in candidates:
            # Calculate cover
            cover = (1 << p)
            # Points blocked by adding p
            # New blocked = old_blocked | union(conflict[p][i] for i in selected)
            # We only care about intersection with available
            temp_block = 0
            temp_sel = selected_mask
            while temp_sel:
                lsb = temp_sel & -temp_sel
                i = lsb.bit_length() - 1
                temp_block |= conflict_masks[p][i]
                temp_sel ^= lsb
            cover |= temp_block
            cover &= available_mask
            count = popcount(cover)
            if count > best_cover_count:
                best_cover_count = count
                best_p = p
                
        if best_p == -1:
            break
            
        # Add best_p
        selected_mask |= (1 << best_p)
        # Update blocked
        temp_block = 0
        temp_sel = selected_mask & ~(1 << best_p) # Exclude p itself from pairs? No, pairs are (p, i)
        # Actually, pairs involving p and existing selected points
        temp_sel = selected_mask & ~(1 << best_p)
        while temp_sel:
            lsb = temp_sel & -temp_sel
            i = lsb.bit_length() - 1
            temp_block |= conflict_masks[best_p][i]
            temp_sel ^= lsb
        blocked_mask |= temp_block
        
        available_mask = get_available(selected_mask, blocked_mask)
        
    return selected_mask

def search(selected_mask, blocked_mask, best_size_holder, deadline):
    # Check time
    if time.time() > deadline:
        return
        
    available_mask = get_available(selected_mask, blocked_mask)
    
    if available_mask == 0:
        size = popcount(selected_mask)
        if size < best_size_holder[0]:
            best_size_holder[0] = size
            best_size_holder[1] = selected_mask
        return
        
    current_size = popcount(selected_mask)
    if current_size >= best_size_holder[0]:
        return
        
    # Lower Bound
    # We need to cover all points in available_mask
    # Each point p in available covers: {p} U {k in available | k in conflict[p][i] for some i in selected}
    
    max_cover = 0
    candidates = []
    temp_mask = available_mask
    while temp_mask:
        lsb = temp_mask & -temp_mask
        p = lsb.bit_length() - 1
        candidates.append(p)
        temp_mask ^= lsb
        
    for p in candidates:
        cover = (1 << p)
        temp_block = 0
        temp_sel = selected_mask
        while temp_sel:
            lsb = temp_sel & -temp_sel
            i = lsb.bit_length() - 1
            temp_block |= conflict_masks[p][i]
            temp_sel ^= lsb
        cover |= temp_block
        cover &= available_mask
        count = popcount(cover)
        if count > max_cover:
            max_cover = count
        # Store cover count for sorting
        candidates[-1] = (p, count)
        
    if max_cover == 0:
        # This shouldn't happen if available is not empty, because p covers itself
        return
        
    lb = (popcount(available_mask) + max_cover - 1) // max_cover
    if current_size + lb >= best_size_holder[0]:
        return
        
    # Sort candidates by cover count descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    for p, _ in candidates:
        if time.time() > deadline:
            return
            
        new_selected_mask = selected_mask | (1 << p)
        
        # Update blocked
        temp_block = 0
        temp_sel = selected_mask
        while temp_sel:
            lsb = temp_sel & -temp_sel
            i = lsb.bit_length() - 1
            temp_block |= conflict_masks[p][i]
            temp_sel ^= lsb
            
        new_blocked_mask = blocked_mask | temp_block
        
        search(new_selected_mask, new_blocked_mask, best_size_holder, deadline)
        
        # If we found a solution of size current_size + 1, and lb was 1, 
        # we might prune others? 
        # If best_size_holder[0] == current_size + 1, we can break?
        # No, there might be another branch with same size. But we only need one.
        # However, if we found a solution of size S, and we are exploring size S, 
        # we don't need to explore other branches that will result in size >= S.
        # But other branches might also result in size S.
        # Since we just want *a* minimal set, finding one of size S is enough if we know S is minimal.
        # But we don't know S is minimal until we prune everything.
        # However, if we found a solution of size `current_size + lb`, and `lb` is the theoretical minimum,
        # then `current_size + lb` is the best possible for this branch.
        # If `best_size_holder[0]` is updated to this value, we can prune other branches at this level
        # if their LB is >= this value.
        
        if best_size_holder[0] == current_size + lb:
             # We found the best possible for this node's subtree?
             # Not necessarily, other children might have different LBs?
             # No, LB is calculated for the current node.
             # If we found a solution of size `current_size + lb`, it is the best possible from this node.
             # We can break the loop if we want to find just one?
             # But we want to be sure.
             # Actually, if we found a solution of size S, and S is the lower bound, 
             # then no other branch from this node can be better.
             # So we can break.
             break

def construct():
    deadline = time.time() + 290
    
    # Initial greedy
    best_size_holder = [float('inf'), 0]
    best_greedy_mask = 0
    
    for i in range(20):
        if time.time() > deadline:
            break
        mask = greedy_construct(i)
        size = popcount(mask)
        if size < best_size_holder[0]:
            best_size_holder[0] = size
            best_size_holder[1] = mask
            best_greedy_mask = mask
            
    # Branch and Bound
    # Start with empty set
    search(0, 0, best_size_holder, deadline)
    
    # If B&B didn't improve, use best greedy
    final_mask = best_size_holder[1]
    
    # Convert mask to list of points
    result = []
    for i in range(NUM_PTS):
        if (final_mask >> i) & 1:
            result.append(POINTS[i])
            
    return result

# EVOLVE-BLOCK-END

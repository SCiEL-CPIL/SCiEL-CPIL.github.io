import random
import itertools
import numpy as np
import time

N = 16
M = N * N

# Precompute conflict masks
# conflict_mask[i][j] is a bitset of points k such that i,j,k form an isosceles triangle
# We only need i < j, but store symmetric for ease
conflict_mask = [[0] * M for _ in range(M)]

def dist_sq(p, q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

# Generate all points
points = [(x, y) for y in range(N) for x in range(N)]
point_to_idx = {p: i for i, p in enumerate(points)}

# Precompute distances between all pairs? Or just compute on the fly for masks.
# M=256, pairs ~ 32k. Dist calc is cheap.

for i in range(M):
    for j in range(i + 1, M):
        p_i = points[i]
        p_j = points[j]
        d_ij = dist_sq(p_i, p_j)
        mask = 0
        for k in range(M):
            if k == i or k == j:
                continue
            p_k = points[k]
            d_ik = dist_sq(p_i, p_k)
            d_jk = dist_sq(p_j, p_k)
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                mask |= (1 << k)
        conflict_mask[i][j] = mask
        conflict_mask[j][i] = mask

def get_indices(mask):
    indices = []
    m = mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
    return indices

def check_state(S_mask):
    """
    Returns (is_valid, is_maximal, size)
    """
    indices = get_indices(S_mask)
    size = len(indices)
    
    # Check Validity: No isosceles triangle in S
    # For all i < j < k in indices, check if they form isosceles
    # Optimization: If we maintain S as valid, we only need to check new additions.
    # But here we check from scratch for simplicity in local search.
    # Since |S| is small, O(|S|^3) is fine.
    is_valid = True
    for i in range(size):
        for j in range(i+1, size):
            for k in range(j+1, size):
                idx_i = indices[i]
                idx_j = indices[j]
                idx_k = indices[k]
                # Check if any of the three points is in the conflict mask of the other two
                # conflict_mask[a][b] contains c if a,b,c is isosceles.
                # So if bit idx_k is in conflict_mask[idx_i][idx_j], it's isosceles.
                if (conflict_mask[idx_i][idx_j] >> idx_k) & 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
    
    if not is_valid:
        return False, False, size

    # Check Maximality
    # Calculate cover: all points that would form an isosceles with any pair in S
    cover = 0
    for i in range(size):
        idx_i = indices[i]
        for j in range(i+1, size):
            idx_j = indices[j]
            cover |= conflict_mask[idx_i][idx_j]
    
    # Points not in S
    not_in_S = (~S_mask) & ((1 << M) - 1)
    # Uncovered points: not in S and not in cover
    uncovered = not_in_S & ~cover
    
    is_maximal = (uncovered == 0)
    
    return True, is_maximal, size

def greedy_construct(order):
    S_mask = 0
    # To speed up validity check during construction, we can check against existing points.
    # But simple check is O(|S|^2) per insertion.
    for idx in order:
        # Check if idx is valid with all pairs in S_mask
        # Actually, just check if idx forms isosceles with any pair in S_mask
        # This is equivalent to: for all i,j in S_mask, idx not in conflict_mask[i][j]
        # AND for all i in S_mask, the pair (idx, i) doesn't form isosceles with any other j in S_mask?
        # No, if S_mask is valid, we just need to ensure (idx, i, j) is not isosceles for any i,j in S_mask.
        # This is exactly: idx not in conflict_mask[i][j] for any i,j in S_mask.
        
        s = S_mask
        valid = True
        while s:
            lsb = s & -s
            i = lsb.bit_length() - 1
            s ^= lsb
            s2 = S_mask
            while s2:
                lsb2 = s2 & -s2
                j = lsb2.bit_length() - 1
                s2 ^= lsb2
                if i == j:
                    continue
                if (conflict_mask[i][j] >> idx) & 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S_mask |= (1 << idx)
    return S_mask

def local_search(S_mask, max_iters=5000):
    """
    Try to minimize S_mask while keeping it valid and maximal.
    """
    current_S = S_mask
    current_size, is_valid, is_maximal = check_state(current_S)[2], check_state(current_S)[0], check_state(current_S)[1]
    if not is_valid:
        return current_S, current_size
        
    # If not maximal, we can't minimize it directly as "minimal maximal".
    # We need to make it maximal first? 
    # Actually, if it's not maximal, it's not a candidate.
    # But we can try to add points to make it maximal, then minimize?
    # Or, we can just accept that this seed is bad and try another.
    # However, a non-maximal set might be "close" to a maximal one.
    # Let's try to add points that cover the most uncovered points.
    
    for _ in range(max_iters):
        is_valid, is_maximal, size = check_state(current_S)
        if not is_valid:
            break
            
        if is_maximal:
            # Try to remove points
            indices = get_indices(current_S)
            random.shuffle(indices)
            removed = False
            for idx in indices:
                new_S = current_S & ~(1 << idx)
                new_valid, new_maximal, new_size = check_state(new_S)
                if new_valid and new_maximal:
                    current_S = new_S
                    removed = True
                    break
            if removed:
                continue
                
            # Try to swap to escape local minimum (keep size same, might allow future removals)
            # Pick random in, random out
            in_indices = get_indices(current_S)
            if not in_indices:
                break
            out_indices = get_indices(~current_S & ((1 << M) - 1))
            if not out_indices:
                break
            
            u = random.choice(in_indices)
            v = random.choice(out_indices)
            
            new_S = (current_S & ~(1 << u)) | (1 << v)
            new_valid, new_maximal, new_size = check_state(new_S)
            if new_valid and new_maximal:
                current_S = new_S
                # Don't break, continue searching
            else:
                # If no removal and swap failed, try another swap
                pass
        else:
            # Not maximal. Try to add a point to cover uncovered points.
            # This increases size, but is necessary to reach a maximal state.
            # However, we want MINIMAL maximal. 
            # Adding points is bad. 
            # Maybe we should just discard this seed if it's not maximal after some tries?
            # Or, add the point that covers the most uncovered points.
            
            # Calculate uncovered
            cover = 0
            indices = get_indices(current_S)
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    cover |= conflict_mask[indices[i]][indices[j]]
            
            not_in_S = (~current_S) & ((1 << M) - 1)
            uncovered = not_in_S & ~cover
            
            if uncovered == 0:
                # Should be maximal, but check_state said no? 
                # check_state uses same logic. 
                # Maybe I made a mistake in check_state?
                # Let's re-verify check_state logic.
                # cover is union of conflicts.
                # uncovered = not_in_S & ~cover.
                # If uncovered == 0, it is maximal.
                # So if is_maximal is False, uncovered != 0.
                pass
            
            # Find point in uncovered that covers most other uncovered points?
            # Or just add a random uncovered point?
            # Adding a point might allow removing others later.
            
            unc_indices = get_indices(uncovered)
            if not unc_indices:
                break
                
            # Heuristic: Add point that is in conflict with most pairs in S? 
            # No, we want to cover uncovered points.
            # Let's just add a random one to move towards maximality.
            add_idx = random.choice(unc_indices)
            
            # Check if adding is valid
            s = current_S
            valid = True
            while s:
                lsb = s & -s
                i = lsb.bit_length() - 1
                s ^= lsb
                s2 = current_S
                while s2:
                    lsb2 = s2 & -s2
                    j = lsb2.bit_length() - 1
                    s2 ^= lsb2
                    if i == j:
                        continue
                    if (conflict_mask[i][j] >> add_idx) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_S |= (1 << add_idx)
            else:
                # Can't add any uncovered point? 
                # Then it's not maximal, but we can't fix it by adding?
                # This shouldn't happen if uncovered is non-empty and valid.
                # If no uncovered point can be added, it means for every p in uncovered, 
                # there is a pair in S that conflicts with p.
                # But that means p is in cover!
                # Contradiction: p is in uncovered (not in cover) and p is in cover.
                # So if uncovered is non-empty, at least one point in uncovered is NOT in cover.
                # If it's not in cover, it doesn't form isosceles with any pair in S.
                # So it CAN be added.
                # So valid should be True.
                break

    final_valid, final_maximal, final_size = check_state(current_S)
    if final_valid and final_maximal:
        return current_S, final_size
    else:
        return current_S, M + 1 # Invalid

def construct():
    best_S = 0
    best_size = M + 1
    start_time = time.time()
    time_limit = 290.0
    
    # Multiple random restarts
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        order = list(range(M))
        random.shuffle(order)
        
        S = greedy_construct(order)
        
        # Local Search
        S, size = local_search(S, max_iters=2000)
        
        if size < best_size:
            best_size = size
            best_S = S
            
        # If we found a very small set, we might break early?
        # Theoretical lower bound?
        # For 16x16, known results?
        # Let's just run for a bit.
        
    if best_S == 0:
        return []
        
    result = []
    s = best_S
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        s ^= lsb
        result.append(points[idx])
    return result

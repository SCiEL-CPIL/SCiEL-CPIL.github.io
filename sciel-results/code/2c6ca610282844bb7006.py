import time
import random

# Precomputation
N = 256
four_aps = []
three_aps = []

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        ap = (a, b, c, e)
        four_aps.append(ap)

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        ap = (a, b, c)
        three_aps.append(ap)

# Precompute conflict masks for 4-APs
# For each element x, list of masks of the other 3 elements in each 4-AP containing x
four_ap_conflict_masks = {i: [] for i in range(1, N + 1)}
for ap in four_aps:
    a, b, c, e = ap
    four_ap_conflict_masks[a].append((1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
    four_ap_conflict_masks[b].append((1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
    four_ap_conflict_masks[c].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1)))
    four_ap_conflict_masks[e].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)))

# Precompute 3-AP partner masks
# For each element x, list of masks of the other 2 elements in each 3-AP containing x
three_ap_partner_masks = {i: [] for i in range(1, N + 1)}
for ap in three_aps:
    a, b, c = ap
    three_ap_partner_masks[a].append((1 << (b - 1)) | (1 << (c - 1)))
    three_ap_partner_masks[b].append((1 << (a - 1)) | (1 << (c - 1)))
    three_ap_partner_masks[c].append((1 << (a - 1)) | (1 << (b - 1)))


def set_to_mask(A_set):
    mask = 0
    for x in A_set:
        mask |= (1 << (x - 1))
    return mask


def is_4ap_free_addition_mask(A_mask, x):
    for cmask in four_ap_conflict_masks[x]:
        if (A_mask & cmask) == cmask:
            return False
    return True


def delta_3aps_for_addition_mask(A_mask, x):
    count = 0
    for pmask in three_ap_partner_masks[x]:
        if (A_mask & pmask) == pmask:
            count += 1
    return count


def count_3aps_mask(A_mask):
    count = 0
    for a, b, c in three_aps:
        if (A_mask & (1 << (a - 1))) and (A_mask & (1 << (b - 1))) and (A_mask & (1 << (c - 1))):
            count += 1
    return count


def greedy_initial_mask(seed_val=0):
    random.seed(seed_val)
    A_mask = 0
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        if is_4ap_free_addition_mask(A_mask, x):
            A_mask |= (1 << (x - 1))
    return A_mask


def local_search_mask(A_mask_init, deadline):
    A_mask = A_mask_init
    current_score = count_3aps_mask(A_mask)
    
    # Precompute L_x for all x in A
    L = {}
    for x in range(1, N + 1):
        if A_mask & (1 << (x - 1)):
            A_mask_without_x = A_mask & ~(1 << (x - 1))
            L[x] = 0
            for pmask in three_ap_partner_masks[x]:
                if (A_mask_without_x & pmask) == pmask:
                    L[x] += 1
    
    while time.time() < deadline:
        improved = False
        
        # Try Additions
        best_add_x = -1
        best_add_delta = 0
        for x in range(1, N + 1):
            if A_mask & (1 << (x - 1)):
                continue
            if is_4ap_free_addition_mask(A_mask, x):
                delta = delta_3aps_for_addition_mask(A_mask, x)
                if delta > best_add_delta:
                    best_add_delta = delta
                    best_add_x = x
        
        if best_add_x != -1:
            A_mask |= (1 << (best_add_x - 1))
            current_score += best_add_delta
            improved = True
            continue
        
        # Try Swaps
        best_swap = None
        best_swap_score = current_score
        
        for x in range(1, N + 1):
            if not (A_mask & (1 << (x - 1))):
                continue
            
            A_mask_without_x = A_mask & ~(1 << (x - 1))
            L_x = L[x]
            
            for y in range(1, N + 1):
                if y == x or (A_mask & (1 << (y - 1))):
                    continue
                if not is_4ap_free_addition_mask(A_mask_without_x, y):
                    continue
                
                G_y = 0
                for pmask in three_ap_partner_masks[y]:
                    if (A_mask_without_x & pmask) == pmask:
                        G_y += 1
                
                new_score = current_score - L_x + G_y
                if new_score > best_swap_score:
                    best_swap_score = new_score
                    best_swap = (x, y)
        
        if best_swap:
            x, y = best_swap
            A_mask = (A_mask & ~(1 << (x - 1))) | (1 << (y - 1))
            current_score = best_swap_score
            improved = True
            continue
        
        if not improved:
            break
            
    return A_mask


def polish_pass(A_mask, deadline):
    current_score = count_3aps_mask(A_mask)
    
    while time.time() < deadline:
        improved = False
        
        # Try removals
        for x in range(1, N + 1):
            if not (A_mask & (1 << (x - 1))):
                continue
            A_mask_trial = A_mask & ~(1 << (x - 1))
            new_score = count_3aps_mask(A_mask_trial)
            if new_score > current_score:
                A_mask = A_mask_trial
                current_score = new_score
                improved = True
                break
        
        if improved:
            continue
        
        # Try additions
        for x in range(1, N + 1):
            if A_mask & (1 << (x - 1)):
                continue
            if is_4ap_free_addition_mask(A_mask, x):
                delta = delta_3aps_for_addition_mask(A_mask, x)
                if delta > 0:
                    A_mask |= (1 << (x - 1))
                    current_score += delta
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swaps
        found_swap = False
        for x in range(1, N + 1):
            if not (A_mask & (1 << (x - 1))):
                continue
            A_mask_without_x = A_mask & ~(1 << (x - 1))
            L_x = 0
            for pmask in three_ap_partner_masks[x]:
                if (A_mask_without_x & pmask) == pmask:
                    L_x += 1
            
            for y in range(1, N + 1):
                if y == x or (A_mask & (1 << (y - 1))):
                    continue
                if not is_4ap_free_addition_mask(A_mask_without_x, y):
                    continue
                G_y = 0
                for pmask in three_ap_partner_masks[y]:
                    if (A_mask_without_x & pmask) == pmask:
                        G_y += 1
                
                new_score = current_score - L_x + G_y
                if new_score > current_score:
                    A_mask = A_mask_without_x | (1 << (y - 1))
                    current_score = new_score
                    found_swap = True
                    break
            
            if found_swap:
                break
        
        if found_swap:
            continue
        
        break
    
    return A_mask


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s budget, 5s safety margin
    
    best_mask = None
    best_score = -1
    
    # Generate diverse initial candidates using bitmask greedy
    candidates = []
    
    # Candidate 0: Deterministic Greedy
    candidates.append(greedy_initial_mask(0))
    
    # Candidates 1-15: Randomized Greedies
    for i in range(1, 16):
        if time.time() > deadline - 20:
            break
        candidates.append(greedy_initial_mask(i))
    
    # Evaluate and optimize each candidate
    for idx, init_mask in enumerate(candidates):
        if time.time() > deadline - 15:
            break
            
        # Quick score check
        s = count_3aps_mask(init_mask)
        
        # Run local search with remaining time
        remaining = deadline - time.time()
        if remaining <= 5:
            break
            
        # Allocate time: give more time to promising candidates
        # Use a fraction of remaining time, leaving buffer for later candidates
        search_deadline = min(deadline - 10, time.time() + max(2, remaining / 3))
        optimized = local_search_mask(init_mask, search_deadline)
        
        # Polish pass
        polish_deadline = min(deadline - 5, time.time() + 3)
        optimized = polish_pass(optimized, polish_deadline)
        
        opt_score = count_3aps_mask(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_mask = optimized
    
    # Final polish on the best set with remaining time
    if best_mask is not None:
        final_deadline = deadline - 2
        best_mask = polish_pass(best_mask, final_deadline)
    
    if best_mask is None:
        # Fallback
        best_mask = candidates[0] if candidates else greedy_initial_mask(0)
    
    # Convert to sorted list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
    
    return sorted(result)

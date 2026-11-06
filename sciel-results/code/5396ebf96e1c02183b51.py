import time
import random
import math

N = 256

def construct():
    # Precomputation
    four_ap_masks = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_val = a + 3 * d
            
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (d_val - 1))
            four_ap_masks[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (d_val - 1))
            four_ap_masks[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (d_val - 1))
            four_ap_masks[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            four_ap_masks[d_val].append(mask_d)

    def count_3aps_involving(x, A_mask):
        # Count 3-APs that include x and other elements in A_mask
        count = 0
        # x is middle: (x-d, x, x+d)
        for d in range(1, min(x - 1, N - x) + 1):
            a = x - d
            c = x + d
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                count += 1
        # x is first: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if b > N or c > N: break
            if ((A_mask >> (b - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                count += 1
        # x is third: (x-2d, x-d, x)
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a < 1: break
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (b - 1)) & 1):
                count += 1
        return count

    def can_add(A_mask, x):
        idx = x - 1
        if (A_mask >> idx) & 1:
            return False
        for mask in four_ap_masks[x]:
            if (A_mask & mask) == mask:
                return False
        return True

    def total_3aps(A_mask):
        # Count all 3-APs in A_mask
        # To avoid double counting, we can sum count_3aps_involving(x, A_mask) for all x in A
        # and divide by 3? No, each 3-AP has 3 elements.
        # So sum(count_3aps_involving(x)) / 3 is the total count.
        total = 0
        for x in range(1, N + 1):
            if (A_mask >> (x - 1)) & 1:
                total += count_3aps_involving(x, A_mask)
        return total // 3

    # Greedy Initialization
    A_mask = 0
    for x in range(1, N + 1):
        if can_add(A_mask, x):
            A_mask |= (1 << (x - 1))
            
    current_score = total_3aps(A_mask)
    best_mask = A_mask
    best_score = current_score
    
    start_time = time.time()
    timeout = 55.0
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < timeout:
        # Pick a random element to remove
        in_set = [i + 1 for i in range(N) if (A_mask >> i) & 1]
        if not in_set:
            break
        x_remove = random.choice(in_set)
        
        # Pick a random element to add (not in set)
        not_in_set = [i + 1 for i in range(N) if not (A_mask >> i) & 1]
        if not not_in_set:
            break
        x_add = random.choice(not_in_set)
        
        # Perform move: Remove x_remove, Add x_add
        # Check if x_add can be added to (A_mask without x_remove)
        temp_mask = A_mask & ~(1 << (x_remove - 1))
        
        if not can_add(temp_mask, x_add):
            # Move invalid (creates 4-AP)
            T *= cooling
            continue
            
        new_mask = temp_mask | (1 << (x_add - 1))
        
        # Calculate score change
        # Old score: current_score
        # New score: total_3aps(new_mask)
        # To be fast, we can calculate delta.
        # Delta = - (3-APs involving x_remove in A_mask) 
        #         + (3-APs involving x_add in temp_mask)
        #         + (3-APs involving BOTH x_add and x_remove? No, they are disjoint in the "involving" count if we are careful? 
        #         Actually, 3-APs involving x_remove are gone.
        #         3-APs involving x_add are new.
        #         Are there 3-APs that involve both? No, a 3-AP has 3 elements. If it involves x_remove and x_add, it has a third element.
        #         When we removed x_remove, that 3-AP was destroyed.
        #         When we add x_add, that 3-AP is created.
        #         So:
        #         removed_count = count_3aps_involving(x_remove, A_mask)
        #         added_count = count_3aps_involving(x_add, temp_mask)
        #         new_score = current_score - removed_count + added_count
        
        removed_count = count_3aps_involving(x_remove, A_mask)
        added_count = count_3aps_involving(x_add, temp_mask)
        
        new_score = current_score - removed_count + added_count
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            A_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = A_mask
        
        T *= cooling

    return sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])

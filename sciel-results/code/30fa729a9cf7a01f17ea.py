import random
import time

def construct():
    n = 256
    best_size = 0
    best_sets = ([], [])
    
    # Bit masks for sums and differences
    # Sum range: 2 to 512. Index = sum.
    # Diff range: -255 to 255. Index = diff + 256.
    
    def try_add(set_list, sum_mask, diff_mask_self, diff_mask_other, x):
        """
        Try to add x to set_list.
        Returns (new_set_list, new_sum_mask, new_diff_mask) if successful, else None.
        """
        # 1. Check sum collisions within the set
        new_sums = 1 << (2 * x)
        # 2. Check diff collisions within the set and with other set
        new_diffs = 0
        
        for s in set_list:
            # Sum collision
            new_sums |= 1 << (x + s)
            
            # Diff collisions
            d1 = x - s
            d2 = s - x
            new_diffs |= 1 << (d1 + 256)
            new_diffs |= 1 << (d2 + 256)
        
        # Check if any new sum collides with existing sums
        if new_sums & sum_mask:
            return None
            
        # Check if any new diff collides with existing diffs in self
        if new_diffs & diff_mask_self:
            return None
            
        # Check if any new diff collides with existing diffs in other
        if new_diffs & diff_mask_other:
            return None
            
        # Update masks
        new_sum_mask = sum_mask | new_sums
        new_diff_mask = diff_mask_self | new_diffs
        new_set = set_list + [x]
        
        return new_set, new_sum_mask, new_diff_mask

    # Greedy with restarts
    start_time = time.time()
    max_time = 55.0 # Leave 5s buffer
    
    candidates = list(range(1, n + 1))
    
    while time.time() - start_time < max_time:
        random.shuffle(candidates)
        
        A = []
        B = []
        sum_mask_A = 0
        diff_mask_A = 0
        sum_mask_B = 0
        diff_mask_B = 0
        
        for x in candidates:
            # Try adding to A
            res_A = try_add(A, sum_mask_A, diff_mask_A, diff_mask_B, x)
            if res_A:
                A, sum_mask_A, diff_mask_A = res_A
                continue # If added to A, skip B for this x? 
                         # The problem doesn't say x can't be in both.
                         # "A and B need not themselves be disjoint."
                         # So we should try B as well if we can?
                         # But usually, we want to maximize total size.
                         # If we add to A, we use up some diff slots.
                         # Adding to B uses different diff slots (disjoint).
                         # So we should try to add to BOTH if possible?
                         # The loop structure above adds to A and continues.
                         # This means x is only in A.
                         # We should modify to try B independently?
                         # Or try to add to B as well?
                         # Let's try to add to B in the same iteration if possible?
                         # No, the state changes.
                         # Let's just do:
                         # 1. Try A. If success, update A.
                         # 2. Try B. If success, update B.
                         # This allows x to be in both.
            
            # Try adding to B (using updated A state? No, B addition depends on B's masks and A's diff mask)
            # If we added to A, diff_mask_A changed. So B's check `new_diffs & diff_mask_A` will be stricter.
            # This is correct.
            res_B = try_add(B, sum_mask_B, diff_mask_B, diff_mask_A, x)
            if res_B:
                B, sum_mask_B, diff_mask_B = res_B
        
        total_size = len(A) + len(B)
        if total_size > best_size:
            best_size = total_size
            best_sets = (A, B)
            
    return [best_sets[0], best_sets[1]]

import random
import time

def construct():
    n = 256
    best_size = 0
    best_sets = ([], [])
    
    # Bit masks for sums and differences
    # Sum range: 2 to 512. Index = sum.
    # Diff range: -255 to 255. Index = diff + 256.
    
    def try_add(set_list, sum_mask, diff_mask_self, diff_mask_other, x, in_set):
        """
        Try to add x to set_list.
        in_set is a boolean array of size n+1 indicating membership.
        Returns (new_set_list, new_sum_mask, new_diff_mask) if successful, else None.
        """
        if in_set[x]:
            return None

        # 1. Check sum collisions within the set
        new_sums = 1 << (2 * x)
        for s in set_list:
            new_sums |= 1 << (x + s)
            
        if new_sums & sum_mask:
            return None
            
        # 2. Check diff collisions
        new_diffs = 0
        for s in set_list:
            d1 = x - s
            d2 = s - x
            new_diffs |= 1 << (d1 + 256)
            new_diffs |= 1 << (d2 + 256)
            
        if new_diffs & diff_mask_self:
            return None
            
        if new_diffs & diff_mask_other:
            return None
            
        # Update masks
        new_sum_mask = sum_mask | new_sums
        new_diff_mask = diff_mask_self | new_diffs
        new_set = set_list + [x]
        
        return new_set, new_sum_mask, new_diff_mask

    def try_remove(set_list, sum_mask, diff_mask_self, x):
        """
        Remove x from set_list.
        Returns (new_set_list, new_sum_mask, new_diff_mask)
        """
        new_sum_mask = sum_mask
        new_sum_mask &= ~(1 << (2 * x))
        
        new_diff_mask = diff_mask_self
        
        new_set = []
        for s in set_list:
            if s == x:
                continue
            new_set.append(s)
            # Clear sums involving x and s
            new_sum_mask &= ~(1 << (x + s))
            # Clear diffs involving x and s
            d1 = x - s
            d2 = s - x
            new_diff_mask &= ~(1 << (d1 + 256))
            new_diff_mask &= ~(1 << (d2 + 256))
            
        return new_set, new_sum_mask, new_diff_mask

    start_time = time.time()
    max_time = 55.0
    
    candidates = list(range(1, n + 1))
    
    while time.time() - start_time < max_time:
        random.shuffle(candidates)
        
        A = []
        B = []
        in_A = [False] * (n + 1)
        in_B = [False] * (n + 1)
        
        sum_mask_A = 0
        diff_mask_A = 0
        sum_mask_B = 0
        diff_mask_B = 0
        
        # Greedy Phase
        for x in candidates:
            # Try adding to A
            res_A = try_add(A, sum_mask_A, diff_mask_A, diff_mask_B, x, in_A)
            if res_A:
                A, sum_mask_A, diff_mask_A = res_A
                in_A[x] = True
                continue
            
            # Try adding to B
            res_B = try_add(B, sum_mask_B, diff_mask_B, diff_mask_A, x, in_B)
            if res_B:
                B, sum_mask_B, diff_mask_B = res_B
                in_B[x] = True

        # Local Search Phase: Try to add more, or swap to enable additions
        # We try to add any unused element to A or B
        # If we can't, we try to remove an element and replace it with two? 
        # Or just random restarts are better?
        # Let's try a simple local search: 
        # 1. Try to add any candidate not in A or B to A or B.
        # 2. If no additions, try to swap: remove x from A, try to add y (not in A) to A.
        #    This doesn't increase size, but might change the mask to allow future additions.
        #    Actually, a "kick" is better: remove a random element, then greedily refill.
        
        # Let's implement a few iterations of "Try to add, if fail, kick"
        for _ in range(50): # Max 50 kick iterations per restart
            if time.time() - start_time > max_time:
                break
                
            improved = False
            # Try to add any candidate
            for x in candidates:
                if in_A[x] or in_B[x]:
                    continue
                res_A = try_add(A, sum_mask_A, diff_mask_A, diff_mask_B, x, in_A)
                if res_A:
                    A, sum_mask_A, diff_mask_A = res_A
                    in_A[x] = True
                    improved = True
                    break
                res_B = try_add(B, sum_mask_B, diff_mask_B, diff_mask_A, x, in_B)
                if res_B:
                    B, sum_mask_B, diff_mask_B = res_B
                    in_B[x] = True
                    improved = True
                    break
            
            if improved:
                continue
                
            # If no improvement, kick: remove a random element from A or B
            # and allow greedy to refill
            if A or B:
                if random.random() < 0.5 and A:
                    # Remove from A
                    x = random.choice(A)
                    A, sum_mask_A, diff_mask_A = try_remove(A, sum_mask_A, diff_mask_A, x)
                    in_A[x] = False
                elif B:
                    # Remove from B
                    x = random.choice(B)
                    B, sum_mask_B, diff_mask_B = try_remove(B, sum_mask_B, diff_mask_B, x)
                    in_B[x] = False
                else:
                    break
                    
                # Refill greedily
                for x in candidates:
                    if in_A[x] or in_B[x]:
                        continue
                    res_A = try_add(A, sum_mask_A, diff_mask_A, diff_mask_B, x, in_A)
                    if res_A:
                        A, sum_mask_A, diff_mask_A = res_A
                        in_A[x] = True
                        continue
                    res_B = try_add(B, sum_mask_B, diff_mask_B, diff_mask_A, x, in_B)
                    if res_B:
                        B, sum_mask_B, diff_mask_B = res_B
                        in_B[x] = True

        total_size = len(A) + len(B)
        if total_size > best_size:
            best_size = total_size
            best_sets = (A[:], B[:])
            
    return [best_sets[0], best_sets[1]]

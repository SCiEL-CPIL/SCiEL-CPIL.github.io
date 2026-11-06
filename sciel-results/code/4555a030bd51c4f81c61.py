import random
import time

def construct():
    n = 256
    max_time = 55.0
    start_time = time.time()
    
    best_size = 0
    best_A = []
    best_B = []
    
    # Bit offsets
    # Sum s in [2, 512]. Index s.
    # Diff d in [-255, 255]. Index d + 256.
    
    def get_sum_bits(x, S):
        """Calculate sum bits for adding x to set S."""
        bits = 1 << (2 * x)
        for s in S:
            bits |= 1 << (x + s)
        return bits

    def get_diff_bits(x, S):
        """Calculate diff bits for adding x to set S."""
        bits = 0
        for s in S:
            d1 = x - s
            d2 = s - x
            bits |= 1 << (d1 + 256)
            bits |= 1 << (d2 + 256)
        return bits

    def can_add(x, S, sum_mask, diff_mask_self, diff_mask_other):
        """Check if x can be added to S."""
        new_sums = get_sum_bits(x, S)
        if new_sums & sum_mask:
            return False
        new_diffs = get_diff_bits(x, S)
        if new_diffs & diff_mask_self:
            return False
        if new_diffs & diff_mask_other:
            return False
        return True

    def add_x(x, S, sum_mask, diff_mask_self):
        """Add x to S and update masks. Returns new masks."""
        new_sums = get_sum_bits(x, S)
        new_diffs = get_diff_bits(x, S)
        return S + [x], sum_mask | new_sums, diff_mask_self | new_diffs

    def remove_x(x, S, sum_mask, diff_mask_self):
        """Remove x from S and update masks. Returns new S, masks."""
        # Since Sidon, bits are unique to pairs.
        # We need to clear bits involving x.
        old_S = S
        S = [s for s in S if s != x]
        
        # Recalculate masks for the new S to be safe and simple, 
        # or clear specific bits. 
        # Given small size, recalculation is O(|S|^2) which is ~4000 ops. 
        # Clearing is also O(|S|). 
        # Let's clear specific bits to be faster.
        
        bits_to_clear_sums = 1 << (2 * x)
        bits_to_clear_diffs = 0
        for s in S:
            bits_to_clear_sums |= 1 << (x + s)
            d1 = x - s
            d2 = s - x
            bits_to_clear_diffs |= 1 << (d1 + 256)
            bits_to_clear_diffs |= 1 << (d2 + 256)
            
        new_sum_mask = sum_mask & ~bits_to_clear_sums
        new_diff_mask = diff_mask_self & ~bits_to_clear_diffs
        return S, new_sum_mask, new_diff_mask

    def greedy_construct():
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        A = []
        B = []
        sum_mask_A = 0
        diff_mask_A = 0
        sum_mask_B = 0
        diff_mask_B = 0
        
        used = set()
        
        for x in candidates:
            # Try A
            if can_add(x, A, sum_mask_A, diff_mask_A, diff_mask_B):
                A, sum_mask_A, diff_mask_A = add_x(x, A, sum_mask_A, diff_mask_A)
                used.add(x)
            
            # Try B
            # Note: if added to A, diff_mask_A changed. 
            # If not added to A, diff_mask_A is same.
            # We should try B even if added to A? 
            # The problem allows overlap. 
            # If we add to A, we use diff slots in A. 
            # Adding to B uses diff slots in B. 
            # The constraint is diff_A disjoint from diff_B.
            # So adding to A makes it harder to add to B (stricter check on diff_B vs diff_A).
            # But adding to B doesn't affect A's validity.
            # So we should try B independently.
            if can_add(x, B, sum_mask_B, diff_mask_B, diff_mask_A):
                B, sum_mask_B, diff_mask_B = add_x(x, B, sum_mask_B, diff_mask_B)
                used.add(x)
                
        return A, B, sum_mask_A, diff_mask_A, sum_mask_B, diff_mask_B, used

    def local_search(A, B, sum_mask_A, diff_mask_A, sum_mask_B, diff_mask_B, used):
        size = len(A) + len(B)
        no_improve_count = 0
        max_no_improve = 1000 # Threshold for restart
        
        while time.time() - start_time < max_time:
            if no_improve_count > max_no_improve:
                break
            
            # 1. Try to add a new element
            # Pick random x not in used
            if len(used) < n:
                available = [x for x in range(1, n+1) if x not in used]
                if available:
                    x = random.choice(available)
                    
                    # Try A
                    if can_add(x, A, sum_mask_A, diff_mask_A, diff_mask_B):
                        A, sum_mask_A, diff_mask_A = add_x(x, A, sum_mask_A, diff_mask_A)
                        used.add(x)
                        size += 1
                        no_improve_count = 0
                        continue
                        
                    # Try B
                    if can_add(x, B, sum_mask_B, diff_mask_B, diff_mask_A):
                        B, sum_mask_B, diff_mask_B = add_x(x, B, sum_mask_B, diff_mask_B)
                        used.add(x)
                        size += 1
                        no_improve_count = 0
                        continue
            
            # 2. Try 1-for-1 swap (remove y, add x)
            # This doesn't increase size, but might allow future additions.
            # However, the task is to MAXIMIZE size. 
            # Swaps that keep size constant are only useful if they lead to growth.
            # Given time constraints, maybe just focus on growth?
            # But greedy gets stuck. Swaps are needed to escape.
            
            # Let's try swap: remove y from A, add x to A.
            # We need x not in used? No, x can be in B? 
            # "A and B need not themselves be disjoint."
            # So x can be in B already.
            # If we move x from B to A, size doesn't change.
            # If we add x (new) to A and remove y from A, size doesn't change.
            # We want to increase size. 
            # So we should try: 
            #   Remove y from A. 
            #   Add x (new) to A. 
            #   This keeps size same. 
            #   Then try to add z (new) to A or B.
            
            # Simpler strategy: 
            # Try to add x to A. If fail, try removing a random y from A and adding x to A.
            # If that works, we have a new A. Then we can try to add more.
            
            if A:
                y = random.choice(A)
                # Check if x can be added to A \ {y}
                # We need to check if x collides with A \ {y}
                # And if (x - a) collides with diffs of A \ {y} or B
                # And if (a - x) collides...
                
                # Efficient check:
                # 1. Calculate bits for x in A\{y}
                # 2. Check against sum_mask_A (minus bits of y)
                # 3. Check against diff_mask_A (minus bits of y) and diff_mask_B
                
                # Let's compute the "clear" masks for y in A
                bits_y_sums = 1 << (2 * y)
                bits_y_diffs = 0
                for s in A:
                    if s == y: continue
                    bits_y_sums |= 1 << (y + s)
                    d1 = y - s
                    d2 = s - y
                    bits_y_diffs |= 1 << (d1 + 256)
                    bits_y_diffs |= 1 << (d2 + 256)
                    
                temp_sum_mask_A = sum_mask_A & ~bits_y_sums
                temp_diff_mask_A = diff_mask_A & ~bits_y_diffs
                
                # Now check if x fits in A' = A \ {y}
                # But we also need to ensure x doesn't collide with y? 
                # No, y is removed.
                
                if can_add(x, [s for s in A if s != y], temp_sum_mask_A, temp_diff_mask_A, diff_mask_B):
                    # Perform swap
                    A, sum_mask_A, diff_mask_A = add_x(x, [s for s in A if s != y], temp_sum_mask_A, temp_diff_mask_A)
                    # Note: x might already be in used? 
                    # If x was in used, it must be in B. 
                    # If x in B, and we add to A, it's still in used.
                    # Size doesn't change. 
                    # But this state might be better for future adds.
                    no_improve_count += 1 # It's a move, not an improvement in size
                    # Actually, if we just swap, we should reset no_improve? 
                    # No, we want to restart if we don't grow.
                    # Let's count swaps.
                    continue
            
            if B:
                y = random.choice(B)
                bits_y_sums = 1 << (2 * y)
                bits_y_diffs = 0
                for s in B:
                    if s == y: continue
                    bits_y_sums |= 1 << (y + s)
                    d1 = y - s
                    d2 = s - y
                    bits_y_diffs |= 1 << (d1 + 256)
                    bits_y_diffs |= 1 << (d2 + 256)
                    
                temp_sum_mask_B = sum_mask_B & ~bits_y_sums
                temp_diff_mask_B = diff_mask_B & ~bits_y_diffs
                
                if can_add(x, [s for s in B if s != y], temp_sum_mask_B, temp_diff_mask_B, diff_mask_A):
                    B, sum_mask_B, diff_mask_B = add_x(x, [s for s in B if s != y], temp_sum_mask_B, temp_diff_mask_B)
                    no_improve_count += 1
                    continue

            # If we got here, no add, no successful swap
            no_improve_count += 1
            
        return A, B, size

    # Main Loop
    while time.time() - start_time < max_time:
        A, B, sum_mask_A, diff_mask_A, sum_mask_B, diff_mask_B, used = greedy_construct()
        A, B, size = local_search(A, B, sum_mask_A, diff_mask_A, sum_mask_B, diff_mask_B, used)
        
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B

    return [best_A, best_B]

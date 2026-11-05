import time
import random

def construct():
    deadline = time.time() + 55.0  # safety margin before 90s kill
    best_size = 0
    best_A = []
    best_B = []
    n = 256
    max_each = 64

    def can_add(x, S, sums_S, diffs_S_bits, diffs_other_bits):
        # Check differences against other set's diffs using bitset
        for s in S:
            d = abs(x - s)
            if (diffs_other_bits >> d) & 1:
                return False
        # Check sums: x+s for s in S, and 2*x
        for s in S:
            if (x + s) in sums_S:
                return False
        if (2 * x) in sums_S:
            return False
        return True

    def add_to(x, S, sums_S, diffs_S_bits):
        new_bits = diffs_S_bits
        for s in S:
            sums_S.add(x + s)
            d = abs(x - s)
            new_bits |= (1 << d)
        sums_S.add(2 * x)
        S.append(x)
        return new_bits

    def remove_from(x, S, sums_S, diffs_S_bits):
        # We need to recompute diffs_S_bits if we remove, because 
        # a difference might be shared by multiple pairs? 
        # No, in a Sidon set, all non-zero differences are unique.
        # Proof: If a1-a2 = a3-a4, then a1+a4 = a2+a3. 
        # Since Sidon, {a1,a4} must be {a2,a3}. 
        # If a1=a2, then a4=a3, diff is 0 (excluded).
        # So each non-zero difference corresponds to exactly one pair {u,v} in S.
        # So we can just clear the bits for |x-s|.
        
        # Wait, is it possible that |x-s1| = |x-s2|? No, s1 != s2.
        # Is it possible that |x-s1| = |y-s2| for y != x?
        # If |x-s1| = |y-s2|, then this difference is in the set.
        # When removing x, we remove the differences contributed by x.
        # Since each difference in a Sidon set is unique to a pair, 
        # removing x removes exactly |S|-1 differences.
        # Are any of these differences also formed by pairs not involving x?
        # No, because if d = |x-s| = |u-v| with u,v != x, 
        # then we have two pairs with same diff, which is impossible in Sidon.
        # So we can safely clear the bits.
        
        idx = S.index(x)
        S.pop(idx)
        for s in S:
            sums_S.discard(x + s)
            d = abs(x - s)
            diffs_S_bits &= ~(1 << d)
        sums_S.discard(2 * x)
        return diffs_S_bits

    def local_search(A, B, sums_A, sums_B, diffs_A_bits, diffs_B_bits):
        improved = True
        # Limit iterations to avoid infinite loops or excessive time
        max_iters = 1000
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline - 2.0:
            improved = False
            iter_count += 1
            
            # Try 2-swaps: remove x from A, y from B, add u to A, v to B
            # Or simpler: Remove x from A, try to add u to A (swap in A)
            # Or: Remove x from A, try to add x to B (transfer)
            
            # 1. Try swapping within A
            for i in range(len(A)):
                x = A[i]
                # Remove x
                old_len_A = len(A)
                diffs_A_bits_tmp = remove_from(x, A, sums_A, diffs_A_bits)
                
                # Try to add any y to A
                added = False
                for y in range(1, n + 1):
                    if y in A or y in B:
                        continue
                    if can_add(y, A, sums_A, diffs_A_bits_tmp, diffs_B_bits):
                        diffs_A_bits_new = add_to(y, A, sums_A, diffs_A_bits_tmp)
                        # Note: add_to modifies A and sums_A
                        # We need to update diffs_A_bits to new value
                        diffs_A_bits = diffs_A_bits_new
                        added = True
                        break
                
                if not added:
                    # Restore x
                    diffs_A_bits = add_to(x, A, sums_A, diffs_A_bits_tmp)
                
                if added:
                    improved = True
                    break

            if improved:
                continue
                
            # 2. Try swapping within B
            for i in range(len(B)):
                x = B[i]
                old_len_B = len(B)
                diffs_B_bits_tmp = remove_from(x, B, sums_B, diffs_B_bits)
                
                added = False
                for y in range(1, n + 1):
                    if y in A or y in B:
                        continue
                    if can_add(y, B, sums_B, diffs_B_bits_tmp, diffs_A_bits):
                        diffs_B_bits_new = add_to(y, B, sums_B, diffs_B_bits_tmp)
                        diffs_B_bits = diffs_B_bits_new
                        added = True
                        break
                
                if not added:
                    diffs_B_bits = add_to(x, B, sums_B, diffs_B_bits_tmp)
                
                if added:
                    improved = True
                    break

            if improved:
                continue

            # 3. Try transferring x from A to B (and vice versa)
            # Remove x from A, try to add x to B
            for i in range(len(A)):
                x = A[i]
                diffs_A_bits_tmp = remove_from(x, A, sums_A, diffs_A_bits)
                
                if can_add(x, B, sums_B, diffs_B_bits, diffs_A_bits_tmp):
                    diffs_B_bits_new = add_to(x, B, sums_B, diffs_B_bits)
                    diffs_A_bits = diffs_A_bits_tmp
                    diffs_B_bits = diffs_B_bits_new
                    improved = True
                    break
                else:
                    diffs_A_bits = add_to(x, A, sums_A, diffs_A_bits_tmp)
            
            if improved:
                continue

            for i in range(len(B)):
                x = B[i]
                diffs_B_bits_tmp = remove_from(x, B, sums_B, diffs_B_bits)
                
                if can_add(x, A, sums_A, diffs_A_bits, diffs_B_bits_tmp):
                    diffs_A_bits_new = add_to(x, A, sums_A, diffs_A_bits)
                    diffs_A_bits = diffs_A_bits_new
                    diffs_B_bits = diffs_B_bits_tmp
                    improved = True
                    break
                else:
                    diffs_B_bits = add_to(x, B, sums_B, diffs_B_bits_tmp)

        return A, B, sums_A, sums_B, diffs_A_bits, diffs_B_bits

    # Multiple randomized restarts
    while time.time() < deadline - 2.0:
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A_bits = 0
        diffs_B_bits = 0
        
        # Randomize candidate order
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Greedy construction
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            ca = can_add(x, A, sums_A, diffs_A_bits, diffs_B_bits)
            cb = can_add(x, B, sums_B, diffs_B_bits, diffs_A_bits)
            
            if ca and cb:
                if len(A) < len(B):
                    diffs_A_bits = add_to(x, A, sums_A, diffs_A_bits)
                elif len(B) < len(A):
                    diffs_B_bits = add_to(x, B, sums_B, diffs_B_bits)
                else:
                    if random.random() < 0.5:
                        diffs_A_bits = add_to(x, A, sums_A, diffs_A_bits)
                    else:
                        diffs_B_bits = add_to(x, B, sums_B, diffs_B_bits)
            elif ca:
                diffs_A_bits = add_to(x, A, sums_A, diffs_A_bits)
            elif cb:
                diffs_B_bits = add_to(x, B, sums_B, diffs_B_bits)
        
        # Local search
        A, B, sums_A, sums_B, diffs_A_bits, diffs_B_bits = local_search(A, B, sums_A, sums_B, diffs_A_bits, diffs_B_bits)
        
        # Try more additions after local search
        order = list(range(1, n + 1))
        random.shuffle(order)
        # Re-implement try_add_any for bitsets
        for x in order:
            if x in A or x in B:
                continue
            ca = can_add(x, A, sums_A, diffs_A_bits, diffs_B_bits)
            cb = can_add(x, B, sums_B, diffs_B_bits, diffs_A_bits)
            if ca and cb:
                if len(A) < len(B):
                    diffs_A_bits = add_to(x, A, sums_A, diffs_A_bits)
                elif len(B) < len(A):
                    diffs_B_bits = add_to(x, B, sums_B, diffs_B_bits)
                else:
                    if random.random() < 0.5:
                        diffs_A_bits = add_to(x, A, sums_A, diffs_A_bits)
                    else:
                        diffs_B_bits = add_to(x, B, sums_B, diffs_B_bits)
            elif ca:
                diffs_A_bits = add_to(x, A, sums_A, diffs_A_bits)
            elif cb:
                diffs_B_bits = add_to(x, B, sums_B, diffs_B_bits)

        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)
            
            # If we've hit the theoretical max, stop early
            if best_size >= 24:
                break

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]

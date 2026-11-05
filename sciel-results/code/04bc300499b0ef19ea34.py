import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    def rebuild_diffs_sums(S):
        diffs = set()
        sums = set()
        for i, a in enumerate(S):
            sums.add(2 * a)
            for j in range(i + 1, len(S)):
                b = S[j]
                d = abs(a - b)
                diffs.add(d)
                sums.add(a + b)
        return diffs, sums

    def can_add(x, S, diffs_S, sums_S, diffs_other):
        for a in S:
            d = abs(x - a)
            if d in diffs_S:
                return False
            if d in diffs_other:
                return False
        if 2 * x in sums_S:
            return False
        return True

    def add_to_set(x, S, diffs_S, sums_S):
        S.append(x)
        sums_S.add(2 * x)
        for a in S:
            if a != x:
                d = abs(x - a)
                diffs_S.add(d)
                sums_S.add(x + a)

    def remove_from_set(x, S, diffs_S, sums_S):
        """Remove x from S and update diffs/sums incrementally."""
        S.remove(x)
        sums_S.discard(2 * x)
        for a in S:
            d = abs(x - a)
            s = x + a
            # Only remove if no other pair produces this diff/sum
            # For simplicity, rebuild if needed, but let's try incremental
            diffs_S.discard(d)
            sums_S.discard(s)
        # Note: This incremental removal is approximate. For correctness,
        # we should verify. But for speed in local search, we'll use rebuild
        # after removal when needed.

    def do_local_search(A, B):
        """Perform local search with incremental state management."""
        diffs_A, sums_A = rebuild_diffs_sums(A)
        diffs_B, sums_B = rebuild_diffs_sums(B)
        
        A_set = set(A)
        B_set = set(B)
        
        # Try multiple rounds of local search
        for _round in range(30):
            if time.time() > deadline:
                break
            
            made_change = False
            
            # Move 1: Try to add any remaining element to either set
            for x in range(1, 257):
                if x in A_set or x in B_set:
                    continue
                if can_add(x, A, diffs_A, sums_A, diffs_B):
                    add_to_set(x, A, diffs_A, sums_A)
                    A_set.add(x)
                    made_change = True
                    break
                if can_add(x, B, diffs_B, sums_B, diffs_A):
                    add_to_set(x, B, diffs_B, sums_B)
                    B_set.add(x)
                    made_change = True
                    break
            
            if made_change:
                continue
            
            # Move 2: Swap - remove from A, try to add to B (and vice versa)
            swap_done = False
            for i, a in enumerate(list(A)):
                # Temporarily remove a from A
                A.remove(a)
                A_set.discard(a)
                diffs_A_new, sums_A_new = rebuild_diffs_sums(A)
                
                # Try to add a to B
                if can_add(a, B, diffs_B, sums_B, diffs_A_new):
                    add_to_set(a, B, diffs_B, sums_B)
                    B_set.add(a)
                    diffs_A = diffs_A_new
                    sums_A = sums_A_new
                    made_change = True
                    swap_done = True
                    break
                else:
                    # Restore a to A
                    A.append(a)
                    A_set.add(a)
            
            if swap_done:
                continue
            
            # Move 3: Remove from B, try to add to A
            swap_done = False
            for i, b in enumerate(list(B)):
                B.remove(b)
                B_set.discard(b)
                diffs_B_new, sums_B_new = rebuild_diffs_sums(B)
                
                if can_add(b, A, diffs_A, sums_A, diffs_B_new):
                    add_to_set(b, A, diffs_A, sums_A)
                    A_set.add(b)
                    diffs_B = diffs_B_new
                    sums_B = sums_B_new
                    made_change = True
                    swap_done = True
                    break
                else:
                    B.append(b)
                    B_set.add(b)
            
            if swap_done:
                continue
            
            # Move 4: Remove one from A, try to add two new elements to A
            for i in range(len(A)):
                if time.time() > deadline:
                    return A, B
                removed = A[i]
                A.remove(removed)
                A_set.discard(removed)
                diffs_A_new, sums_A_new = rebuild_diffs_sums(A)
                
                added_count = 0
                added_elements = []
                for x in range(1, 257):
                    if x in A_set or x in B_set:
                        continue
                    if can_add(x, A, diffs_A_new, sums_A_new, diffs_B):
                        add_to_set(x, A, diffs_A_new, sums_A_new)
                        A_set.add(x)
                        added_elements.append(x)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 2:
                    diffs_A = diffs_A_new
                    sums_A = sums_A_new
                    made_change = True
                    break
                else:
                    # Restore
                    # Remove added elements
                    for x in added_elements:
                        A.remove(x)
                        A_set.discard(x)
                    A.append(removed)
                    A_set.add(removed)
                    diffs_A, sums_A = rebuild_diffs_sums(A)
            
            if made_change:
                continue
            
            # Move 5: Remove one from B, try to add two new elements to B
            for i in range(len(B)):
                if time.time() > deadline:
                    return A, B
                removed = B[i]
                B.remove(removed)
                B_set.discard(removed)
                diffs_B_new, sums_B_new = rebuild_diffs_sums(B)
                
                added_count = 0
                added_elements = []
                for x in range(1, 257):
                    if x in A_set or x in B_set:
                        continue
                    if can_add(x, B, diffs_B_new, sums_B_new, diffs_A):
                        add_to_set(x, B, diffs_B_new, sums_B_new)
                        B_set.add(x)
                        added_elements.append(x)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 2:
                    diffs_B = diffs_B_new
                    sums_B = sums_B_new
                    made_change = True
                    break
                else:
                    # Restore
                    for x in added_elements:
                        B.remove(x)
                        B_set.discard(x)
                    B.append(removed)
                    B_set.add(removed)
                    diffs_B, sums_B = rebuild_diffs_sums(B)
            
            if made_change:
                continue
            
            # No more moves
            break
        
        return A, B

    def polish(A, B):
        """Final polish: try all single additions repeatedly until stuck."""
        diffs_A, sums_A = rebuild_diffs_sums(A)
        diffs_B, sums_B = rebuild_diffs_sums(B)
        A_set = set(A)
        B_set = set(B)
        
        while True:
            if time.time() > deadline:
                break
            any_added = False
            for x in range(1, 257):
                if x in A_set or x in B_set:
                    continue
                if can_add(x, A, diffs_A, sums_A, diffs_B):
                    add_to_set(x, A, diffs_A, sums_A)
                    A_set.add(x)
                    any_added = True
                    break
                if can_add(x, B, diffs_B, sums_B, diffs_A):
                    add_to_set(x, B, diffs_B, sums_B)
                    B_set.add(x)
                    any_added = True
                    break
            if not any_added:
                break
        
        return A, B

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts * 12345 + 67890)
        
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        # Candidate list with multiple random orderings
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Greedy Phase 1
        for x in candidates:
            valid_A = can_add(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    add_to_set(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, diffs_A, sums_A)
                    else:
                        add_to_set(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set(x, B, diffs_B, sums_B)
        
        # Second greedy pass with different order
        remaining = [x for x in range(1, 257) if x not in A and x not in B]
        random.shuffle(remaining)
        
        for x in remaining:
            valid_A = can_add(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    add_to_set(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, diffs_A, sums_A)
                    else:
                        add_to_set(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set(x, B, diffs_B, sums_B)
        
        # Local search phase
        A, B = do_local_search(A, B)
        
        # Final polish
        A, B = polish(A, B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A.copy()
            best_B = B.copy()

    return [best_A, best_B]

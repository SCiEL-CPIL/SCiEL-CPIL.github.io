import time
import random

def _check_sums_valid(x, S, sums):
    if x + x in sums:
        return False
    for a in S:
        if x + a in sums:
            return False
    return True

def _check_diffs_valid(x, S, other_diffs):
    for a in S:
        d1 = x - a
        d2 = a - x
        if d1 in other_diffs or d2 in other_diffs:
            return False
    return True

def _add_to_set(x, S, sums, diffs):
    S.append(x)
    sums.add(x + x)
    for a in S[:-1]:
        sums.add(x + a)
        diffs.add(x - a)
        diffs.add(a - x)

def _remove_from_set(x, S, sums, diffs):
    # Remove x from S
    S.remove(x)
    # Update sums and diffs
    # We need to remove sums/diffs involving x
    # This is tricky because sums/diffs are sets. 
    # We can't easily remove just the ones involving x without knowing which ones they are.
    # However, since we only remove and then potentially add, 
    # and we want to check validity of *other* elements, 
    # the current sums/diffs sets contain "extra" entries for x.
    # This will make validity checks for *other* elements stricter (false negatives).
    # To do this correctly, we should rebuild or track counts.
    # Given N=256, rebuilding is O(k^2) which is fast.
    # But we are doing this in a loop.
    # Alternative: Don't remove. Just simulate.
    # Or: Rebuild sums and diffs for the modified set.
    
    # For simplicity and correctness in the local search, 
    # we will implement a helper to rebuild sums/diffs from a list.
    pass

def _rebuild_sums_diffs(S):
    sums = set()
    diffs = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
        for j in range(len(S)):
            if i != j:
                diffs.add(S[i] - S[j])
    return sums, diffs

def _local_improve_swap(A, B, sums_A, sums_B, diffs_A, diffs_B, start_time, timeout):
    """
    Attempt to improve by removing an element from one set and adding a better one.
    """
    improved = True
    while improved:
        if time.time() - start_time > timeout:
            break
        improved = False
        current_size = len(A) + len(B)
        all_used = set(A) | set(B)
        candidates = [x for x in range(1, 257) if x not in all_used]
        
        # Try removing from A
        for i in range(len(A)):
            if time.time() - start_time > timeout:
                break
            a_to_remove = A[i]
            
            # Simulate removal
            A_temp = A[:]
            A_temp.remove(a_to_remove)
            sums_A_temp, diffs_A_temp = _rebuild_sums_diffs(A_temp)
            
            # Try to add any candidate to A or B
            best_add = None
            best_side = None
            best_size_gain = 0
            
            for x in candidates:
                # Check if x fits in A_temp
                valid_A = _check_sums_valid(x, A_temp, sums_A_temp) and _check_diffs_valid(x, A_temp, diffs_B)
                # Check if x fits in B
                valid_B = _check_sums_valid(x, B, sums_B) and _check_diffs_valid(x, B, diffs_A_temp)
                
                if valid_A and not valid_B:
                    # Adding to A increases size by 1 (since we removed 1)
                    # Net change 0? No, we removed 1, added 1. Size is same.
                    # But maybe it allows *future* additions?
                    # For now, we just want to find a valid swap.
                    # If we find a valid swap, we accept it to change the landscape.
                    # To prefer "better" swaps, we could look ahead, but that's expensive.
                    # Let's just accept the first valid swap found, or the one that adds to the smaller set.
                    if len(A_temp) < len(B):
                        best_add = x
                        best_side = 'A'
                        break
                elif valid_B and not valid_A:
                    if len(B) < len(A_temp):
                        best_add = x
                        best_side = 'B'
                        break
                elif valid_A and valid_B:
                    # Prefer smaller set
                    if len(A_temp) <= len(B):
                        best_add = x
                        best_side = 'A'
                        break
                    else:
                        best_add = x
                        best_side = 'B'
                        break
            
            if best_add is not None:
                # Commit the change
                A = A_temp
                sums_A = sums_A_temp
                diffs_A = diffs_A_temp
                
                if best_side == 'A':
                    _add_to_set(best_add, A, sums_A, diffs_A)
                    candidates.remove(best_add)
                else:
                    _add_to_set(best_add, B, sums_B, diffs_B)
                    candidates.remove(best_add)
                
                improved = True
                # Restart the while loop to re-evaluate
                break
        
        if improved:
            continue
            
        # Try removing from B
        for i in range(len(B)):
            if time.time() - start_time > timeout:
                break
            b_to_remove = B[i]
            
            # Simulate removal
            B_temp = B[:]
            B_temp.remove(b_to_remove)
            sums_B_temp, diffs_B_temp = _rebuild_sums_diffs(B_temp)
            
            # Try to add any candidate to A or B_temp
            best_add = None
            best_side = None
            
            for x in candidates:
                valid_A = _check_sums_valid(x, A, sums_A) and _check_diffs_valid(x, A, diffs_B_temp)
                valid_B = _check_sums_valid(x, B_temp, sums_B_temp) and _check_diffs_valid(x, B_temp, diffs_A)
                
                if valid_A and not valid_B:
                    if len(A) < len(B_temp):
                        best_add = x
                        best_side = 'A'
                        break
                elif valid_B and not valid_A:
                    if len(B_temp) < len(A):
                        best_add = x
                        best_side = 'B'
                        break
                elif valid_A and valid_B:
                    if len(A) <= len(B_temp):
                        best_add = x
                        best_side = 'A'
                        break
                    else:
                        best_add = x
                        best_side = 'B'
                        break
            
            if best_add is not None:
                B = B_temp
                sums_B = sums_B_temp
                diffs_B = diffs_B_temp
                
                if best_side == 'A':
                    _add_to_set(best_add, A, sums_A, diffs_A)
                    candidates.remove(best_add)
                else:
                    _add_to_set(best_add, B, sums_B, diffs_B)
                    candidates.remove(best_add)
                
                improved = True
                break
        
    return len(A) + len(B), A, B, sums_A, sums_B, diffs_A, diffs_B

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    best_size = 0
    best_A = []
    best_B = []
    
    max_restarts = 5000

    for _ in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Candidates
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        # Bias towards smaller numbers?
        # Sidon sets often pack better with smaller numbers if we are careful.
        # Let's try a mix.
        if random.random() < 0.5:
             candidates.sort()
             # Add some noise
             for i in range(len(candidates)):
                 if random.random() < 0.2:
                     j = random.randint(0, len(candidates)-1)
                     candidates[i], candidates[j] = candidates[j], candidates[i]
        else:
             random.shuffle(candidates)

        for x in candidates:
            if time.time() - start_time > timeout:
                break
            
            valid_A = _check_sums_valid(x, A, sums_A) and _check_diffs_valid(x, A, diffs_B)
            valid_B = _check_sums_valid(x, B, sums_B) and _check_diffs_valid(x, B, diffs_A)
            
            if valid_A and valid_B:
                # Heuristic: Put in the set with fewer elements to balance
                if len(A) < len(B):
                    side = 'A'
                elif len(B) < len(A):
                    side = 'B'
                else:
                    # If balanced, pick randomly or based on "cost"
                    side = random.choice(['A', 'B'])
            elif valid_A:
                side = 'A'
            elif valid_B:
                side = 'B'
            else:
                continue
            
            if side == 'A':
                _add_to_set(x, A, sums_A, diffs_A)
            else:
                _add_to_set(x, B, sums_B, diffs_B)
        
        # Local improvement 1: Add remaining
        all_used = set(A) | set(B)
        remaining = [x for x in range(1, 257) if x not in all_used]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() - start_time > timeout:
                break
            valid_A = _check_sums_valid(x, A, sums_A) and _check_diffs_valid(x, A, diffs_B)
            valid_B = _check_sums_valid(x, B, sums_B) and _check_diffs_valid(x, B, diffs_A)
            
            if valid_A and not valid_B:
                _add_to_set(x, A, sums_A, diffs_A)
            elif valid_B and not valid_A:
                _add_to_set(x, B, sums_B, diffs_B)
            elif valid_A and valid_B:
                if len(A) <= len(B):
                    _add_to_set(x, A, sums_A, diffs_A)
                else:
                    _add_to_set(x, B, sums_B, diffs_B)
        
        # Local improvement 2: Swap
        current_size, A, B, sums_A, sums_B, diffs_A, diffs_B = _local_improve_swap(A, B, sums_A, sums_B, diffs_A, diffs_B, start_time, timeout)
        
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]

    return [best_A, best_B]

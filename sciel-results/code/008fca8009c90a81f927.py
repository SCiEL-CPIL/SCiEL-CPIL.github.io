import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    max_val = 284
    target_size = 20  # Theoretical max without constraints, aim for this or close
    
    # State
    current_set = prescribed[:]
    current_sums = set()
    
    # Helper to compute all pairwise sums for a sorted list
    def compute_sums(marks):
        sums = set()
        n = len(marks)
        for i in range(n):
            for j in range(i, n):  # a <= b, so i <= j
                s = marks[i] + marks[j]
                if s in sums:
                    return None # Invalid Sidon set
                sums.add(s)
        return sums

    # Initialize sums
    current_sums = compute_sums(current_set)
    
    # Best solution found so far
    best_set = current_set[:]
    best_size = len(best_set)

    # Candidate pool: all numbers in [1, 284] not in prescribed
    all_candidates = [x for x in range(1, max_val + 1) if x not in prescribed]
    
    # Function to check if x can be added
    def can_add(x, marks, sums):
        for m in marks:
            if m > x:
                continue
            # Sum is x + m
            s = x + m
            if s in sums:
                return False
        return True

    # Function to add x to marks and sums
    def add_element(x, marks, sums):
        bisect.insort(marks, x)
        for m in marks:
            if m > x:
                continue
            sums.add(x + m)

    # Function to remove x from marks and sums
    # Note: Removing sums is tricky if sums are shared. 
    # However, in a valid Sidon set, each sum is unique to a specific pair.
    # So we can safely remove all sums involving x.
    def remove_element(x, marks, sums):
        # Remove sums involving x
        for m in marks:
            if m > x:
                continue
            s = x + m
            if s in sums:
                sums.remove(s)
        # Remove x from list
        idx = bisect.bisect_left(marks, x)
        if idx < len(marks) and marks[idx] == x:
            marks.pop(idx)

    # Main search loop
    while time.time() < deadline:
        # If we reached the theoretical max, we can stop
        if len(current_set) >= target_size:
            break
            
        # Phase 1: Try to add any valid element
        added = False
        # Shuffle candidates to avoid bias
        random.shuffle(all_candidates)
        for x in all_candidates:
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                add_element(x, current_set, current_sums)
                if len(current_set) > best_size:
                    best_set = current_set[:]
                    best_size = len(best_set)
                added = True
                break
        
        if added:
            continue
            
        # Phase 2: Stuck. Perform a "kick" (swap)
        # Remove a random non-prescribed element and try to add a different one
        non_prescribed_in_set = [x for x in current_set if x not in prescribed]
        
        if not non_prescribed_in_set:
            # We only have prescribed elements and can't add anything else?
            # This is unlikely given the range, but if it happens, break.
            break
            
        # Pick a random element to remove
        x_to_remove = random.choice(non_prescribed_in_set)
        remove_element(x_to_remove, current_set, current_sums)
        
        # Try to add a new element
        swap_success = False
        random.shuffle(all_candidates)
        for x in all_candidates:
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                add_element(x, current_set, current_sums)
                swap_success = True
                break
                
        if not swap_success:
            # Revert the removal to keep the set valid (even if size dropped temporarily)
            # Actually, if we can't add anything, the set with x_to_remove was better or equal.
            # We should put it back.
            add_element(x_to_remove, current_set, current_sums)
            # To avoid infinite loop of same kick, we could randomize more, 
            # but for now, just continue. The shuffle of candidates changes the next attempt.
            
    return best_set

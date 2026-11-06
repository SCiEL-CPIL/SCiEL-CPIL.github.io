import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Initial set: Seed
    A = list(range(64, 129))
    n = 128
    # counts[s] = number of pairs {a,b} in A such that a+b=s
    counts = [0] * (2 * n + 1)
    
    # Helper to update counts for adding/removing a pair
    # We will manage counts manually for speed
    
    # Initialize counts
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            counts[s] += 1
            
    # Calculate metrics
    def get_metrics():
        max_mult = 0
        sq_sum = 0
        for s in range(128, 257):
            c = counts[s]
            if c > max_mult:
                max_mult = c
            sq_sum += c * c
        return max_mult, sq_sum

    curr_max, curr_sq = get_metrics()
    best_max = curr_max
    best_sq = curr_sq
    best_A = A.copy()
    
    # Available elements for swapping
    in_A = [False] * (n + 1)
    for x in A:
        in_A[x] = True
    avail = [x for x in range(n + 1) if not in_A[x]]
    
    T = 1.0
    cool_rate = 0.99
    
    while time.time() < deadline:
        # Pick random old and new
        idx_old = random.randint(0, len(A) - 1)
        a_old = A[idx_old]
        
        idx_new = random.randint(0, len(avail) - 1)
        a_new = avail[idx_new]
        
        if a_old == a_new:
            continue
            
        # Calculate changes
        # Pairs to remove: (a_old, a_old) and (a_old, x) for x in A, x != a_old
        # Pairs to add: (a_new, a_new) and (a_new, x) for x in A, x != a_old
        
        # We need to check coverage.
        # A move is valid if for all s in [128, 256], new_count[s] > 0.
        
        # To avoid modifying counts and then reverting, we can calculate the delta
        # and check validity on the fly.
        
        # However, checking validity requires knowing the new counts.
        # Let's simulate the counts change.
        
        # List of sums affected
        # Sums involving a_old:
        # 2*a_old
        # a_old + x for x in A, x != a_old
        
        # Sums involving a_new:
        # 2*a_new
        # a_new + x for x in A, x != a_old
        
        # Note: If a_new is in A? No, a_new is from avail, so not in A.
        # So A_new = A \ {a_old} U {a_new}
        
        # Let's collect the sums that will change.
        # We need to check if any count in [128, 256] becomes 0.
        
        # Optimization: Only check sums that are currently 1 and will be decremented.
        # If a sum S is currently 1, and we decrement it, it becomes 0 -> Invalid.
        
        # So, we just need to check:
        # 1. Does 2*a_old fall in [128, 256] and counts[2*a_old] == 1? If yes, invalid.
        # 2. For each x in A (x != a_old), does a_old + x fall in [128, 256] and counts[a_old + x] == 1? If yes, invalid.
        
        # If valid, apply changes.
        
        valid = True
        # Check 2*a_old
        s = 2 * a_old
        if 128 <= s <= 256 and counts[s] == 1:
            valid = False
            
        if valid:
            for x in A:
                if x == a_old:
                    continue
                s = a_old + x
                if 128 <= s <= 256 and counts[s] == 1:
                    valid = False
                    break
                    
        if not valid:
            continue
            
        # Apply changes
        # Remove a_old
        # 2*a_old
        counts[2 * a_old] -= 1
        for x in A:
            if x == a_old:
                continue
            s = a_old + x
            counts[s] -= 1
            
        # Add a_new
        counts[2 * a_new] += 1
        for x in A:
            if x == a_old:
                continue
            s = a_new + x
            counts[s] += 1
            
        # Update structures
        A[idx_old] = a_new
        in_A[a_old] = False
        in_A[a_new] = True
        avail[idx_new] = a_old # Swap in avail list? 
        # Wait, avail list logic:
        # We picked a_new from avail. We remove it from avail.
        # We add a_old to avail.
        # So avail[idx_new] = a_old is correct if we just swap the value.
        
        # Calculate new metrics
        new_max, new_sq = get_metrics()
        
        # Acceptance
        accept = False
        if new_max < best_max:
            accept = True
            best_max = new_max
            best_sq = new_sq
            best_A = A.copy()
        elif new_max == best_max:
            if new_sq < best_sq:
                accept = True
                best_sq = new_sq
                best_A = A.copy()
            else:
                # SA step
                delta_sq = new_sq - best_sq
                if delta_sq < 0 or random.random() < math.exp(-delta_sq / T):
                    accept = True
                    best_sq = new_sq # Update current best sq? 
                    # Actually, we should track current state separately from best state
                    # If we reject, we must revert.
                    # If we accept, we update current state.
                    # Let's simplify: 
                    # If we accept, we keep the change.
                    # If we reject, we revert.
                    
                    # Let's re-structure to keep track of current_max, current_sq
                    # and best_max, best_sq.
                    
                    # For now, let's just accept if it improves or is neutral with prob.
                    # This is getting complicated. Let's use a simpler heuristic:
                    # Accept if new_max <= curr_max.
                    # If new_max < curr_max, update curr_max.
                    # If new_max == curr_max, accept with prob based on sq_sum.
                    
                    pass # Placeholder for proper SA logic
        
        # Proper SA Logic Implementation:
        # We need to revert if rejected.
        
        # Let's restart the loop body logic with proper revert.
        
        # This draft is getting messy. I will write the final code carefully.
        
        T *= cool_rate
        
    return best_A

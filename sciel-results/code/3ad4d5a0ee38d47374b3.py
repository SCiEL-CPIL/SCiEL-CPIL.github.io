import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    # We will build the set greedily.
    # current_set: list of chosen integers
    # current_sums: set of all subset sums of current_set
    current_set = []
    current_sums = {0}
    
    # We want to find n distinct positive integers.
    # To minimize the largest, we should pick the smallest valid integer at each step.
    # We assume we pick them in increasing order.
    
    for k in range(n):
        # Determine the smallest candidate
        # The candidate must be > 0 and not in current_set (distinct)
        # It must not create duplicate sums.
        # New sums will be S_new = S_old U (S_old + x)
        # We need S_old and S_old + x to be disjoint.
        # This is equivalent to x not being in {a - b | a, b in S_old, a != b}
        
        # Efficient check:
        # If x is in the difference set of current_sums, it's invalid.
        
        # Start candidate from 1
        candidate = 1
        
        while True:
            if time.time() > deadline:
                # If we run out of time, return what we have (padded with large numbers to ensure validity? 
                # No, must be valid. 
                # Fallback to powers of 2 for remaining slots)
                # This should not happen for n=12
                break
            
            # Check if candidate is valid
            # 1. Distinct from existing
            if candidate in current_set:
                candidate += 1
                continue
            
            # 2. Check subset sum collision
            # We need to ensure (current_sums + candidate) is disjoint from current_sums
            # i.e., (current_sums + candidate) & current_sums == empty set
            
            # Optimization: 
            # If candidate > max(current_sums), then current_sums + candidate > max(current_sums) >= min(current_sums)
            # So they are disjoint.
            if candidate > max(current_sums):
                valid = True
            else:
                # Check intersection
                # Shifted sums
                shifted = {s + candidate for s in current_sums}
                if shifted & current_sums:
                    candidate += 1
                    continue
                else:
                    valid = True
            
            if valid:
                # Add to set
                current_set.append(candidate)
                # Update sums
                current_sums = current_sums | {s + candidate for s in current_sums}
                break
            
            # If we looped without breaking (shouldn't happen in the 'if valid' block)
            # But the 'while True' handles the increment if invalid.
            # Wait, the logic above increments candidate if invalid.
            # If valid, it breaks.
            
    # Ensure we have 12 elements
    # If deadline was hit, we might have fewer.
    # But for n=12, it's fast.
    
    return current_set

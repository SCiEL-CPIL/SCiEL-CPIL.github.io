import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop at 110s to be very safe.
    deadline = start_time + 110.0
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Helper to check if a candidate x can be added to the current set
    def can_add(x, a_set, sums_set):
        if x in a_set:
            return False
        # Check sums with existing elements
        for y in a_set:
            s = x + y
            if s in sums_set:
                return False
        # Check double sum
        if 2 * x in sums_set:
            return False
        return True

    def get_sums(a_list):
        s = set()
        for i, x in enumerate(a_list):
            for y in a_list[i:]:
                s.add(x + y)
        return s

    # Initialize with fixed elements
    current_set = set(fixed_elements)
    current_sums = get_sums(list(fixed_elements))
    best_set = set(fixed_elements)
    best_size = len(best_set)
    
    # Pre-compute all possible candidates (1..284)
    all_candidates = list(range(1, 285))
    
    while time.time() < deadline:
        # 1. Try to add elements using a randomized greedy pass
        # Create a random permutation of candidates not in current_set
        candidates = [x for x in all_candidates if x not in current_set]
        random.shuffle(candidates)
        
        added_any = False
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if can_add(x, current_set, current_sums):
                # Add x
                current_set.add(x)
                # Update sums: add x+y for all y in current_set (including x? No, x is not in set yet)
                # Wait, can_add checks against current_set.
                # So we add x to set, then add sums involving x and all other elements in set.
                # Note: x is now in current_set.
                # We need to add sums: x + y for all y in current_set (including x? No, 2x is separate or included if we iterate carefully).
                # Let's iterate over a snapshot of the set before adding x? 
                # Actually, can_add already verified x is not in set.
                # So:
                for y in current_set:
                    current_sums.add(x + y)
                # Note: when y=x, x+x=2x is added.
                # But x was just added to current_set. So the loop includes y=x.
                # This is correct.
                
                added_any = True
                # Update best
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        # 2. If we couldn't add anything in this pass, we are in a local optimum.
        if not added_any:
            non_fixed = [x for x in current_set if x not in fixed_set]
            
            if not non_fixed:
                # We are at the base size. We can't kick.
                # Try a few more random restarts? Or just break.
                # Since we can't improve, break.
                break
            
            # Perform a kick: remove 1 or 2 random non-fixed elements
            # Removing too many might be too costly, removing too few might not help.
            # Let's try removing 1 element first. If that fails repeatedly, try 2.
            # Simple strategy: remove 1 random non-fixed element.
            
            k = 1
            if random.random() < 0.2:
                k = 2
            if random.random() < 0.05:
                k = 3
            
            # Select elements to remove
            to_remove = random.sample(non_fixed, min(k, len(non_fixed)))
            
            # Remove them
            for x in to_remove:
                current_set.remove(x)
                # Remove sums involving x
                # Sums to remove: {x + y for y in current_set} U {2x}
                # Note: current_set no longer contains x.
                # So x+y for y in current_set covers all cross sums.
                # 2x is not covered if x is not in current_set.
                # So we add 2x to the removal list.
                sums_to_remove = set()
                for y in current_set:
                    sums_to_remove.add(x + y)
                sums_to_remove.add(2 * x)
                
                for s in sums_to_remove:
                    if s in current_sums:
                        current_sums.remove(s)
                        
    return list(best_set)

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found size: {len(res)}")
    # Verify
    s = set()
    valid = True
    for i in range(len(res)):
        for j in range(i, len(res)):
            sm = res[i] + res[j]
            if sm in s:
                valid = False
            s.add(sm)
    print(f"Valid: {valid}")
    print(f"Contains fixed: {set([1,2,5,11]).issubset(set(res))}")
